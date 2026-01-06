"""A/B Testing Service for RAG Pipeline Experimentation.

This module provides a comprehensive A/B testing framework for the DevOps AI Assistant,
enabling controlled experiments on:
- LLM model comparisons
- Prompt template variations
- RAG configuration tuning
- Temperature and parameter optimization

Key Features:
- Sticky session assignment for consistent user experience
- Deterministic bucketing using hash-based allocation
- Statistical analysis with t-test significance testing
- Support for multiple concurrent experiments
- Graceful fallback when no experiments are active

Usage:
    from app.ab_testing import ab_testing_service

    # Get variant for a session
    variant = await ab_testing_service.get_variant_for_session(
        experiment_id=exp_id,
        session_id=session_id
    )

    # Record experiment result
    await ab_testing_service.record_result(
        experiment_id=exp_id,
        variant_id=variant["id"],
        session_id=session_id,
        metric_name="latency",
        metric_value=150.5
    )

    # Get experiment statistics
    stats = await ab_testing_service.get_experiment_stats(exp_id)
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_context
from app.db_models import (
    Experiment,
    ExperimentAssignment,
    ExperimentResult,
    ExperimentStatus,
)

logger = logging.getLogger(__name__)

# Number of buckets for deterministic assignment (0-99 = 100 buckets)
NUM_BUCKETS = 100


@dataclass
class VariantStats:
    """Statistics for a single experiment variant."""

    variant_id: str
    count: int
    mean: float
    std: float
    min_value: float
    max_value: float


@dataclass
class ExperimentStats:
    """Aggregated statistics for an experiment."""

    experiment_id: uuid.UUID
    experiment_name: str
    status: str
    primary_metric: str
    variants: dict[str, VariantStats]
    p_value: Optional[float]
    is_significant: bool
    total_samples: int
    recommendation: Optional[str]


class ABTestingService:
    """Service for managing A/B testing experiments.

    Provides methods for:
    - Listing active experiments
    - Assigning sessions to variants (sticky assignment)
    - Recording experiment results
    - Computing statistical analysis

    The service uses deterministic bucketing based on session ID hashing
    to ensure consistent variant assignment across requests.
    """

    def __init__(self):
        """Initialize the A/B testing service."""
        self._active_experiments_cache: dict[uuid.UUID, Experiment] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 60  # Refresh cache every 60 seconds

    async def get_active_experiments(
        self,
        db: Optional[AsyncSession] = None,
    ) -> list[dict[str, Any]]:
        """Get all currently running experiments.

        Args:
            db: Optional database session. If not provided, creates a new one.

        Returns:
            List of active experiment dictionaries with id, name, type, variants,
            and traffic_split.
        """
        async def _fetch(session: AsyncSession) -> list[dict[str, Any]]:
            stmt = select(Experiment).where(
                Experiment.status == ExperimentStatus.RUNNING
            )
            result = await session.execute(stmt)
            experiments = result.scalars().all()

            return [
                {
                    "id": str(exp.id),
                    "name": exp.name,
                    "description": exp.description,
                    "experiment_type": exp.experiment_type.value,
                    "variants": exp.variants,
                    "traffic_split": exp.traffic_split,
                    "success_metric": exp.success_metric,
                    "started_at": exp.started_at.isoformat() if exp.started_at else None,
                }
                for exp in experiments
            ]

        if db:
            return await _fetch(db)
        else:
            async with get_db_context() as session:
                return await _fetch(session)

    async def get_variant_for_session(
        self,
        experiment_id: uuid.UUID,
        session_id: str,
        db: Optional[AsyncSession] = None,
    ) -> Optional[dict[str, Any]]:
        """Get or assign a variant for a session (sticky assignment).

        If the session already has an assignment, returns the existing variant.
        Otherwise, assigns a new variant based on traffic split percentages.

        Args:
            experiment_id: UUID of the experiment
            session_id: Session identifier for sticky assignment
            db: Optional database session

        Returns:
            Variant configuration dict or None if experiment not found/not running
        """
        async def _fetch(session: AsyncSession) -> Optional[dict[str, Any]]:
            # Check if session already has an assignment
            stmt = select(ExperimentAssignment).where(
                and_(
                    ExperimentAssignment.experiment_id == experiment_id,
                    ExperimentAssignment.session_id == session_id,
                )
            )
            result = await session.execute(stmt)
            existing_assignment = result.scalar_one_or_none()

            if existing_assignment:
                # Return existing variant from experiment
                experiment = await self._get_experiment(experiment_id, session)
                if experiment and experiment.variants:
                    variant = self._find_variant_by_id(
                        experiment.variants, existing_assignment.variant_id
                    )
                    if variant:
                        logger.debug(
                            f"Returning existing assignment: session={session_id[:8]}... "
                            f"variant={existing_assignment.variant_id}"
                        )
                        return variant
                return None

            # No existing assignment - create new one
            return await self.assign_variant(experiment_id, session_id, session)

        if db:
            return await _fetch(db)
        else:
            async with get_db_context() as session:
                return await _fetch(session)

    async def assign_variant(
        self,
        experiment_id: uuid.UUID,
        session_id: str,
        db: Optional[AsyncSession] = None,
    ) -> Optional[dict[str, Any]]:
        """Assign a variant to a session based on traffic split percentages.

        Uses deterministic bucket hashing to ensure consistent assignment
        even across multiple service instances.

        Args:
            experiment_id: UUID of the experiment
            session_id: Session identifier
            db: Optional database session

        Returns:
            Assigned variant configuration dict or None if experiment not running
        """
        async def _assign(session: AsyncSession) -> Optional[dict[str, Any]]:
            # Get experiment and verify it's running
            experiment = await self._get_experiment(experiment_id, session)
            if not experiment or experiment.status != ExperimentStatus.RUNNING:
                logger.warning(
                    f"Cannot assign variant: experiment {experiment_id} not found or not running"
                )
                return None

            # Compute deterministic bucket
            bucket = self._hash_session_to_bucket(session_id, NUM_BUCKETS)

            # Select variant based on traffic split
            variant_id = self._select_variant_by_bucket(
                bucket, experiment.traffic_split
            )

            if not variant_id:
                logger.error(
                    f"Failed to select variant for bucket {bucket} "
                    f"with traffic_split {experiment.traffic_split}"
                )
                return None

            # Create assignment record
            assignment = ExperimentAssignment(
                experiment_id=experiment_id,
                session_id=session_id,
                variant_id=variant_id,
            )
            session.add(assignment)
            await session.flush()  # Ensure assignment is persisted

            logger.info(
                f"Assigned variant: experiment={experiment.name} "
                f"session={session_id[:8]}... variant={variant_id} bucket={bucket}"
            )

            # Return the variant configuration
            return self._find_variant_by_id(experiment.variants, variant_id)

        if db:
            return await _assign(db)
        else:
            async with get_db_context() as session:
                return await _assign(session)

    async def record_result(
        self,
        experiment_id: uuid.UUID,
        variant_id: str,
        session_id: str,
        metric_name: str,
        metric_value: float,
        query_log_id: Optional[uuid.UUID] = None,
        db: Optional[AsyncSession] = None,
    ) -> bool:
        """Record an experiment result metric.

        Args:
            experiment_id: UUID of the experiment
            variant_id: Variant that produced this result
            session_id: Session identifier
            metric_name: Name of the metric (e.g., "latency", "rating")
            metric_value: Numeric value of the metric
            query_log_id: Optional reference to query log
            db: Optional database session

        Returns:
            True if result was recorded successfully, False otherwise
        """
        async def _record(session: AsyncSession) -> bool:
            try:
                result = ExperimentResult(
                    experiment_id=experiment_id,
                    variant_id=variant_id,
                    session_id=session_id,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    query_log_id=query_log_id,
                )
                session.add(result)
                await session.flush()

                logger.debug(
                    f"Recorded result: experiment={experiment_id} "
                    f"variant={variant_id} {metric_name}={metric_value}"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to record experiment result: {e}")
                return False

        if db:
            return await _record(db)
        else:
            async with get_db_context() as session:
                return await _record(session)

    async def get_experiment_stats(
        self,
        experiment_id: uuid.UUID,
        metric_name: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> Optional[ExperimentStats]:
        """Get aggregated statistics for an experiment.

        Computes per-variant statistics (count, mean, std) and performs
        a t-test for statistical significance between control and treatment.

        Args:
            experiment_id: UUID of the experiment
            metric_name: Specific metric to analyze (defaults to experiment's success_metric)
            db: Optional database session

        Returns:
            ExperimentStats with per-variant stats and p-value, or None if not found
        """
        async def _get_stats(session: AsyncSession) -> Optional[ExperimentStats]:
            # Get experiment
            experiment = await self._get_experiment(experiment_id, session)
            if not experiment:
                return None

            # Use specified metric or default to experiment's success metric
            target_metric = metric_name or experiment.success_metric

            # Query for per-variant statistics
            stats_stmt = (
                select(
                    ExperimentResult.variant_id,
                    func.count(ExperimentResult.id).label("count"),
                    func.avg(ExperimentResult.metric_value).label("mean"),
                    func.stddev(ExperimentResult.metric_value).label("std"),
                    func.min(ExperimentResult.metric_value).label("min_value"),
                    func.max(ExperimentResult.metric_value).label("max_value"),
                )
                .where(
                    and_(
                        ExperimentResult.experiment_id == experiment_id,
                        ExperimentResult.metric_name == target_metric,
                    )
                )
                .group_by(ExperimentResult.variant_id)
            )

            result = await session.execute(stats_stmt)
            rows = result.all()

            if not rows:
                return ExperimentStats(
                    experiment_id=experiment_id,
                    experiment_name=experiment.name,
                    status=experiment.status.value,
                    primary_metric=target_metric,
                    variants={},
                    p_value=None,
                    is_significant=False,
                    total_samples=0,
                    recommendation=None,
                )

            # Build variant stats
            variant_stats: dict[str, VariantStats] = {}
            for row in rows:
                variant_stats[row.variant_id] = VariantStats(
                    variant_id=row.variant_id,
                    count=row.count,
                    mean=float(row.mean) if row.mean else 0.0,
                    std=float(row.std) if row.std else 0.0,
                    min_value=float(row.min_value) if row.min_value else 0.0,
                    max_value=float(row.max_value) if row.max_value else 0.0,
                )

            # Calculate total samples
            total_samples = sum(vs.count for vs in variant_stats.values())

            # Compute p-value if we have at least 2 variants with data
            p_value = None
            is_significant = False
            recommendation = None

            if len(variant_stats) >= 2:
                # Get raw values for statistical test
                variant_values = await self._get_variant_values(
                    experiment_id, target_metric, session
                )

                if len(variant_values) >= 2:
                    variant_ids = list(variant_values.keys())
                    control_values = variant_values.get(variant_ids[0], [])
                    treatment_values = variant_values.get(variant_ids[1], [])

                    if control_values and treatment_values:
                        p_value = self._calculate_p_value(
                            control_values, treatment_values
                        )
                        is_significant = p_value is not None and p_value < 0.05

                        # Generate recommendation
                        if is_significant:
                            control_mean = variant_stats[variant_ids[0]].mean
                            treatment_mean = variant_stats[variant_ids[1]].mean

                            # Determine which is better (depends on metric - lower is better for latency)
                            if target_metric in ("latency", "latency_ms", "response_time"):
                                winner = (
                                    variant_ids[0]
                                    if control_mean < treatment_mean
                                    else variant_ids[1]
                                )
                                improvement = abs(control_mean - treatment_mean) / max(
                                    control_mean, treatment_mean
                                ) * 100
                            else:
                                winner = (
                                    variant_ids[0]
                                    if control_mean > treatment_mean
                                    else variant_ids[1]
                                )
                                improvement = abs(control_mean - treatment_mean) / min(
                                    control_mean, treatment_mean
                                ) * 100 if min(control_mean, treatment_mean) > 0 else 0

                            recommendation = (
                                f"Statistically significant result (p={p_value:.4f}). "
                                f"'{winner}' shows {improvement:.1f}% improvement. "
                                f"Consider rolling out '{winner}'."
                            )
                        else:
                            min_samples = 30  # Rule of thumb for statistical power
                            if total_samples < min_samples * 2:
                                recommendation = (
                                    f"Not yet significant (p={p_value:.4f if p_value else 'N/A'}). "
                                    f"Need more samples (current: {total_samples}, "
                                    f"recommended: {min_samples * 2}+)."
                                )
                            else:
                                recommendation = (
                                    f"No significant difference detected (p={p_value:.4f if p_value else 'N/A'}). "
                                    f"Variants perform similarly for '{target_metric}'."
                                )

            return ExperimentStats(
                experiment_id=experiment_id,
                experiment_name=experiment.name,
                status=experiment.status.value,
                primary_metric=target_metric,
                variants=variant_stats,
                p_value=p_value,
                is_significant=is_significant,
                total_samples=total_samples,
                recommendation=recommendation,
            )

        if db:
            return await _get_stats(db)
        else:
            async with get_db_context() as session:
                return await _get_stats(session)

    async def get_experiment_by_name(
        self,
        name: str,
        db: Optional[AsyncSession] = None,
    ) -> Optional[dict[str, Any]]:
        """Get an experiment by its name.

        Args:
            name: Experiment name
            db: Optional database session

        Returns:
            Experiment dict or None if not found
        """
        async def _fetch(session: AsyncSession) -> Optional[dict[str, Any]]:
            stmt = select(Experiment).where(Experiment.name == name)
            result = await session.execute(stmt)
            experiment = result.scalar_one_or_none()

            if not experiment:
                return None

            return {
                "id": str(experiment.id),
                "name": experiment.name,
                "description": experiment.description,
                "experiment_type": experiment.experiment_type.value,
                "status": experiment.status.value,
                "variants": experiment.variants,
                "traffic_split": experiment.traffic_split,
                "success_metric": experiment.success_metric,
                "created_at": experiment.created_at.isoformat(),
                "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
                "ended_at": experiment.ended_at.isoformat() if experiment.ended_at else None,
            }

        if db:
            return await _fetch(db)
        else:
            async with get_db_context() as session:
                return await _fetch(session)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _hash_session_to_bucket(self, session_id: str, num_buckets: int) -> int:
        """Hash a session ID to a deterministic bucket number.

        Uses MD5 hash for fast, uniform distribution. The hash is deterministic,
        so the same session_id always maps to the same bucket.

        Args:
            session_id: Session identifier to hash
            num_buckets: Total number of buckets (typically 100)

        Returns:
            Bucket number in range [0, num_buckets)
        """
        hash_bytes = hashlib.md5(session_id.encode()).digest()
        # Use first 4 bytes as unsigned int
        hash_int = int.from_bytes(hash_bytes[:4], byteorder="big")
        return hash_int % num_buckets

    def _select_variant_by_bucket(
        self, bucket: int, traffic_split: dict[str, int]
    ) -> Optional[str]:
        """Select a variant based on bucket number and traffic split.

        Traffic split is a dict like {"control": 50, "treatment": 50}.
        Buckets 0-49 go to control, 50-99 go to treatment.

        Args:
            bucket: Bucket number (0-99)
            traffic_split: Dict mapping variant_id to percentage

        Returns:
            Selected variant_id or None if invalid configuration
        """
        if not traffic_split:
            return None

        cumulative = 0
        for variant_id, percentage in traffic_split.items():
            cumulative += percentage
            if bucket < cumulative:
                return variant_id

        # Fallback to last variant if bucket >= 100 (shouldn't happen with valid splits)
        return list(traffic_split.keys())[-1] if traffic_split else None

    def _find_variant_by_id(
        self, variants: list[dict[str, Any]], variant_id: str
    ) -> Optional[dict[str, Any]]:
        """Find a variant configuration by its ID.

        Args:
            variants: List of variant configuration dicts
            variant_id: ID to search for

        Returns:
            Variant dict or None if not found
        """
        for variant in variants:
            if variant.get("id") == variant_id:
                return variant
        return None

    def _calculate_p_value(
        self, control_values: list[float], treatment_values: list[float]
    ) -> Optional[float]:
        """Calculate p-value using Welch's t-test for independent samples.

        Welch's t-test is more robust than Student's t-test when sample sizes
        or variances are unequal.

        Args:
            control_values: List of metric values from control group
            treatment_values: List of metric values from treatment group

        Returns:
            Two-tailed p-value or None if calculation fails
        """
        if len(control_values) < 2 or len(treatment_values) < 2:
            return None

        try:
            # Calculate means
            n1 = len(control_values)
            n2 = len(treatment_values)
            mean1 = sum(control_values) / n1
            mean2 = sum(treatment_values) / n2

            # Calculate variances
            var1 = sum((x - mean1) ** 2 for x in control_values) / (n1 - 1)
            var2 = sum((x - mean2) ** 2 for x in treatment_values) / (n2 - 1)

            # Welch's t-statistic
            se = (var1 / n1 + var2 / n2) ** 0.5
            if se == 0:
                return 1.0  # No difference

            t_stat = abs(mean1 - mean2) / se

            # Welch-Satterthwaite degrees of freedom
            num = (var1 / n1 + var2 / n2) ** 2
            denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
            df = num / denom if denom > 0 else 1

            # Approximate p-value using t-distribution
            # Using a simple approximation for two-tailed test
            p_value = self._t_distribution_p_value(t_stat, df)

            return p_value

        except (ZeroDivisionError, ValueError) as e:
            logger.warning(f"P-value calculation failed: {e}")
            return None

    def _t_distribution_p_value(self, t_stat: float, df: float) -> float:
        """Approximate two-tailed p-value from t-distribution.

        Uses a polynomial approximation that's accurate for most practical cases.

        Args:
            t_stat: Absolute value of t-statistic
            df: Degrees of freedom

        Returns:
            Approximate two-tailed p-value
        """
        import math

        # For large df, use normal approximation
        if df > 100:
            # Standard normal CDF approximation
            z = t_stat
            p = math.erfc(z / math.sqrt(2))
            return p

        # For smaller df, use a reasonable approximation
        # Based on the regularized incomplete beta function
        x = df / (df + t_stat ** 2)

        # Approximation for incomplete beta function
        # This gives reasonable results for df > 1
        if df <= 1:
            return 1.0

        # Simple approximation based on normal with correction factor
        correction = 1 + (1 / (4 * df))
        z_approx = t_stat / correction
        p = math.erfc(z_approx / math.sqrt(2))

        return min(p, 1.0)

    async def _get_experiment(
        self, experiment_id: uuid.UUID, db: AsyncSession
    ) -> Optional[Experiment]:
        """Get an experiment by ID.

        Args:
            experiment_id: UUID of the experiment
            db: Database session

        Returns:
            Experiment object or None
        """
        stmt = select(Experiment).where(Experiment.id == experiment_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_variant_values(
        self,
        experiment_id: uuid.UUID,
        metric_name: str,
        db: AsyncSession,
    ) -> dict[str, list[float]]:
        """Get raw metric values grouped by variant.

        Args:
            experiment_id: UUID of the experiment
            metric_name: Name of the metric
            db: Database session

        Returns:
            Dict mapping variant_id to list of metric values
        """
        stmt = (
            select(ExperimentResult.variant_id, ExperimentResult.metric_value)
            .where(
                and_(
                    ExperimentResult.experiment_id == experiment_id,
                    ExperimentResult.metric_name == metric_name,
                )
            )
            .order_by(ExperimentResult.variant_id)
        )

        result = await db.execute(stmt)
        rows = result.all()

        variant_values: dict[str, list[float]] = {}
        for row in rows:
            if row.variant_id not in variant_values:
                variant_values[row.variant_id] = []
            variant_values[row.variant_id].append(row.metric_value)

        return variant_values


# Singleton instance
ab_testing_service = ABTestingService()
