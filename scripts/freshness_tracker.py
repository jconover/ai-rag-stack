"""Document freshness tracking and staleness detection.

Tracks when documentation was last updated and detects stale content
that may contain outdated information.
"""

import os
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Freshness thresholds in days (low/medium/high risk)
FRESHNESS_THRESHOLDS = {
    'kubernetes': {'low': 7, 'medium': 30, 'high': 90},
    'terraform': {'low': 14, 'medium': 60, 'high': 180},
    'docker': {'low': 14, 'medium': 60, 'high': 180},
    'ansible': {'low': 30, 'medium': 90, 'high': 365},
    'prometheus': {'low': 30, 'medium': 90, 'high': 180},
    'github-actions': {'low': 7, 'medium': 21, 'high': 60},
    'helm': {'low': 14, 'medium': 60, 'high': 180},
    'default': {'low': 30, 'medium': 90, 'high': 365},
}

FRESHNESS_DB_PATH = Path("data/freshness_registry.json")


@dataclass
class SourceFreshness:
    """Freshness status for a documentation source."""
    source_type: str
    last_download: Optional[str]  # ISO format datetime
    last_commit_date: Optional[str]  # From git log
    days_since_update: int
    staleness_risk: str  # "fresh", "low", "medium", "high", "critical"
    recommended_action: str

    def to_dict(self) -> dict:
        return asdict(self)


class FreshnessTracker:
    """Track documentation freshness and detect staleness."""

    def __init__(self, db_path: Path = FRESHNESS_DB_PATH):
        self.db_path = db_path
        self._registry: Dict[str, dict] = {}
        self._load_registry()

    def _load_registry(self):
        """Load freshness registry from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    self._registry = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load freshness registry: {e}")
                self._registry = {}

    def _save_registry(self):
        """Save freshness registry to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, 'w') as f:
            json.dump(self._registry, f, indent=2)

    def record_download(self, source_type: str, docs_path: str):
        """Record a documentation download with commit info."""
        commit_date = self._get_git_commit_date(docs_path)

        self._registry[source_type] = {
            'last_download': datetime.utcnow().isoformat(),
            'last_commit_date': commit_date,
            'docs_path': docs_path,
        }
        self._save_registry()
        logger.info(f"Recorded download for {source_type}, commit date: {commit_date}")

    def _get_git_commit_date(self, path: str) -> Optional[str]:
        """Get the last commit date from a git repository."""
        try:
            result = subprocess.run(
                ['git', '-C', path, 'log', '-1', '--format=%aI'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Failed to get git commit date for {path}: {e}")
        return None

    def get_freshness(self, source_type: str) -> SourceFreshness:
        """Get freshness status for a source."""
        record = self._registry.get(source_type, {})
        thresholds = FRESHNESS_THRESHOLDS.get(source_type, FRESHNESS_THRESHOLDS['default'])

        if not record.get('last_commit_date'):
            return SourceFreshness(
                source_type=source_type,
                last_download=record.get('last_download'),
                last_commit_date=None,
                days_since_update=-1,
                staleness_risk="unknown",
                recommended_action="Run ingestion to track freshness"
            )

        commit_date = datetime.fromisoformat(record['last_commit_date'].replace('Z', '+00:00'))
        days_old = (datetime.now(commit_date.tzinfo) - commit_date).days

        if days_old <= thresholds['low']:
            risk = "fresh"
            action = "No action needed"
        elif days_old <= thresholds['medium']:
            risk = "low"
            action = "Consider updating soon"
        elif days_old <= thresholds['high']:
            risk = "medium"
            action = "Update recommended"
        else:
            risk = "high"
            action = "Update required - documentation may be outdated"

        return SourceFreshness(
            source_type=source_type,
            last_download=record.get('last_download'),
            last_commit_date=record.get('last_commit_date'),
            days_since_update=days_old,
            staleness_risk=risk,
            recommended_action=action
        )

    def get_all_freshness(self) -> List[SourceFreshness]:
        """Get freshness status for all tracked sources."""
        return [self.get_freshness(src) for src in self._registry.keys()]

    def get_stale_sources(self, max_risk: str = "medium") -> List[SourceFreshness]:
        """Get sources that need updating."""
        risk_levels = ["fresh", "low", "medium", "high", "critical"]
        max_idx = risk_levels.index(max_risk)

        return [
            f for f in self.get_all_freshness()
            if f.staleness_risk in risk_levels[max_idx:]
        ]

    def print_report(self):
        """Print a freshness report to stdout."""
        print("\n" + "="*60)
        print("DOCUMENTATION FRESHNESS REPORT")
        print("="*60 + "\n")

        for freshness in sorted(self.get_all_freshness(), key=lambda x: x.days_since_update, reverse=True):
            status_icon = {"fresh": "[OK]", "low": "[LOW]", "medium": "[MED]", "high": "[HIGH]", "unknown": "[?]"}.get(freshness.staleness_risk, "[?]")
            print(f"{status_icon} {freshness.source_type}")
            print(f"   Days old: {freshness.days_since_update}")
            print(f"   Risk: {freshness.staleness_risk}")
            print(f"   Action: {freshness.recommended_action}")
            print()


# Singleton
freshness_tracker = FreshnessTracker()


if __name__ == "__main__":
    freshness_tracker.print_report()
