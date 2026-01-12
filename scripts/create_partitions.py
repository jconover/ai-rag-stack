#!/usr/bin/env python3
"""
Partition maintenance script for query_logs table.

This script manages monthly partitions for the query_logs table:
- Creates future partitions (default: 3 months ahead)
- Optionally drops old partitions beyond retention period
- Lists current partition status
- Validates partition health

Usage:
    # Create partitions for next 3 months
    python scripts/create_partitions.py

    # Create partitions for next 6 months
    python scripts/create_partitions.py --months-ahead 6

    # Drop partitions older than 12 months
    python scripts/create_partitions.py --drop-old --retention-months 12

    # List current partitions
    python scripts/create_partitions.py --list

    # Dry run (show what would be done)
    python scripts/create_partitions.py --dry-run

Environment variables:
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB

Cron example (run monthly on the 1st at 2 AM):
    0 2 1 * * cd /app && python scripts/create_partitions.py --months-ahead 3 >> /var/log/partition_maintenance.log 2>&1
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    import psycopg2
    from psycopg2 import sql
except ImportError:
    print("Error: psycopg2 is required. Install with: pip install psycopg2-binary")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class PartitionInfo:
    """Information about a single partition."""

    name: str
    start_date: datetime
    end_date: datetime
    row_count: int
    size_bytes: int


def get_connection_params() -> dict:
    """Get database connection parameters from environment variables."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
        "database": os.getenv("POSTGRES_DB", "devops_assistant"),
    }


def get_connection():
    """Create a database connection."""
    params = get_connection_params()
    logger.debug(f"Connecting to PostgreSQL at {params['host']}:{params['port']}/{params['database']}")
    return psycopg2.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        dbname=params["database"],
    )


def is_table_partitioned(conn) -> bool:
    """Check if query_logs table exists and is partitioned."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'query_logs'
                AND c.relkind = 'p'  -- 'p' = partitioned table
                AND n.nspname = 'public'
            )
        """)
        return cur.fetchone()[0]


def get_partition_name(year: int, month: int) -> str:
    """Generate partition name for a given year and month."""
    return f"query_logs_y{year:04d}m{month:02d}"


def parse_partition_date(partition_name: str) -> Optional[datetime]:
    """Extract the start date from a partition name."""
    import re

    match = re.match(r"query_logs_y(\d{4})m(\d{2})", partition_name)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        return datetime(year, month, 1)
    return None


def partition_exists(conn, partition_name: str) -> bool:
    """Check if a specific partition exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = %s
                AND n.nspname = 'public'
            )
            """,
            (partition_name,),
        )
        return cur.fetchone()[0]


def list_partitions(conn) -> List[PartitionInfo]:
    """List all partitions of the query_logs table with their details."""
    partitions = []

    with conn.cursor() as cur:
        # Get all child tables of query_logs
        cur.execute("""
            SELECT
                c.relname AS partition_name,
                pg_get_expr(c.relpartbound, c.oid) AS partition_range,
                pg_relation_size(c.oid) AS size_bytes
            FROM pg_class c
            JOIN pg_inherits i ON c.oid = i.inhrelid
            JOIN pg_class parent ON i.inhparent = parent.oid
            WHERE parent.relname = 'query_logs'
            ORDER BY c.relname
        """)

        for row in cur.fetchall():
            partition_name, partition_range, size_bytes = row

            # Get row count (approximate from pg_class for performance)
            cur.execute(
                "SELECT reltuples::BIGINT FROM pg_class WHERE relname = %s",
                (partition_name,),
            )
            row_count_result = cur.fetchone()
            row_count = row_count_result[0] if row_count_result else 0

            # Parse date range from partition_range
            start_date = parse_partition_date(partition_name)
            end_date = None
            if start_date:
                if start_date.month == 12:
                    end_date = datetime(start_date.year + 1, 1, 1)
                else:
                    end_date = datetime(start_date.year, start_date.month + 1, 1)

            partitions.append(
                PartitionInfo(
                    name=partition_name,
                    start_date=start_date,
                    end_date=end_date,
                    row_count=int(row_count) if row_count else 0,
                    size_bytes=size_bytes or 0,
                )
            )

    return partitions


def create_partition(conn, year: int, month: int, dry_run: bool = False) -> bool:
    """Create a monthly partition for the specified year and month."""
    partition_name = get_partition_name(year, month)

    if partition_exists(conn, partition_name):
        logger.info(f"Partition {partition_name} already exists, skipping")
        return False

    # Calculate date range
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)

    if dry_run:
        logger.info(
            f"[DRY RUN] Would create partition {partition_name} "
            f"({start_date.date()} to {end_date.date()})"
        )
        return True

    # Create the partition
    create_sql = sql.SQL("""
        CREATE TABLE {partition} PARTITION OF query_logs
        FOR VALUES FROM (%s) TO (%s)
    """).format(partition=sql.Identifier(partition_name))

    with conn.cursor() as cur:
        cur.execute(create_sql, (start_date, end_date))

    conn.commit()
    logger.info(
        f"Created partition {partition_name} ({start_date.date()} to {end_date.date()})"
    )
    return True


def drop_partition(conn, partition_name: str, dry_run: bool = False) -> bool:
    """Drop a partition."""
    if not partition_exists(conn, partition_name):
        logger.warning(f"Partition {partition_name} does not exist")
        return False

    if dry_run:
        logger.info(f"[DRY RUN] Would drop partition {partition_name}")
        return True

    drop_sql = sql.SQL("DROP TABLE {partition}").format(
        partition=sql.Identifier(partition_name)
    )

    with conn.cursor() as cur:
        cur.execute(drop_sql)

    conn.commit()
    logger.info(f"Dropped partition {partition_name}")
    return True


def create_future_partitions(conn, months_ahead: int = 3, dry_run: bool = False) -> int:
    """Create partitions for the next N months."""
    now = datetime.now()
    partitions_created = 0

    for i in range(months_ahead + 1):  # +1 to include current month
        target_date = now + timedelta(days=30 * i)
        year, month = target_date.year, target_date.month

        if create_partition(conn, year, month, dry_run):
            partitions_created += 1

    return partitions_created


def drop_old_partitions(
    conn, retention_months: int = 12, dry_run: bool = False
) -> int:
    """Drop partitions older than the retention period."""
    cutoff_date = datetime.now() - timedelta(days=30 * retention_months)
    partitions_dropped = 0

    partitions = list_partitions(conn)
    for partition in partitions:
        # Skip default partition and partitions without parseable dates
        if partition.start_date is None:
            continue

        if partition.name == "query_logs_default":
            continue

        if partition.start_date < cutoff_date:
            if drop_partition(conn, partition.name, dry_run):
                partitions_dropped += 1

    return partitions_dropped


def print_partition_status(conn):
    """Print a formatted table of partition status."""
    partitions = list_partitions(conn)

    if not partitions:
        print("No partitions found. Is query_logs partitioned?")
        return

    print("\nquery_logs Partition Status")
    print("=" * 80)
    print(
        f"{'Partition Name':<30} {'Date Range':<25} {'Rows':>12} {'Size':>10}"
    )
    print("-" * 80)

    total_rows = 0
    total_size = 0

    for p in partitions:
        if p.start_date and p.end_date:
            date_range = f"{p.start_date.date()} to {p.end_date.date()}"
        elif p.name == "query_logs_default":
            date_range = "DEFAULT (catch-all)"
        else:
            date_range = "Unknown"

        size_str = format_bytes(p.size_bytes)
        print(f"{p.name:<30} {date_range:<25} {p.row_count:>12,} {size_str:>10}")

        total_rows += p.row_count
        total_size += p.size_bytes

    print("-" * 80)
    print(
        f"{'TOTAL':<30} {'':<25} {total_rows:>12,} {format_bytes(total_size):>10}"
    )
    print(f"\nTotal partitions: {len(partitions)}")


def format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def validate_partitions(conn) -> Tuple[bool, List[str]]:
    """Validate partition health and identify issues."""
    issues = []
    now = datetime.now()
    current_month = datetime(now.year, now.month, 1)

    # Check if table is partitioned
    if not is_table_partitioned(conn):
        issues.append("query_logs table is not partitioned")
        return False, issues

    partitions = list_partitions(conn)
    partition_dates = {
        p.start_date for p in partitions if p.start_date and p.name != "query_logs_default"
    }

    # Check for current month partition
    if current_month not in partition_dates:
        issues.append(f"Missing partition for current month: {current_month.date()}")

    # Check for next month partition
    if current_month.month == 12:
        next_month = datetime(current_month.year + 1, 1, 1)
    else:
        next_month = datetime(current_month.year, current_month.month + 1, 1)

    if next_month not in partition_dates:
        issues.append(f"Missing partition for next month: {next_month.date()}")

    # Check for default partition receiving data (indicates missing partitions)
    for p in partitions:
        if p.name == "query_logs_default" and p.row_count > 0:
            issues.append(
                f"Default partition contains {p.row_count} rows - "
                "missing partitions for some date ranges"
            )

    is_healthy = len(issues) == 0
    return is_healthy, issues


def main():
    parser = argparse.ArgumentParser(
        description="Manage query_logs table partitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create partitions for next 3 months
    python scripts/create_partitions.py --months-ahead 3

    # List current partitions
    python scripts/create_partitions.py --list

    # Drop partitions older than 12 months (dry run)
    python scripts/create_partitions.py --drop-old --retention-months 12 --dry-run

    # Validate partition health
    python scripts/create_partitions.py --validate
        """,
    )

    parser.add_argument(
        "--months-ahead",
        type=int,
        default=3,
        help="Number of months ahead to create partitions (default: 3)",
    )
    parser.add_argument(
        "--drop-old",
        action="store_true",
        help="Drop partitions older than retention period",
    )
    parser.add_argument(
        "--retention-months",
        type=int,
        default=12,
        help="Retention period in months for --drop-old (default: 12)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List current partitions and exit",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate partition health and report issues",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        conn = get_connection()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    try:
        # Check if table is partitioned
        if not is_table_partitioned(conn):
            logger.error(
                "query_logs table is not partitioned. "
                "Run scripts/migrations/partition_query_logs.sql first."
            )
            sys.exit(1)

        # List mode
        if args.list:
            print_partition_status(conn)
            sys.exit(0)

        # Validate mode
        if args.validate:
            is_healthy, issues = validate_partitions(conn)
            if is_healthy:
                print("Partition health check: PASSED")
                print("All partitions are properly configured.")
            else:
                print("Partition health check: FAILED")
                print("\nIssues found:")
                for issue in issues:
                    print(f"  - {issue}")
            sys.exit(0 if is_healthy else 1)

        # Create future partitions
        logger.info(f"Creating partitions for next {args.months_ahead} months...")
        created = create_future_partitions(conn, args.months_ahead, args.dry_run)
        logger.info(f"Created {created} new partition(s)")

        # Drop old partitions if requested
        if args.drop_old:
            logger.info(
                f"Dropping partitions older than {args.retention_months} months..."
            )
            dropped = drop_old_partitions(conn, args.retention_months, args.dry_run)
            logger.info(f"Dropped {dropped} old partition(s)")

        # Show final status
        if not args.dry_run:
            print_partition_status(conn)

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
