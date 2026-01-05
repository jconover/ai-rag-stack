"""User feedback logging for RAG response quality tracking.

This module provides structured logging for user feedback (thumbs up/down)
to track response quality and enable analytics.

Log format: JSON lines for easy parsing and aggregation.
"""

import json
import uuid
import statistics
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import os


# Configure logger for feedback
FEEDBACK_LOG_PATH = os.getenv("FEEDBACK_LOG_PATH", "/data/logs/user_feedback.jsonl")

feedback_logger = logging.getLogger("user_feedback")
feedback_logger.setLevel(logging.INFO)
feedback_logger.propagate = False

# Try to set up file handler, fall back to /tmp if permissions fail
try:
    Path(FEEDBACK_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(FEEDBACK_LOG_PATH, mode='a')
except (PermissionError, OSError):
    FEEDBACK_LOG_PATH = "/tmp/user_feedback.jsonl"
    file_handler = logging.FileHandler(FEEDBACK_LOG_PATH, mode='a')
    logging.warning(f"Could not write to original path, using fallback: {FEEDBACK_LOG_PATH}")

file_handler.setFormatter(logging.Formatter('%(message)s'))
feedback_logger.addHandler(file_handler)


@dataclass
class FeedbackEntry:
    """Structured feedback data."""
    timestamp: str
    feedback_id: str
    session_id: str
    message_index: Optional[int]
    helpful: bool
    query_hash: Optional[str]

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), default=str)

    @classmethod
    def create(
        cls,
        session_id: str,
        helpful: bool,
        message_index: Optional[int] = None,
        query_hash: Optional[str] = None
    ) -> "FeedbackEntry":
        """Create a new feedback entry with generated ID and timestamp."""
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            feedback_id=str(uuid.uuid4()),
            session_id=session_id,
            message_index=message_index,
            helpful=helpful,
            query_hash=query_hash
        )


class FeedbackLogger:
    """Logger for user feedback on RAG responses."""

    def log_feedback(
        self,
        session_id: str,
        helpful: bool,
        message_index: Optional[int] = None,
        query_hash: Optional[str] = None
    ) -> FeedbackEntry:
        """Log user feedback to JSON lines file.

        Args:
            session_id: Session ID for the conversation
            helpful: True for thumbs up, False for thumbs down
            message_index: Optional index of the response in session
            query_hash: Optional hash for correlation with retrieval metrics

        Returns:
            FeedbackEntry with generated ID and timestamp
        """
        entry = FeedbackEntry.create(
            session_id=session_id,
            helpful=helpful,
            message_index=message_index,
            query_hash=query_hash
        )

        feedback_logger.info(entry.to_json())
        return entry


# Singleton instance
feedback_log = FeedbackLogger()


def get_feedback_summary(log_path: str = FEEDBACK_LOG_PATH, last_n: int = 100) -> Dict[str, Any]:
    """Read recent feedback and compute summary statistics.

    Args:
        log_path: Path to feedback log file
        last_n: Number of recent entries to analyze

    Returns:
        Summary statistics dictionary
    """
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        recent_lines = lines[-last_n:] if len(lines) > last_n else lines

        total = 0
        helpful_count = 0
        not_helpful_count = 0
        sessions = set()

        for line in recent_lines:
            try:
                entry = json.loads(line.strip())
                total += 1
                sessions.add(entry.get('session_id'))

                if entry.get('helpful'):
                    helpful_count += 1
                else:
                    not_helpful_count += 1

            except json.JSONDecodeError:
                continue

        summary = {
            "total_feedback": total,
            "helpful_count": helpful_count,
            "not_helpful_count": not_helpful_count,
            "unique_sessions": len(sessions),
            "log_path": log_path
        }

        if total > 0:
            summary["helpful_rate"] = round(helpful_count / total, 4)

        return summary

    except FileNotFoundError:
        return {
            "total_feedback": 0,
            "helpful_count": 0,
            "not_helpful_count": 0,
            "unique_sessions": 0,
            "log_path": log_path,
            "note": "No feedback logged yet"
        }
    except Exception as e:
        return {"error": str(e)}
