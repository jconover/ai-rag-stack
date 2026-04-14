"""RAG scoring helpers using RAGAS.

Exposes :func:`run_ragas_scoring` which runs a small subset of RAGAS metrics
over a single ``(question, answer, contexts[, ground_truth])`` sample and
returns a plain dict of metric name -> float.

We intentionally keep the LLM wiring pragmatic: RAGAS historically expects
an OpenAI-compatible LLM, and wiring arbitrary providers into it is not
stable across versions. For the local learning setup we therefore attempt
to use RAGAS with whatever ``LLM_PROVIDER`` is currently configured via a
LangChain wrapper when possible, and fall back to a lightweight
heuristic-based scoring path if ragas / datasets are unavailable or fail
to initialize. The heuristic keeps the endpoint usable for experimentation
without cloud credentials.

Results are also persisted to a Qdrant collection ``eval_results`` via
``vector_store.store_eval_result`` so you can inspect the history of
scoring runs over time.
"""

from __future__ import annotations

import importlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _heuristic_scores(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str],
) -> Dict[str, float]:
    """Fallback scoring when RAGAS cannot be initialized.

    Uses simple token-overlap heuristics so the endpoint still returns
    useful numbers for local learning / smoke testing.
    """

    def toks(s: str) -> set:
        return {t.lower() for t in (s or "").split() if len(t) > 2}

    q_tokens = toks(question)
    a_tokens = toks(answer)
    ctx_tokens = set().union(*(toks(c) for c in contexts)) if contexts else set()

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    faithfulness = jaccard(a_tokens, ctx_tokens)
    answer_relevancy = jaccard(a_tokens, q_tokens)
    context_precision = jaccard(ctx_tokens, q_tokens)
    scores: Dict[str, Any] = {
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(answer_relevancy, 4),
        "context_precision": round(context_precision, 4),
    }
    if ground_truth:
        gt_tokens = toks(ground_truth)
        scores["context_recall"] = round(jaccard(gt_tokens, ctx_tokens), 4)
    scores["_engine"] = "heuristic"
    return scores


def run_ragas_scoring(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Dict[str, Any]:
    """Run RAGAS metrics on a single sample and return a dict of scores.

    Metrics:
      - faithfulness
      - answer_relevancy
      - context_precision
      - context_recall (only if ``ground_truth`` is provided)
    """
    try:
        datasets_mod = importlib.import_module("datasets")
        ragas_mod = importlib.import_module("ragas")
        metrics_mod = importlib.import_module("ragas.metrics")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ragas/datasets unavailable, using heuristic scores (%s)", exc)
        return _heuristic_scores(question, answer, contexts, ground_truth)

    Dataset = getattr(datasets_mod, "Dataset")
    runner = getattr(ragas_mod, "evaluate")
    faithfulness_m = getattr(metrics_mod, "faithfulness")
    answer_relevancy_m = getattr(metrics_mod, "answer_relevancy")
    context_precision_m = getattr(metrics_mod, "context_precision")
    context_recall_m = getattr(metrics_mod, "context_recall")

    sample = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts or [""]],
    }
    metrics = [faithfulness_m, answer_relevancy_m, context_precision_m]
    if ground_truth:
        sample["ground_truth"] = [ground_truth]
        metrics.append(context_recall_m)

    try:
        ds = Dataset.from_dict(sample)
        result = runner(ds, metrics=metrics)
        try:
            scores = {
                k: float(v)
                for k, v in dict(result).items()
                if isinstance(v, (int, float))
            }
        except Exception:
            scores = {
                m.name: float(result[m.name]) for m in metrics if m.name in result
            }
        scores["_engine"] = "ragas"
        return scores
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ragas runner failed (%s); falling back to heuristic scoring. "
            "This usually means ragas needs an OpenAI-compatible LLM configured.",
            exc,
        )
        return _heuristic_scores(question, answer, contexts, ground_truth)


def build_eval_record(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the payload stored in the ``eval_results`` Qdrant collection."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
        "metrics": metrics,
    }
