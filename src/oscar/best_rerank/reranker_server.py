"""
Celery worker that serves the Reranker as a background service.

Configure via environment variables:
  RERANKER_METHOD        - "method1" or "method2" (default: "method2")
  RERANKER_DEVICE        - torch device (default: "cuda")
  RERANKER_MAX_LENGTH    - max token length (default: 512)
  RERANKER_BATCH_SIZE    - inference batch size (default: model-specific)
  RERANKER_MODEL_PATH    - path to method2 LoRA weights (default: method2_mse beside this file)
  RERANKER_MODEL_NAME    - HF model name for method1 (default: Qwen/Qwen3-Reranker-8B)
  CELERY_BROKER_URL      - Redis/RabbitMQ broker (default: redis://localhost:6379/0)
  CELERY_RESULT_BACKEND  - result backend (default: redis://localhost:6379/0)

Start the worker:
  celery -A oscar.best_rerank.reranker_server worker --pool=solo --loglevel=info

Send a task:
  from oscar.best_rerank.reranker_server import rerank
  result = rerank.delay(goal="...", insights=["...", "..."])
  print(result.get(timeout=120))
"""

import os
from pathlib import Path
from typing import List

from celery import Celery
from celery.signals import worker_process_init

from oscar.best_rerank.reranker import Reranker

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------
broker = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

app = Celery("reranker", broker=broker, backend=backend)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# ---------------------------------------------------------------------------
# Singleton model — loaded once per worker process
# ---------------------------------------------------------------------------
_reranker: Reranker | None = None


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        method = os.environ.get("RERANKER_METHOD", "method2")
        device = os.environ.get("RERANKER_DEVICE", "cuda")
        max_length = int(os.environ.get("RERANKER_MAX_LENGTH", "512"))
        batch_size_raw = os.environ.get("RERANKER_BATCH_SIZE")
        batch_size = int(batch_size_raw) if batch_size_raw else None
        model_path = os.environ.get(
            "RERANKER_MODEL_PATH",
            str(Path(__file__).resolve().parent / "method2_no_lora"),
        )
        model_name = os.environ.get(
            "RERANKER_MODEL_NAME", "Qwen/Qwen3-Reranker-8B"
        )

        _reranker = Reranker(
            method=method,
            method2_model_path=model_path,
            method1_model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
    return _reranker


@worker_process_init.connect
def _preload_model(**_kwargs):
    """Eagerly load the model when a worker process starts."""
    _get_reranker()


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
@app.task(name="reranker.rerank", bind=True, max_retries=1)
def rerank(self, goal: str, insights: List[str]) -> list:
    """Rerank *insights* against *goal* and return the ranked list of dicts.

    Each dict contains: rank, index, insight, predicted_score.
    """
    try:
        rr = _get_reranker()
        return rr.rerank(goal, insights)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)
