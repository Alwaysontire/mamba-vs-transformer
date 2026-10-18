"""HTML-отчёты по запускам: контракт входа + сборка"""

from src.reporting.builder import build_report
from src.reporting.inputs import EvaluationMetrics, RunResult, TrainManifest

__all__ = ["build_report", "EvaluationMetrics", "RunResult", "TrainManifest"]
