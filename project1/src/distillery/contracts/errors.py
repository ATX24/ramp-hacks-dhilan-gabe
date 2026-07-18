"""Typed, fail-loud errors. Every error carries a machine code and retryability."""
from __future__ import annotations


class DistilleryError(Exception):
    code: str = "INTERNAL_ERROR"
    retryable: bool = False
    http_status: int = 400

    def __init__(self, message: str, *, details: dict | None = None, run_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.run_id = run_id

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
            "run_id": self.run_id,
        }


def _make(code: str, retryable: bool = False, http_status: int = 400):
    return type(code.title().replace("_", ""), (DistilleryError,), {
        "code": code, "retryable": retryable, "http_status": http_status,
    })


InvalidDataset = _make("INVALID_DATASET")
SchemaMismatch = _make("SCHEMA_MISMATCH")
DataLeakageDetected = _make("DATA_LEAKAGE_DETECTED")
UnsupportedLabelSource = _make("UNSUPPORTED_LABEL_SOURCE")
ModelRevisionUnpinned = _make("MODEL_REVISION_UNPINNED")
TokenizerMismatch = _make("TOKENIZER_MISMATCH")
ChatTemplateMismatch = _make("CHAT_TEMPLATE_MISMATCH")
LicenseGateUnresolved = _make("LICENSE_GATE_UNRESOLVED")
OutputUseNotAllowed = _make("OUTPUT_USE_NOT_ALLOWED")
RecipeNotImplemented = _make("RECIPE_NOT_IMPLEMENTED", http_status=422)
RecipeIncompatible = _make("RECIPE_INCOMPATIBLE", http_status=422)
CapabilityUnavailable = _make("CAPABILITY_UNAVAILABLE", http_status=422)
MemoryDryRunFailed = _make("MEMORY_DRY_RUN_FAILED")
EstimatedBudgetExceeded = _make("ESTIMATED_BUDGET_EXCEEDED")
TeacherUnavailable = _make("TEACHER_UNAVAILABLE", retryable=True, http_status=503)
SubmissionFailed = _make("SUBMISSION_FAILED", retryable=True, http_status=502)
RunTimeout = _make("RUN_TIMEOUT")
Cancelled = _make("CANCELLED")
ArtifactIntegrityFailed = _make("ARTIFACT_INTEGRITY_FAILED")
EvaluationIncomplete = _make("EVALUATION_INCOMPLETE")
InsufficientEvidence = _make("INSUFFICIENT_EVIDENCE")
NotFound = _make("NOT_FOUND", http_status=404)
