"""Structured subtask termination and execution-quality accounting."""

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    EXHAUSTED = "exhausted"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class SubtaskExecution:
    """Keep the first terminal outcome while accumulating non-terminal errors."""

    termination_reason: str = None
    execution_status: ExecutionStatus = None
    termination_detail: dict = field(default_factory=dict)
    error_counts: Counter = field(default_factory=Counter)

    def record_error(self, category, count=1):
        count = int(count)
        if count > 0:
            self.error_counts[str(category)] += count

    def terminate(self, status, reason, detail=None, error_category=None):
        status = ExecutionStatus(status)
        if error_category is not None:
            self.record_error(error_category)
        if self.termination_reason is not None:
            return False
        self.execution_status = status
        self.termination_reason = str(reason)
        self.termination_detail = dict(detail or {})
        return True

    def finalize(self, task_success):
        if self.termination_reason is None:
            if task_success:
                self.terminate(ExecutionStatus.COMPLETED, "legacy_success_stop")
            else:
                self.terminate(ExecutionStatus.EXHAUSTED, "step_budget_exhausted")
        return {
            "execution_status": self.execution_status.value,
            "termination_reason": self.termination_reason,
            "termination_detail": dict(self.termination_detail),
            "error_counts": dict(sorted(self.error_counts.items())),
        }


def summarize_execution_quality(records):
    """Summarize new records while explicitly identifying legacy omissions."""

    statuses = Counter()
    reasons = Counter()
    errors = Counter()
    missing = []
    for subtask_id, record in records.items():
        status = record.get("execution_status")
        reason = record.get("termination_reason")
        if not status or not reason:
            status = ExecutionStatus.UNKNOWN.value
            reason = "unknown"
            missing.append(str(subtask_id))
        statuses[str(status)] += 1
        reasons[str(reason)] += 1
        for category, count in (record.get("error_counts") or {}).items():
            errors[str(category)] += int(count)
    return {
        "total": len(records),
        "statuses": dict(sorted(statuses.items())),
        "termination_reasons": dict(sorted(reasons.items())),
        "error_counts": dict(sorted(errors.items())),
        "missing_outcome_count": len(missing),
        "missing_outcome_ids": sorted(missing),
        "acceptance_ready": (bool(records) and not missing
                             and not statuses[ExecutionStatus.ERROR.value]
                             and not errors),
    }
