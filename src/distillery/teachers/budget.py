"""Hard request, rate, retry, and cost ceilings."""

from __future__ import annotations

import time
from collections.abc import Callable

from distillery.teachers.errors import TeacherErrorCode, raise_teacher_error
from distillery.teachers.types import CostRecord, TeacherBudget, TokenUsage


class RequestGovernor:
    """Session-scoped hard limits shared by retries and priority candidates."""

    def __init__(
        self,
        budget: TeacherBudget,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.budget = budget
        self._clock = clock
        self._sleep = sleep
        self.requests_started = 0
        self.spent_usd = 0.0
        self._last_started_at: float | None = None

    def before_network_call(
        self,
        *,
        estimated_input_tokens: int,
        max_output_tokens: int,
    ) -> None:
        """Start only if this attempt's worst case fits every hard ceiling."""
        if self.requests_started >= self.budget.max_requests:
            raise_teacher_error(
                TeacherErrorCode.REQUEST_CAP_EXCEEDED,
                "teacher request cap exhausted",
                details={
                    "max_requests": self.budget.max_requests,
                    "requests_started": self.requests_started,
                },
            )
        maximum = self.cost_for(
            TokenUsage(
                input_tokens=estimated_input_tokens,
                output_tokens=max_output_tokens,
                total_tokens=estimated_input_tokens + max_output_tokens,
            )
        )
        projected = round(self.spent_usd + maximum.total_usd, 10)
        if projected > self.budget.cost_ceiling_usd + 1e-12:
            raise_teacher_error(
                TeacherErrorCode.COST_EXHAUSTED,
                "teacher request worst-case cost exceeds the remaining ceiling",
                details={
                    "cost_ceiling_usd": self.budget.cost_ceiling_usd,
                    "spent_usd": self.spent_usd,
                    "maximum_request_usd": maximum.total_usd,
                    "projected_usd": projected,
                },
            )

        now = self._clock()
        if self._last_started_at is not None:
            elapsed = now - self._last_started_at
            delay = self.budget.min_request_interval_seconds - elapsed
            if delay > 0:
                self._sleep(delay)
                now = self._clock()
        self.requests_started += 1
        self._last_started_at = now

    def cost_for(self, usage: TokenUsage) -> CostRecord:
        input_usd = (usage.input_tokens / 1000.0) * self.budget.input_usd_per_1k_tokens
        output_usd = (usage.output_tokens / 1000.0) * self.budget.output_usd_per_1k_tokens
        return CostRecord(
            input_usd=round(input_usd, 10),
            output_usd=round(output_usd, 10),
            total_usd=round(input_usd + output_usd, 10),
            pricing_version=self.budget.pricing_version,
        )

    def record_usage(self, usage: TokenUsage) -> CostRecord:
        cost = self.cost_for(usage)
        projected = round(self.spent_usd + cost.total_usd, 10)
        if projected > self.budget.cost_ceiling_usd + 1e-12:
            raise_teacher_error(
                TeacherErrorCode.COST_EXHAUSTED,
                "reported teacher usage exceeds the cost ceiling",
                details={
                    "cost_ceiling_usd": self.budget.cost_ceiling_usd,
                    "spent_usd": self.spent_usd,
                    "request_usd": cost.total_usd,
                    "projected_usd": projected,
                },
            )
        self.spent_usd = projected
        return cost


def estimate_token_count(text: str) -> int:
    """Deterministic conservative byte proxy when no tokenizer is available."""
    return max(1, (len(text.encode("utf-8")) + 2) // 3)


__all__ = ["RequestGovernor", "estimate_token_count"]
