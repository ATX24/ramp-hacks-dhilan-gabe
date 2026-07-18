"""Time-boxed emergency AWS smoke-training path.

Isolated from the main Distillery trainer/container workstreams. Imports
stable Distillery contracts, finance data, artifacts, and loss helpers, but
does not mutate those modules.
"""

from __future__ import annotations

__all__ = ["EMERGENCY_PROFILE_NAME"]

EMERGENCY_PROFILE_NAME = "aws_smoke_emergency_v1"
