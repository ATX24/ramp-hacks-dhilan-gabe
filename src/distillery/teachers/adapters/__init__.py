"""Teacher generator adapters."""

from distillery.teachers.adapters.anthropic import (
    AnthropicAPIError,
    AnthropicClientFactory,
    AnthropicMessagesClient,
    AnthropicModelDiscovery,
    DirectAnthropicTeacher,
    OfficialAnthropicClientFactory,
    build_anthropic_request,
)
from distillery.teachers.adapters.bedrock import (
    BedrockConverseClient,
    BedrockConverseTeacher,
    BedrockError,
    build_converse_request,
)
from distillery.teachers.adapters.local import (
    LocalOpenWeightClient,
    LocalOpenWeightTeacher,
)

__all__ = [
    "AnthropicAPIError",
    "AnthropicClientFactory",
    "AnthropicMessagesClient",
    "AnthropicModelDiscovery",
    "BedrockConverseClient",
    "BedrockConverseTeacher",
    "BedrockError",
    "DirectAnthropicTeacher",
    "LocalOpenWeightClient",
    "LocalOpenWeightTeacher",
    "OfficialAnthropicClientFactory",
    "build_anthropic_request",
    "build_converse_request",
]
