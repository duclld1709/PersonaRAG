"""OpenAI pricing table for local cost estimation.

Prices are USD per 1M tokens. Keep the table small and well-documented;
it is the single source of truth for usage tracking in this app.
Update when OpenAI publishes new pricing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token pricing for a single OpenAI model."""

    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


# Source: https://openai.com/api/pricing/ (chat + embeddings tiers)
_CHAT_PRICING: dict[str, ModelPricing] = {
    "gpt-4.1": ModelPricing(2.00, 0.50, 8.00),
    "gpt-4.1-mini": ModelPricing(0.40, 0.10, 1.60),
    "gpt-4.1-nano": ModelPricing(0.10, 0.025, 0.40),
    "gpt-4o": ModelPricing(2.50, 1.25, 10.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.075, 0.60),
}

# Embedding models price only the input side.
_EMBEDDING_PRICING: dict[str, float] = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
}

# Moderation is free at the time of writing.
_MODERATION_PRICING: dict[str, float] = {}


def estimate_cost(
    model: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
) -> float:
    """Return the estimated USD cost for a chat completion call."""

    pricing = _CHAT_PRICING.get(model)
    if pricing is None:
        return 0.0
    billable_input = max(prompt_tokens - cached_tokens, 0)
    cost = (
        billable_input * pricing.input_per_million
        + cached_tokens * pricing.cached_input_per_million
        + completion_tokens * pricing.output_per_million
    ) / 1_000_000
    return round(cost, 6)


def estimate_embedding_cost(model: str, *, tokens: int) -> float:
    """Return the estimated USD cost for an embedding call."""

    rate = _EMBEDDING_PRICING.get(model)
    if rate is None:
        return 0.0
    return round(tokens * rate / 1_000_000, 6)
