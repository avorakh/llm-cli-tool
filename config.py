MODELS = {
    "gpt-4o-mini": {
        "provider":       "openai",
        "context_window": 128_000,
        "cost": {"input": 0.00000015, "output": 0.00000060},
    },
    "gpt-4o": {
        "provider":       "openai",
        "context_window": 128_000,
        "cost": {"input": 0.0000025, "output": 0.00001},
    },
    "claude-3-5-haiku": {
        "provider":       "anthropic",
        "context_window": 200_000,
        "cost": {"input": 0.0000008, "output": 0.0000024},
    },
    "claude-3-5-sonnet": {
        "provider":       "anthropic",
        "context_window": 200_000,
        "cost": {"input": 0.000003, "output": 0.000015},
    },
}

DEFAULT_MODEL = "gpt-4o-mini"
