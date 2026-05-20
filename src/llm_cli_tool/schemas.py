import json


FEATURE_EXTRACTION_SCHEMA = "feature-extraction"
FEATURE_IMPORTANCE_LEVELS = {"low", "medium", "high"}
JSON_SCHEMAS = (FEATURE_EXTRACTION_SCHEMA,)


class JsonResponseValidationError(ValueError):
    """Raised when a model response is not valid JSON for the requested schema."""


def get_json_schema_instruction(schema_name: str) -> str:
    if schema_name == FEATURE_EXTRACTION_SCHEMA:
        return (
            "Return a JSON object with a top-level 'features' array. "
            "Each feature must be an object with string fields 'name' and 'importance'. "
            "The 'importance' value must be one of: low, medium, high."
        )

    raise JsonResponseValidationError(f"Unknown JSON schema: {schema_name}")


def validate_json_response(text: str, schema_name: str | None = None) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JsonResponseValidationError(f"Model returned invalid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise JsonResponseValidationError("Model must return a top-level JSON object")

    if schema_name is None:
        return payload

    if schema_name == FEATURE_EXTRACTION_SCHEMA:
        _validate_feature_extraction_payload(payload)
        return payload

    raise JsonResponseValidationError(f"Unknown JSON schema: {schema_name}")


def _validate_feature_extraction_payload(payload: dict) -> None:
    features = payload.get("features")
    if not isinstance(features, list):
        raise JsonResponseValidationError("Schema requires a 'features' array")

    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            raise JsonResponseValidationError(f"Feature #{index} must be an object")

        name = feature.get("name")
        if not isinstance(name, str) or not name.strip():
            raise JsonResponseValidationError(f"Feature #{index} requires a non-empty string 'name'")

        importance = feature.get("importance")
        if importance not in FEATURE_IMPORTANCE_LEVELS:
            allowed_values = ", ".join(sorted(FEATURE_IMPORTANCE_LEVELS))
            raise JsonResponseValidationError(
                f"Feature #{index} 'importance' must be one of: {allowed_values}"
            )
