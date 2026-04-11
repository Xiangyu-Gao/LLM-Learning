"""
schema_check.py — JSON schema compliance evaluator.
====================================================
When evaluating structured-output tasks (tool use, function calling,
data extraction), the primary metric is schema compliance:

  "Does the model output conform to the required JSON schema?"

This is more useful than BLEU because a response with 99% BLEU score
but wrong field names is completely unusable downstream.

Schema format (simplified, no external jsonschema dependency)
--------------------------------------------------------------
A schema is a dict describing expected structure:

    {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age":  {"type": "integer", "minimum": 0},
            "tags": {"type": "array",   "items": {"type": "string"}}
        }
    }

Supported types: "string", "integer", "number", "boolean", "array", "object", "null"
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass
class SchemaResult:
    valid:    bool
    errors:   List[str] = field(default_factory=list)
    raw_json: Optional[Any] = None          # parsed value if valid JSON

    def __bool__(self):
        return self.valid


# ─── Core validator ───────────────────────────────────────────────────────────

def _type_ok(value: Any, type_name: str) -> bool:
    mapping = {
        "string":  str,
        "integer": int,
        "number":  (int, float),
        "boolean": bool,
        "array":   list,
        "object":  dict,
        "null":    type(None),
    }
    expected = mapping.get(type_name)
    if expected is None:
        return True   # unknown type → skip check
    # bool is subclass of int in Python; treat booleans as NOT integers
    if type_name == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _validate(value: Any, schema: Dict, path: str, errors: List[str]) -> None:
    """Recursive schema validator."""
    # Type check
    if "type" in schema:
        if not _type_ok(value, schema["type"]):
            errors.append(
                f"{path}: expected type '{schema['type']}', "
                f"got {type(value).__name__}"
            )
            return   # no point checking further

    # String constraints
    if schema.get("type") == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: string too short (min {schema['minLength']})")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: string too long (max {schema['maxLength']})")
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}: value not in enum {schema['enum']}")

    # Numeric constraints
    if schema.get("type") in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: value {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: value {value} > maximum {schema['maximum']}")

    # Array constraints
    if schema.get("type") == "array":
        if "items" in schema:
            for i, item in enumerate(value):
                _validate(item, schema["items"], f"{path}[{i}]", errors)
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: array too short (min {schema['minItems']} items)")

    # Object constraints
    if schema.get("type") == "object":
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}: missing required field '{key}'")
        properties = schema.get("properties", {})
        for key, sub_schema in properties.items():
            if key in value:
                _validate(value[key], sub_schema, f"{path}.{key}", errors)


def validate_schema(output_str: str, schema: Dict) -> SchemaResult:
    """
    Parse output_str as JSON and validate against schema.

    Parameters
    ----------
    output_str : raw string from the model (may contain markdown fences)
    schema     : dict in the simplified schema format above

    Returns
    -------
    SchemaResult with valid=True and parsed value, or errors list.
    """
    # Strip common markdown code fences
    text = output_str.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else ""

    # Parse JSON
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        return SchemaResult(valid=False, errors=[f"JSON parse error: {e}"])

    # Validate
    errors: List[str] = []
    _validate(value, schema, "$", errors)

    return SchemaResult(valid=(len(errors) == 0), errors=errors, raw_json=value)


def batch_schema_eval(
    outputs: List[str],
    schema:  Dict,
) -> Dict[str, Any]:
    """
    Evaluate a list of model outputs against a schema.

    Returns
    -------
    dict with:
      compliance_rate — fraction of outputs that are valid
      results         — list of SchemaResult per output
      error_summary   — most common error types
    """
    results = [validate_schema(o, schema) for o in outputs]
    n_valid = sum(r.valid for r in results)

    all_errors: List[str] = []
    for r in results:
        all_errors.extend(r.errors)

    # Count error prefixes (e.g. "JSON parse error", "missing required field")
    from collections import Counter
    error_types = Counter(e.split(":")[0].strip() for e in all_errors)

    return {
        "compliance_rate": n_valid / max(len(results), 1),
        "n_valid":         n_valid,
        "n_total":         len(results),
        "results":         results,
        "error_summary":   dict(error_types.most_common(5)),
    }
