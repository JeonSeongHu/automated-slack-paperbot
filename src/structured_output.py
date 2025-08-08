import json
from typing import Any, Dict, Tuple

from jsonschema import Draft202012Validator, ValidationError


# JSON Schema for one paper line produced by LLM
PAPER_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "Relevancy score",
        "Novelty score",
        "Priority",
        "Reasons for match",
    ],
    "properties": {
        "Relevancy score": {"type": ["integer", "string"], "minimum": 1, "maximum": 10},
        "Novelty score": {"type": ["integer", "string"], "minimum": 1, "maximum": 10},
        "Priority": {
            "type": "string",
            "enum": ["Must-read", "Skim", "Low"],
        },
        "Reasons for match": {"type": "string", "minLength": 3},
        "Venue": {"type": "string"},
        "Project page": {"type": "string"},
    },
    "additionalProperties": True,
}

_validator = Draft202012Validator(PAPER_SCHEMA)


def coerce_and_validate(obj: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    """Coerce common minor issues (string digits to int, priority normalization)
    and validate against schema.
    Returns (ok, fixed_obj, err_msg).
    """
    fixed = dict(obj)
    # ints in strings
    for k in ["Relevancy score", "Novelty score"]:
        v = fixed.get(k)
        if isinstance(v, str):
            v = v.strip()
            if "/" in v:
                v = v.split("/")[0].strip()
            if v.isdigit():
                try:
                    fixed[k] = int(v)
                except Exception:
                    pass
    # priority normalization
    pri = fixed.get("Priority")
    if isinstance(pri, str):
        p = pri.strip().lower()
        if p in ("must read", "must-read", "mustread"):
            fixed["Priority"] = "Must-read"
        elif p in ("skim", "skim-read"):
            fixed["Priority"] = "Skim"
        elif p in ("low", "ignore"):
            fixed["Priority"] = "Low"

    # empty optional fields => empty string
    for k in ["Venue", "Project page"]:
        if fixed.get(k) is None:
            fixed[k] = ""

    try:
        _validator.validate(fixed)
        return True, fixed, ""
    except ValidationError as e:
        return False, fixed, str(e)


