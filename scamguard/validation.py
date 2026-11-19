"""Tool schemas (shown to the LLM) + validation (enforced before ANY function runs).

One source of truth: the same schema is used to tell the LLM what it may call and to reject bad calls.
"""
import json
import re

from .tools import SCAM_TYPES


def _fn(name, description, properties, required):
    """Build one tool schema in the OpenAI / Hugging Face "function" format."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOL_SCHEMAS = [
    _fn("match_scam_patterns",
        "Check the FULL text of a suspicious message against known scam patterns. Also lists any URLs found in it.",
        {"text": {"type": "string", "minLength": 3, "maxLength": 2000, "description": "The full message text."}},
        ["text"]),
    _fn("analyze_url",
        "Analyse one URL/domain as text (never visits it). Call once for each URL found in the message.",
        {"url": {"type": "string", "minLength": 4, "maxLength": 300, "description": "The URL or domain."}},
        ["url"]),
    _fn("lookup_scam_type",
        "Look up a known scam type (e.g. parcel, macau, job, love) to explain its typical signs and what to do.",
        {"keyword": {"type": "string", "minLength": 2, "maxLength": 60}},
        ["keyword"]),
    _fn("get_official_contact",
        "Get official domains and contact details for a bank, agency or courier from the verified dataset.",
        {"organisation": {"type": "string", "minLength": 2, "maxLength": 60}},
        ["organisation"]),
    _fn("calculate_risk_score",
        "Combine the evidence gathered by the other tools into a risk level. Call this LAST, after the checks. Takes no arguments.",
        {}, []),
    _fn("draft_report",
        "Draft a scam report for the user. It is NOT submitted: the user must confirm in the interface. Only call if the user wants to report.",
        {"summary": {"type": "string", "minLength": 10, "maxLength": 600, "description": "Short summary of the scam attempt."},
         "scam_type": {"type": "string", "enum": [t["id"] for t in SCAM_TYPES] + ["other"]}},
        ["summary"]),
]

# the allow list: function name -> its parameter rules
ALLOWED = {s["function"]["name"]: s["function"]["parameters"] for s in TOOL_SCHEMAS}
_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


def _check_value(key, value, spec):
    """Type, length and enum checks for one argument. Returns (error or None, value with strings trimmed)."""
    expected = _TYPES[spec["type"]]
    if isinstance(value, bool) and spec["type"] != "boolean":   # in Python, True is also an int
        return f"Argument '{key}' must be {spec['type']}.", value
    if not isinstance(value, expected):
        return f"Argument '{key}' must be {spec['type']}, got {type(value).__name__}.", value
    if spec["type"] == "string":
        value = value.strip()
        if len(value) < spec.get("minLength", 0):
            return f"Argument '{key}' is too short (min {spec['minLength']}).", value
        if len(value) > spec.get("maxLength", 10**9):
            return f"Argument '{key}' is too long (max {spec['maxLength']}).", value
        if "enum" in spec and value not in spec["enum"]:
            return f"Argument '{key}' must be one of {spec['enum']}.", value
    return None, value


def validate_call(name, arguments):
    """Return (ok, error_message, cleaned_arguments)."""
    if not isinstance(name, str) or name not in ALLOWED:
        return False, f"Function '{name}' is not on the allow list. Allowed: {sorted(ALLOWED)}", None
    if isinstance(arguments, str):           # some models send the arguments as a JSON string
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return False, "Arguments are not valid JSON.", None
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return False, "Arguments must be a JSON object.", None

    props, required = ALLOWED[name]["properties"], ALLOWED[name]["required"]
    for key in arguments:
        if key not in props:
            return False, f"Unexpected argument '{key}' for {name}.", None
    for key in required:
        if key not in arguments:
            return False, f"Missing required argument '{key}' for {name}.", None
    cleaned = {}
    for key, value in arguments.items():
        error, cleaned[key] = _check_value(key, value, props[key])
        if error:
            return False, error, None

    # extra, function-specific rules
    if name == "analyze_url":
        u = cleaned["url"]
        if re.search(r"\s", u):
            return False, "URL must not contain spaces or line breaks.", None
        scheme = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*):", u)
        if scheme and scheme.group(1).lower() not in ("http", "https"):
            return False, "Only http/https addresses (or bare domains) are accepted.", None
    return True, None, cleaned
