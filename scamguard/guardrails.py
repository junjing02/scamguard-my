"""Guardrails enforced in Python (never left to the LLM).

G1  Input guardrail  : refuse tracing / contacting / retaliating against scammers, and money actions.
G2  Injection check  : flag attempts to override the agent's rules (the LLM is warned, output is still guarded).
G3  PII redaction    : IC numbers, card numbers and OTP/TAC digits are removed before the LLM sees them.
G4  Output guardrail : the agent may never certify something as "safe", and may never claim a report was submitted.
G5  Procedure check  : before the final answer, the model must have used the dataset tools and finished the steps.
"""
from __future__ import annotations

import re

# ---------- G3: PII redaction ----------
_IC = re.compile(r"\b\d{6}-?\d{2}-?\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_OTP = re.compile(r"(?i)\b(otp|tac|pin|cvv)\b(\D{0,15})(\d{3,8})\b")


def redact_pii(text: str):
    """Return (clean_text, list_of_found_types)."""
    found = []
    if _IC.search(text):
        found.append("IC number")
        text = _IC.sub("[IC REDACTED]", text)
    if _CARD.search(text):
        found.append("card number")
        text = _CARD.sub("[CARD REDACTED]", text)
    if _OTP.search(text):
        found.append("OTP/TAC/PIN")
        text = _OTP.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
    return text, found


# ---------- G1: blocked requests ----------
# The rules target requests made TO the agent. They must not fire on a pasted scam message
# ("Pay RM1.50 fee", "Transfer RM5000 to the safe account", "call back 03-..."), which is what we analyse.
BLOCK_RULES = [
    ("trace_or_doxx",
     r"\b(trace|track down|locate|identify|find out who|who owns|who is behind|dox|doxx)\b.{0,60}"
     r"\b(the scammers?|scammers?|the sender|the caller|this number|that number|this phone number|the owner)\b",
     "I can't trace, identify or investigate who is behind a number or account. That could target the wrong person "
     "and may be illegal. Please report it to the National Scam Response Centre (NSRC) on 997 or to the police."),
    ("retaliation",
     r"\b(hack|ddos|retaliate|revenge|get back at|scam (them|him|her) back|take down their)\b",
     "I can't help with hacking or retaliating against anyone. Please stop engaging and report it to NSRC (997)."),
    ("contact_scammer",
     r"\b(message|text|call|contact|reply to|respond to|whatsapp)\b.{0,15}"
     r"\b(them|the scammers?|the sender|the caller|that number|this number)\b",
     "I can't contact or reply to scammers for you, and I'd advise you not to engage. Block the sender and report it to NSRC (997)."),
    ("money_action",
     r"\b(transfer|send|pay)\b.{0,40}\b(for me|on my behalf|from my account)\b"
     r"|\b(can|could|will|would) you (please )?(transfer|pay)\b",
     "I can't make payments or transfers, and I never will. Please do not pay anything based on an unverified message."),
]

# ---------- G2: prompt-injection indicators ----------
INJECTION_PATTERNS = [
    r"ignore (all |any |your |the |previous |prior |above )+(instructions|rules|prompt|guidelines)",
    r"disregard (all |any |your |the |previous )*(instructions|rules)",
    r"forget (everything|your (instructions|rules))",
    r"\byou are now\b", r"developer mode", r"system prompt", r"jailbreak",
    r"override (your|the) (rules|safety)", r"pretend (that )?you", r"act as (?!if)",
]


def check_user_input(text: str) -> dict:
    """Returns {blocked, rule, reply, injection}."""
    low = text.lower()
    injection = any(re.search(p, low) for p in INJECTION_PATTERNS)
    for rule, pattern, reply in BLOCK_RULES:
        if re.search(pattern, low):
            return {"blocked": True, "rule": rule, "reply": reply, "injection": injection}
    return {"blocked": False, "rule": None, "reply": None, "injection": injection}


# ---------- G4: output guardrail ----------
UNSAFE_CLAIMS = re.compile(
    r"(100\s?%\s?(safe|legit|genuine)|completely safe|totally safe|definitely safe|guaranteed safe|"
    r"is safe to (click|open|visit|use|trust)|\b(this|it|that) (is not|isn't|is definitely not) a scam|"
    r"definitely (legit|genuine|real)|(there is|there's|it has|it carries|poses) no risk)", re.I)
FALSE_SUBMIT = re.compile(
    r"(report (has been|was|is) (successfully )?(submitted|sent|filed)|"
    r"i('ve| have) (submitted|sent|filed) (the|your) report)", re.I)

SAFE_FALLBACK = ("I can't certify any message or link as safe: I only analyse the text, I can't see the sender "
                 "or the live website. ")


def check_output(reply: str, last_risk: dict | None) -> tuple[str, list]:
    """Return (possibly rewritten reply, list of triggered guardrail names)."""
    triggered = []
    if UNSAFE_CLAIMS.search(reply):
        triggered.append("no_safe_verdict")
        extra = ""
        if last_risk:
            extra = ("Based on my checks the risk level is " + last_risk["level"].upper() + ". " +
                     " ".join(last_risk.get("reasons", [])[:3]))
        reply = SAFE_FALLBACK + extra
    if FALSE_SUBMIT.search(reply):
        triggered.append("no_false_submission")
        reply = FALSE_SUBMIT.sub("report draft is ready but NOT submitted: you must press Confirm", reply)
    return reply, triggered


# ---------- G5: procedure check ----------
def procedure_gap(calls: list) -> str | None:
    """Return a reminder if the model tries to answer before it used the dataset tools and finished the
    procedure, else None. `calls` = this message's call log: the record of which steps are done."""
    done = [c for c in calls if c["status"] == "ok"]
    names = [c["function"] for c in done]
    if not names:
        return ("You answered without calling any tool. Answer ONLY from tool results. Call match_scam_patterns "
                "for a message to check; lookup_scam_type for any question about a kind of scam (for example "
                "e-wallet, parcel, job, investment, love, Macau, bank phishing, tax refund, utility bill); "
                "get_official_contact for an organisation. Only if the question is not about scams at all, "
                "say that you can only answer from your scam dataset.")
    if "match_scam_patterns" in names:
        urls = next(c["result"]["urls_found"] for c in reversed(done) if c["function"] == "match_scam_patterns")
        if names.count("analyze_url") < len(urls):
            return (f"You found these links but did not check them all: {urls}. "
                    "Call analyze_url for each one, then calculate_risk_score.")
        if "calculate_risk_score" not in names:
            return "You checked the patterns but did not calculate the risk. Call calculate_risk_score before you answer."
    return None
