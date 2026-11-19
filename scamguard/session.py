"""Per-user session state (the agent's internal state: makes it model-based)."""
import json
import os
from datetime import datetime

REPORT_FILE = os.path.join("reports", "submitted_reports.jsonl")   # simulated report "submission"


class Session:
    """Everything the agent remembers for one user: chat history, call log, evidence and the report draft."""

    def __init__(self):
        self.history = []          # list of (user_text, assistant_text) pairs, plain text only
        self.call_log = []         # every tool call attempted, across all turns
        self.pending_report = None # draft waiting for human confirmation
        self.turn = 0
        self.reset_evidence()

    def reset_evidence(self):
        """Start a new, empty evidence record. It is rebuilt for every message: each check is an independent
        (episodic) task. The tools add to it, and calculate_risk_score reads it."""
        self.evidence = {"pattern_ids": [], "pattern_score": 0, "url_score": 0,
                         "url_flags": [], "urls": [], "official_match": [], "reasons": []}
        self.last_risk = None

    def log(self, step, function, arguments, status, result):
        """Record one tool call or guardrail action. The interface shows this log."""
        self.call_log.append({"turn": self.turn, "step": step, "function": function,
                              "arguments": arguments, "status": status, "result": result})

    # ----- human-in-the-loop report handling (NOT callable by the LLM) -----
    def confirm_report(self):
        """Called by the Confirm button only: save the draft to a local file (a simulation, nothing is sent)."""
        if not self.pending_report:
            return "There is no pending report to confirm."
        os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
        record = dict(self.pending_report, confirmed_at=datetime.now().isoformat(timespec="seconds"))
        with open(REPORT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        rid = self.pending_report["report_id"]
        self.pending_report = None
        return (f"Report {rid} confirmed and saved locally (this is a simulation: nothing was sent to any authority). "
                "To report for real, call the National Scam Response Centre (NSRC) on 997.")

    def cancel_report(self):
        """Called by the Cancel button only: delete the draft."""
        if not self.pending_report:
            return "There is no pending report."
        rid = self.pending_report["report_id"]
        self.pending_report = None
        return f"Report draft {rid} was cancelled. Nothing was saved."
