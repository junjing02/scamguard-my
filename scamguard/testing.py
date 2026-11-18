"""Run one message through the agent and print the input, every function call and the response."""
import json

from .agent import ScamGuardAgent
from .session import Session


def run_case(llm, text, session=None):
    session = session or Session()
    out = ScamGuardAgent(llm).run_turn(session, text)
    print("=" * 80)
    print("INPUT   :", text)
    print("-" * 80)
    if not out["calls"]:
        print("CALLS   : (none)")
    for c in out["calls"]:
        print(f"CALL    : step {c['step']} | {c['function']} | {json.dumps(c['arguments'], ensure_ascii=False)} | {c['status']}")
        print("          result:", json.dumps(c["result"], ensure_ascii=False)[:300])
    print("-" * 80)
    print("RESPONSE:", out["reply"])
    if out["notes"]:
        print("NOTES   :", out["notes"])
    return out, session
