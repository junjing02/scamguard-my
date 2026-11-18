"""Tool functions. Pure Python over the small JSON dataset. No network access at all:
URLs are analysed as text and are NEVER visited."""
import json
import os
import re
from datetime import datetime
from urllib.parse import urlparse

from .guardrails import redact_pii

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


PATTERNS = _load("scam_patterns.json")["patterns"]
SCAM_TYPES = _load("scam_types.json")["types"]
ORGS = _load("official_orgs.json")["organisations"]
URL_FLAGS = _load("url_flags.json")

_TLDS = "|".join(sorted(URL_FLAGS["common_tlds_for_extraction"], key=len, reverse=True))
_URL_RE = re.compile(
    r"(?i)((?:https?://|www\.)[^\s<>\"']+|\b(?:[a-z0-9-]+\.)+(?:" + _TLDS + r")\b(?:/[^\s<>\"']*)?)")
_SECOND_LEVEL = {"com", "net", "org", "gov", "edu", "co"}


def _extract_urls(text):
    urls = []
    for m in _URL_RE.finditer(text):
        u = m.group(1).rstrip(".,;:)!?")
        if u not in urls:
            urls.append(u)
    return urls


def _has_brand(host, kw):
    # short keywords (pos, rhb, hlb...) must be a whole label/token, or "compose.com" would look like Pos Malaysia
    if len(kw) <= 3:
        return kw in re.split(r"[.-]", host)
    return kw in host


def _registered_domain(host):
    labels = host.split(".")
    if len(labels) >= 3 and labels[-2] in _SECOND_LEVEL and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


# ------------------------------------------------------------------ tools
def match_scam_patterns(session, text):
    low = text.lower()
    ev = session.evidence
    matched = []
    for p in PATTERNS:
        hits = [kw for kw in p["keywords"]
                if re.search(r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)", low)]
        if hits:
            matched.append({"id": p["id"], "name": p["name"], "matched_keywords": hits,
                            "weight": p["weight"], "explanation": p["explanation"]})
    for m in matched:
        if m["id"] not in ev["pattern_ids"]:
            ev["pattern_ids"].append(m["id"])
            ev["pattern_score"] += m["weight"]
            ev["reasons"].append(f"Pattern: {m['name']} ({', '.join(m['matched_keywords'][:3])})")
    note = ("No known scam pattern matched. This does NOT mean the message is safe: my pattern list is small."
            if not matched else "Matched patterns are signals, not proof.")
    return {"status": "ok", "matches": matched,
            "pattern_score": sum(m["weight"] for m in matched),
            "urls_found": _extract_urls(text), "note": note}


def analyze_url(session, url):
    raw = url.strip()
    parsed = urlparse(raw if "://" in raw else "//" + raw)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return {"status": "error", "error": "Could not read a hostname from that address."}
    reg = _registered_domain(host)
    flags = []

    def add(flag, weight, why):
        flags.append({"flag": flag, "weight": weight, "why": why})

    official = None
    for org in ORGS:
        for d in org["official_domains"]:
            if host == d or host.endswith("." + d):
                official = org["name"]
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
        add("ip_address", 3, "Uses a raw IP address instead of a normal domain name.")
    if host in URL_FLAGS["shorteners"] or reg in URL_FLAGS["shorteners"]:
        add("url_shortener", 2, "A shortened link hides the real destination.")
    if host.split(".")[-1] in URL_FLAGS["suspicious_tlds"]:
        add("suspicious_tld", 2, f".{host.split('.')[-1]} is a TLD commonly abused in scams.")
    if "xn--" in host:
        add("punycode", 3, "Internationalised (look-alike character) domain.")
    if "@" in raw.split("//")[-1].split("/")[0]:
        add("at_symbol", 3, "'@' in the address can disguise the real destination.")
    if not official:
        for org in ORGS:
            if any(_has_brand(host, kw) for kw in org["brand_keywords"]):
                add("brand_lookalike", 4, f"Contains the name of {org['name']} but is NOT one of its official domains.")
                break
    if host.count("-") >= 2:
        add("many_hyphens", 1, "Several hyphens are typical of throwaway look-alike domains.")
    if any(w in host for w in URL_FLAGS["suspicious_words"]):
        add("suspicious_words", 1, "Domain contains words like secure/verify/login that scammers add to look official.")
    if raw.lower().startswith("http://"):
        add("no_https", 1, "Not using HTTPS.")

    score = sum(f["weight"] for f in flags)
    result = {"status": "ok", "url": raw, "host": host, "registered_domain": reg,
              "official_match": official, "flags": flags, "url_score": score,
              "note": "The address was analysed as text only. The website was NOT visited, so its content is unknown."}
    ev = session.evidence
    if raw in ev["urls"]:          # the model repeated a call: do not count the same URL twice
        return dict(result, note=result["note"] + " This URL was already counted in this check.")
    ev["url_score"] += score
    ev["urls"].append(raw)
    ev["url_flags"] += [f["flag"] for f in flags]
    if official:
        ev["official_match"].append(official)
        ev["reasons"].append(f"URL {host} matches the official domain list for {official}.")
    for f in flags:
        ev["reasons"].append(f"URL {host}: {f['why']}")
    return result


def lookup_scam_type(session, keyword):
    q = keyword.lower().strip()
    scored = []
    for t in SCAM_TYPES:
        s = 0
        if q in t["name"].lower():
            s += 3
        s += sum(2 for kw in t["keywords"] if q in kw or kw in q)
        if s:
            scored.append((s, t))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return {"status": "not_found", "message": "No scam type in my dataset matches that keyword.",
                "known_types": [t["name"] for t in SCAM_TYPES]}
    return {"status": "ok", "results": [{"name": t["name"], "signs": t["signs"], "what_to_do": t["what_to_do"]}
                                         for _, t in scored[:3]]}


_GENERIC = {"bank", "the bank", "my bank", "berhad", "malaysia", "government", "agency", "courier"}


def _word_in(needle, hay):
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", hay) is not None


def get_official_contact(session, organisation):
    q = organisation.lower().strip()
    if q in _GENERIC:              # "bank" alone must not silently become Maybank
        return {"status": "not_found", "message": "That name is too general. Please name the organisation.",
                "known_organisations": [o["name"] for o in ORGS]}
    for org in ORGS:
        names = [org["id"], org["name"].lower()] + [a.lower() for a in org["aliases"]]
        if any(q == n or _word_in(n, q) or (len(q) >= 4 and _word_in(q, n)) for n in names):
            return {"status": "ok", "organisation": org["name"], "type": org["type"],
                    "official_domains": org["official_domains"],
                    "contact": org["contact"] or "No verified phone number in my data: use the number on the official website or the back of your card.",
                    "verified_in_dataset": org["verified"],
                    "note": "Always type the official website address yourself; never use links from messages."}
    return {"status": "not_found", "message": "That organisation is not in my dataset, so I cannot give contact details.",
            "known_organisations": [o["name"] for o in ORGS]}


def calculate_risk_score(session):
    ev = session.evidence
    total = ev["pattern_score"] + ev["url_score"]
    if total >= 8:
        level = "High"
    elif total >= 4:
        level = "Medium"
    elif total >= 1:
        level = "Low"
    elif ev["official_match"]:
        level = "Low"
    else:
        level = "Can't tell"
    caveat = ("Heuristic score from a small rule set. It is never a guarantee, and I cannot see the sender or the live website.")
    if level == "Can't tell":
        caveat = "I found no signals either way (or no checks were run), so I cannot judge this message."
    session.last_risk = {"level": level, "score": total, "reasons": list(ev["reasons"])}
    return {"status": "ok", "level": level, "score": total, "reasons": list(ev["reasons"]), "caveat": caveat}


def draft_report(session, summary, scam_type="other"):
    clean, found = redact_pii(summary)
    rid = "R" + datetime.now().strftime("%Y%m%d%H%M%S")
    session.pending_report = {"report_id": rid, "scam_type": scam_type, "summary": clean,
                              "created_at": datetime.now().isoformat(timespec="seconds")}
    return {"status": "draft_created", "report_id": rid, "redacted": found,
            "message": "Draft prepared but NOT submitted. The user must press the Confirm button in the interface."}


TOOL_FUNCS = {
    "match_scam_patterns": match_scam_patterns,
    "analyze_url": analyze_url,
    "lookup_scam_type": lookup_scam_type,
    "get_official_contact": get_official_contact,
    "calculate_risk_score": calculate_risk_score,
    "draft_report": draft_report,
}
