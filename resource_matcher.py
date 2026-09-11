"""
Resource Navigator — Eligibility Matching Engine
v0.2 — OpenRouter + DeepSeek V4 Flash

Setup:
    pip install openai
    export OPENROUTER_API_KEY=your-key-here   # get one free at openrouter.ai/keys

Get a free key at: https://openrouter.ai/keys
(No credit card required for account creation; small credit needed to run paid models)

Cost: ~$0.003 per full 15-program run at DeepSeek V4 Flash rates.
Privacy: routed through non-US providers. Set PREFER_PROVIDER below.
"""

import os, json, sqlite3, time
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH = "sf_housing_programs.db"
MODEL   = "deepseek/deepseek-v4-pro"  

# Non-US provider preference — OpenRouter will route here when available.
# Options: "SiliconFlow" (China), "Alibaba" (China), "Venice" (no-log),
#          "DeepSeek" (China direct), None (let OpenRouter balance)
PREFER_PROVIDER = "Venice"       # No-log privacy policy; change to "SiliconFlow" or None to auto-route

PRIORITY_ORDER = {
    "LIKELY": 0,
    "POSSIBLE — VERIFY": 1,
    "INSUFFICIENT DATA — CALL FIRST": 2,
    "UNLIKELY": 3,
}

# ── OpenRouter client ─────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "Set OPENROUTER_API_KEY environment variable.\n"
            "Get a free key at https://openrouter.ai/keys"
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an eligibility assessor for a resource navigation service. Your only job is to determine whether a specific person is likely to qualify for a specific program, based on what the program's eligibility requirements actually say.

You are not a search engine. You do not match on keywords or categories. You read.

---

READING PROTOCOL — follow this exactly, in order, for every program:

1. READ the full eligibility text before forming any judgment. Do not skim. Do not stop at the program title or category label.

2. LIST every requirement you find. Include: income limits, household size limits, geographic restrictions, documentation requirements, demographic criteria, time limits, intake hours, waitlist status, exclusion criteria (e.g. "must not have received services in the last 12 months"), and any other conditions.

3. For each requirement, EVALUATE explicitly against the person's situation:
   - MEETS — the person clearly satisfies this requirement based on what they told us
   - DOES NOT MEET — the person clearly does not satisfy this requirement
   - UNCLEAR — the requirement exists but we don't have enough information to evaluate it
   - UNVERIFIABLE — the program page does not state this requirement clearly enough to evaluate

4. ONLY after completing steps 1-3, form your overall assessment.

5. If ANY requirement is DOES NOT MEET, overall assessment is UNLIKELY.
   If ALL requirements are MEETS, overall assessment is LIKELY.
   If requirements are a mix of MEETS and UNCLEAR/UNVERIFIABLE, overall assessment is POSSIBLE — VERIFY.
   INSUFFICIENT DATA — CALL FIRST is ONLY for programs where the eligibility text has fewer than 2 evaluable requirements (e.g. the description is a single sentence with no criteria listed). Do NOT use it because you are uncertain about outcomes — use POSSIBLE — VERIFY for uncertainty.

---

IMPORTANT: You are assessing eligibility based on what the program's text says, not based on real-time verification. You cannot call the program or check live availability — work with what is written. Uncertainty about real-world outcomes goes into UNCLEAR, not INSUFFICIENT DATA.

UNCERTAINTY RULES:
- If you are not sure whether a requirement is met, say UNCLEAR. Do not guess. Do not round up.
- If the program page does not state intake status, do not assume it is open. Flag as unknown.
- If income is stated as a percentage of AMI and you were not given AMI tables, use the person's stated income and note that exact AMI threshold should be confirmed.
- If a requirement is listed but the person did not provide relevant information, flag it as UNCLEAR rather than assuming it is met.
- UNCLEAR requirements push overall to POSSIBLE — VERIFY, never to INSUFFICIENT DATA.

---

CALL SCRIPT RULE:
For every program assessed LIKELY or POSSIBLE — VERIFY, generate a call script.
The call script is a short paragraph the person can read verbatim or hand to a provider.
It must include: household composition, approximate monthly income, urgency, relevant circumstances, and what they are calling to ask about.
Do NOT write "explain your situation." Write the actual words.

---

OUTPUT FORMAT — return valid JSON only, no prose outside the JSON:

{
  "program_name": "...",
  "overall": "LIKELY | POSSIBLE — VERIFY | UNLIKELY | INSUFFICIENT DATA — CALL FIRST",
  "requirements": [
    {
      "requirement": "...",
      "source_text": "exact quote from program description",
      "evaluation": "MEETS | DOES NOT MEET | UNCLEAR | UNVERIFIABLE",
      "reason": "one sentence"
    }
  ],
  "blocking_requirements": [],
  "unclear_requirements": [],
  "call_script": "...",
  "intake_status": "OPEN | CLOSED | WAITLIST | UNKNOWN",
  "intake_hours": "...",
  "contact": "...",
  "flags": []
}"""


# ── Build user message ────────────────────────────────────────────────────────

def build_user_message(program: dict, person: dict) -> str:
    # Format any community-sourced updates for this program
    community_updates = program.get("_community_updates", [])
    if community_updates:
        lines = []
        for u in community_updates:
            reporter = u.get("reporter_type", "anonymous")
            method   = u.get("contact_method_used") or "unknown method"
            ts       = u.get("submitted_at", "unknown time")
            conf     = u.get("confirmed_count", 0)
            disputed = u.get("disputed_count", 0)
            note     = f' — "{u["notes"]}"' if u.get("notes") else ""
            confirmed_str = f", confirmed by {conf} other(s)" if conf else ""
            disputed_str  = f", disputed by {disputed}" if disputed else ""
            lines.append(
                f"  [{u['field_name']}] reported as \"{u['new_value']}\" "
                f"by {reporter} via {method} on {ts}{confirmed_str}{disputed_str}{note}"
            )
        community_section = (
            "\nCOMMUNITY UPDATES (crowd-sourced, timestamped — treat as supplementary "
            "to the program record above; higher confirmed_count = more reliable):\n"
            + "\n".join(lines)
        )
    else:
        community_section = ""

    return f"""PROGRAM:
Name: {program['name']}
Organization: {program['organization']}
Description: {program['description']}
Eligibility text: {program['eligibility_text']}
Documentation required: {program['documentation_required']}
Intake status: {program['intake_status']}
Intake notes: {program.get('intake_notes') or 'unknown'}
Hours: {program.get('hours') or 'unknown'}
Contact: {program.get('phone') or ''} {program.get('email') or ''} {program.get('website') or ''}
{community_section}
PERSON:
Location: {person['location']}
Urgency: {person['urgency']}
Household size: {person['household_size']}
Children: {person.get('children', 'unknown')}
Monthly income (approximate): {person.get('monthly_income', 'unknown')}
Circumstances: {', '.join(person.get('circumstances', []))}
Additional context: {person.get('additional_context', 'none provided')}"""


# ── Load programs from DB ──────────────────────────────────────────────────────

def load_community_updates(conn, program_id: int) -> list:
    """
    Fetch the most recent active community update per field for a program.
    Returns a list of dicts sorted by submitted_at descending.
    Returns empty list if the program_updates table doesn't exist yet.
    """
    try:
        c = conn.cursor()
        c.execute("""
            SELECT pu.*
            FROM program_updates pu
            INNER JOIN (
                SELECT field_name, MAX(submitted_at) AS latest
                FROM program_updates
                WHERE program_id = ? AND is_active = 1
                GROUP BY field_name
            ) m ON pu.field_name = m.field_name
               AND pu.submitted_at = m.latest
            WHERE pu.program_id = ? AND pu.is_active = 1
            ORDER BY pu.submitted_at DESC
        """, (program_id, program_id))
        return [dict(row) for row in c.fetchall()]
    except Exception:
        return []


def load_programs(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM programs WHERE intake_status != 'CLOSED'")
    rows = []
    for row in c.fetchall():
        p = dict(row)
        p["_community_updates"] = load_community_updates(conn, p["id"])
        rows.append(p)
    conn.close()
    return rows


# ── Match one program ─────────────────────────────────────────────────────────

def match_program(client: OpenAI, program: dict, person: dict) -> dict:
    user_msg = build_user_message(program, person)

    # Build provider routing header if preference is set
    extra_headers = {}
    if PREFER_PROVIDER:
        extra_headers["X-OpenRouter-Provider-Preferences"] = json.dumps({
            "require": [{"name": PREFER_PROVIDER}]
        })

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=2500,
            response_format={"type": "json_object"},
            extra_headers=extra_headers or None,
        )
        raw = response.choices[0].message.content
        if raw is None:
            # Provider returned empty content — log finish reason for debugging
            finish = response.choices[0].finish_reason if response.choices else "no choices"
            refusal = getattr(response.choices[0].message, "refusal", None) if response.choices else None
            reason = f"finish_reason={finish}" + (f", refusal={refusal}" if refusal else "")
            return {
                "program_name": program["name"],
                "overall": "INSUFFICIENT DATA — CALL FIRST",
                "error": f"Provider returned empty response ({reason})",
                "contact": program.get("phone") or program.get("website") or "see program record",
                "flags": [f"Provider returned empty response — try different provider. {reason}"],
            }
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "program_name": program["name"],
            "overall": "INSUFFICIENT DATA — CALL FIRST",
            "error": f"JSON parse error: {e} | raw={str(raw)[:200]}",
            "contact": program.get("phone") or program.get("website") or "see program record",
            "flags": [f"JSON parse error — model output was not valid JSON: {str(raw)[:100]}"],
        }
    except Exception as e:
        return {
            "program_name": program["name"],
            "overall": "INSUFFICIENT DATA — CALL FIRST",
            "error": str(e),
            "contact": program.get("phone") or program.get("website") or "see program record",
            "flags": [f"Matching error: {str(e)[:80]}"],
        }


# ── Run full match ────────────────────────────────────────────────────────────

def run_matching(person: dict, db_path: str = DB_PATH, delay: float = 0.3) -> dict:
    client   = get_client()
    programs = load_programs(db_path)

    print(f"\nMatching {len(programs)} programs via {MODEL} ({PREFER_PROVIDER or 'balanced routing'})...")
    print(f"Scenario: {person.get('urgency')} urgency | household of {person.get('household_size')}\n")

    results = []
    for i, program in enumerate(programs, 1):
        print(f"  [{i:2}/{len(programs)}] {program['name'][:60]}", end=" ", flush=True)
        result = match_program(client, program, person)
        results.append(result)
        print(f"→ {result.get('overall', '?')}")
        if delay and i < len(programs):
            time.sleep(delay)

    results.sort(key=lambda r: PRIORITY_ORDER.get(r.get("overall", "UNLIKELY"), 3))

    output = {
        "person": person,
        "model": MODEL,
        "provider_preference": PREFER_PROVIDER,
        "total_programs_checked": len(programs),
        "results": results,
        "action_list": [
            f"{r['overall']}: {r['program_name']} — {r.get('contact', '')}"
            for r in results
            if r.get("overall") in ("LIKELY", "POSSIBLE — VERIFY")
        ]
    }
    return output


# ── Example person — edit to test different scenarios ─────────────────────────

EXAMPLE_PERSON = {
    "location": "San Francisco, CA",
    "urgency": "weeks",
    "household_size": 4,
    "children": "yes — 3 minor children under 18",
    "monthly_income": "approximately $5,000-5,500/mo gross (around 35-40% AMI for 4-person SF household)",
    "circumstances": [
        "facing eviction",
        "authorized occupant / subtenant (not primary leaseholder)",
        "employed full-time",
        "single parent",
        "children under 18",
        "limited savings (~$20k)",
        "pet"
    ],
    "additional_context": (
        "Primary tenant vacated around 2023. User notified landlord in writing in 2023 "
        "that primary tenant had moved out. Landlord refused rent payments directly from user. "
        "Eviction is against primary tenant for COVID-era back rent nonpayment. "
        "User was not party to original eviction case and was not served directly with Unlawful Detainer. "
        "Needs 2BR+ for family of 4. Has approximately $20k savings she is trying to preserve."
    ),
}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_matching(EXAMPLE_PERSON)

    output_file = "matching_results_latest.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_file}")

    print("\n── RECOMMENDED ACTIONS ──────────────────────────────────────")
    for action in results["action_list"]:
        print(f"  {action}")
    print("─────────────────────────────────────────────────────────────\n")
