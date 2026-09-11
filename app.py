"""
Resource Navigator — Community Update API
Flask skeleton: Waze-style crowdsourced resource database

Endpoints
---------
GET  /programs          — all programs + latest community updates + confidence scores
GET  /programs/<id>     — single program
POST /update            — submit a community update (returns gamification response)
POST /vote              — confirm or dispute an existing update
POST /match             — run LLM eligibility matching for a person (slow — ~1 min)

Run
---
    pip install flask flask-cors
    set NAVIGATOR_DB=sf_housing_programs.db   # or export on Mac/Linux
    python app.py
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from math import exp

from flask import Flask, g, jsonify, request
from flask_cors import CORS

# Load .env file for local development (no-op in production where env vars are set directly)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)  # allow browser frontend on a different port during dev

DB_PATH = os.environ.get("NAVIGATOR_DB", "sf_housing_programs.db")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db:
        db.close()

# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def confidence_score(submitted_at_str, confirmed, disputed, reporter_type):
    """
    0.0 – 1.0.
    Decays exponentially with age (~10-day half-life).
    Boosted by confirmations, penalised by disputes.
    Program staff and social workers start higher.
    """
    try:
        submitted = datetime.fromisoformat(submitted_at_str).replace(tzinfo=timezone.utc)
    except Exception:
        submitted = datetime.now(timezone.utc)

    age_days = (datetime.now(timezone.utc) - submitted).total_seconds() / 86400
    base = exp(-age_days / 14)  # still ~0.95 after 1 day, ~0.61 after 7, ~0.37 after 14

    boost = (confirmed * 0.08) - (disputed * 0.15)
    if reporter_type == "program_staff":
        boost += 0.20
    elif reporter_type == "social_worker":
        boost += 0.10

    return round(min(1.0, max(0.0, base + boost)), 3)

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

VALID_FIELDS = {
    "intake_status", "serves", "immigration_agnostic",
    "coordinated_entry_required", "hours", "phone",
    "email", "intake_notes", "general",
}

def ensure_session(db, session_id):
    db.execute(
        "INSERT OR IGNORE INTO contributor_sessions (session_id) VALUES (?)",
        (session_id,),
    )
    db.execute(
        "UPDATE contributor_sessions SET last_active = CURRENT_TIMESTAMP WHERE session_id = ?",
        (session_id,),
    )

def get_session_stats(db, session_id):
    row = db.execute(
        "SELECT updates_submitted, updates_confirmed FROM contributor_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return dict(row) if row else {"updates_submitted": 0, "updates_confirmed": 0}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_update(u):
    return {
        "update_id":      u["id"],
        "value":          u["new_value"],
        "reporter_type":  u["reporter_type"],
        "submitted_at":   u["submitted_at"],
        "notes":          u["notes"],
        "confirmed":      u["confirmed_count"],
        "disputed":       u["disputed_count"],
        "confidence":     confidence_score(
                              u["submitted_at"], u["confirmed_count"],
                              u["disputed_count"], u["reporter_type"],
                          ),
    }

def _gamification_message(count):
    if count == 1:
        return "First update — you started the chain."
    elif count < 5:
        return f"{count} updates. The information is fresher because of you."
    elif count < 15:
        return f"{count} updates. You're one of the people keeping this accurate."
    elif count < 30:
        return f"{count} updates. Consistent. That matters."
    else:
        return f"{count} updates. You've been here. Thank you."

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def serve_index():
    return app.send_static_file("index.html")


@app.get("/programs")
def list_programs():
    """All programs with their latest active community update per field."""
    db = get_db()
    programs = db.execute("SELECT * FROM programs ORDER BY name").fetchall()

    # One query: most recent active update per program+field
    updates_raw = db.execute("""
        SELECT pu.*
        FROM program_updates pu
        INNER JOIN (
            SELECT program_id, field_name, MAX(submitted_at) AS latest
            FROM program_updates
            WHERE is_active = 1
            GROUP BY program_id, field_name
        ) m ON pu.program_id = m.program_id
           AND pu.field_name = m.field_name
           AND pu.submitted_at = m.latest
        WHERE pu.is_active = 1
    """).fetchall()

    updates_by_program: dict = {}
    for u in updates_raw:
        pid = u["program_id"]
        updates_by_program.setdefault(pid, {})[u["field_name"]] = _format_update(u)

    return jsonify([
        {**dict(p), "community_updates": updates_by_program.get(p["id"], {})}
        for p in programs
    ])


@app.get("/programs/<int:program_id>")
def get_program(program_id):
    """Single program with all active community updates."""
    db = get_db()
    p = db.execute("SELECT * FROM programs WHERE id = ?", (program_id,)).fetchone()
    if not p:
        return jsonify({"error": "not found"}), 404

    updates_raw = db.execute("""
        SELECT * FROM program_updates
        WHERE program_id = ? AND is_active = 1
        ORDER BY field_name, submitted_at DESC
    """, (program_id,)).fetchall()

    updates: dict = {}
    for u in updates_raw:
        # keep only the most recent per field (already ordered DESC)
        if u["field_name"] not in updates:
            updates[u["field_name"]] = _format_update(u)

    return jsonify({**dict(p), "community_updates": updates})


@app.post("/update")
def submit_update():
    """
    Submit a community update for a program field.

    Required body fields:
        program_id      int
        field_name      str  (one of VALID_FIELDS)
        new_value       str

    Optional:
        session_id          str   (UUID; generated server-side if omitted)
        reporter_type       str   anonymous | social_worker | program_staff |
                                  community_member | person_seeking
        contact_method_used str   phone | email | walk_in | website
        notes               str   free-text context
    """
    data = request.get_json(force=True)
    db = get_db()

    # Validate
    missing = [f for f in ("program_id", "field_name", "new_value") if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing: {missing}"}), 400
    if data["field_name"] not in VALID_FIELDS:
        return jsonify({"error": f"field_name must be one of: {sorted(VALID_FIELDS)}"}), 400

    program = db.execute(
        "SELECT id, name FROM programs WHERE id = ?", (data["program_id"],)
    ).fetchone()
    if not program:
        return jsonify({"error": "program not found"}), 404

    session_id = data.get("session_id") or str(uuid.uuid4())
    ensure_session(db, session_id)

    # Supersede older updates for same program + field
    db.execute(
        "UPDATE program_updates SET is_active = 0 WHERE program_id = ? AND field_name = ? AND is_active = 1",
        (data["program_id"], data["field_name"]),
    )

    cursor = db.execute("""
        INSERT INTO program_updates
            (program_id, session_id, field_name, new_value,
             reporter_type, contact_method_used, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        data["program_id"],
        session_id,
        data["field_name"],
        data["new_value"],
        data.get("reporter_type", "anonymous"),
        data.get("contact_method_used"),
        data.get("notes"),
    ))
    update_id = cursor.lastrowid

    db.execute(
        "UPDATE contributor_sessions SET updates_submitted = updates_submitted + 1 WHERE session_id = ?",
        (session_id,),
    )
    db.commit()

    stats = get_session_stats(db, session_id)
    total = db.execute("SELECT COUNT(*) FROM program_updates").fetchone()[0]

    return jsonify({
        "ok": True,
        "update_id":  update_id,
        "session_id": session_id,
        "program":    program["name"],
        "field":      data["field_name"],
        "value":      data["new_value"],
        "gamification": {
            "your_updates":            stats["updates_submitted"],
            "your_confirmed":          stats["updates_confirmed"],
            "community_updates_total": total,
            "message": _gamification_message(stats["updates_submitted"]),
        },
    }), 201


@app.post("/vote")
def vote_on_update():
    """
    Confirm or dispute an existing update.

    Body: { "update_id": 5, "session_id": "...", "vote": "confirm" | "dispute" }
    """
    data = request.get_json(force=True)
    db = get_db()

    missing = [f for f in ("update_id", "session_id", "vote") if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing: {missing}"}), 400
    if data["vote"] not in ("confirm", "dispute"):
        return jsonify({"error": "vote must be 'confirm' or 'dispute'"}), 400

    u = db.execute(
        "SELECT * FROM program_updates WHERE id = ?", (data["update_id"],)
    ).fetchone()
    if not u:
        return jsonify({"error": "update not found"}), 404
    if u["session_id"] == data["session_id"]:
        return jsonify({"error": "can't vote on your own update"}), 400

    ensure_session(db, data["session_id"])

    try:
        db.execute(
            "INSERT INTO update_votes (update_id, session_id, vote) VALUES (?, ?, ?)",
            (data["update_id"], data["session_id"], data["vote"]),
        )
    except sqlite3.IntegrityError:
        return jsonify({"error": "already voted"}), 409

    if data["vote"] == "confirm":
        db.execute(
            "UPDATE program_updates SET confirmed_count = confirmed_count + 1 WHERE id = ?",
            (data["update_id"],),
        )
        if u["session_id"]:
            db.execute(
                "UPDATE contributor_sessions SET updates_confirmed = updates_confirmed + 1 WHERE session_id = ?",
                (u["session_id"],),
            )
    else:
        db.execute(
            "UPDATE program_updates SET disputed_count = disputed_count + 1 WHERE id = ?",
            (data["update_id"],),
        )

    db.commit()
    return jsonify({"ok": True, "vote": data["vote"]})


# ---------------------------------------------------------------------------

@app.post("/match")
def match_resources():
    """
    Run LLM eligibility matching for a person.
    Returns ranked results (LIKELY first, UNLIKELY last).

    This is slow — expect ~60 seconds for 15 programs.
    The frontend should show a loading state.

    Body: a person dict, e.g.
    {
        "location": "San Francisco, CA",
        "urgency": "weeks",
        "household_size": 4,
        "children": "yes — 3 minor children under 18",
        "monthly_income": "~$5,000/mo gross (~35% AMI)",
        "circumstances": ["facing eviction", "single parent"],
        "additional_context": "..."
    }
    """
    person = request.get_json(force=True)
    if not person:
        return jsonify({"error": "Person details required in request body"}), 400

    required = ("location", "household_size")
    missing = [f for f in required if not person.get(f)]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        from resource_matcher import run_matching
        results = run_matching(person, db_path=DB_PATH, delay=0.3)
        return jsonify(results)
    except ValueError as e:
        # Missing API key etc.
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
