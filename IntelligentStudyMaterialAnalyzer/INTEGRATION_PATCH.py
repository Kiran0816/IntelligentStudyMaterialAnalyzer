# ═══════════════════════════════════════════════════════════════════════════
# HOW TO INTEGRATE agent_routes.py INTO YOUR EXISTING app.py
# Apply these two minimal edits — nothing else in app.py changes.
# ═══════════════════════════════════════════════════════════════════════════

# ── EDIT 1: Add two imports near the top of app.py ──────────────────────────
# Place these lines immediately after your existing module imports,
# e.g. after "from modules.mcq_generator import generate_mcqs"

from agent_routes import agent_bp, init_agent      # ← ADD THIS LINE


# ── EDIT 2: Register the blueprint + build the graph after init_db() ────────
# In your existing app.py, find this block (around line 27):
#
#     # Initialize Database on startup
#     init_db()
#
# Change it to:

# Initialize Database on startup
# init_db()           ← your existing line (keep it)
# init_agent(app)     ← ADD THIS LINE directly below init_db()
# app.register_blueprint(agent_bp)   ← ADD THIS LINE

# ── That's it. ──────────────────────────────────────────────────────────────
# Your existing routes are completely untouched.
# New routes available after the edit:
#
#   POST /api/agent/ask/<upload_id>          ← main entry point
#   GET  /api/agent/flashcards/<upload_id>   ← cached flashcards
#   GET  /api/agent/revision-notes/<upload_id> ← cached revision notes
#   GET  /api/agent/capabilities             ← UI hint endpoint


# ── Example of the final init block in app.py ───────────────────────────────
"""
# Initialize Database on startup
init_db()
init_agent(app)
app.register_blueprint(agent_bp)
"""
