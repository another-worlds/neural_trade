"""PreToolUse hook on the Agent tool (D-061): project roles only, and a ledger line per agent call.

The lead delegates through the project roles in .claude/agents/ so each role's pinned model and effort
apply (implementer, experimenter: Sonnet 5.5 medium; qa: Opus 5.5 medium; qa-deep: Opus 5.5 high;
tracker: Haiku 4.5). A generic agent (general-purpose, claude) would inherit the lead's Opus high and
bypass the scheme, so it is denied with a message naming the roles. Read-only built-ins stay allowed.
Every call, allowed or denied, is appended to .claude/agent_ledger.jsonl (local, gitignored): the
owner's and the lead's record of which role ran, on what, with which override.
Fails open: any error in this script lets the call through and logs nothing.
"""
import datetime
import json
import os
import sys

ROLES = {"implementer", "experimenter", "qa", "qa-deep", "tracker"}
BUILTINS = {"Explore", "Plan", "claude-code-guide", "statusline-setup"}


def main() -> None:
    data = json.load(sys.stdin)
    ti = data.get("tool_input") or {}
    agent_type = ti.get("subagent_type") or ti.get("agent_type") or "general-purpose"
    allowed = agent_type in ROLES or agent_type in BUILTINS
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or "."
    line = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "agent_type": agent_type,
        "model_override": ti.get("model"),
        "description": ti.get("description"),
        "background": ti.get("run_in_background"),
        "decision": "allow" if allowed else "deny",
        "session": data.get("session_id"),
    }
    try:
        with open(os.path.join(root, ".claude", "agent_ledger.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError:
        pass
    if not allowed:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"D-061: '{agent_type}' is not a project role. Use subagent_type implementer, experimenter, "
                "qa (Opus medium), qa-deep (Opus high, on triggers), tracker (Haiku), or the read-only "
                "built-ins Explore, Plan, claude-code-guide. Research goes through a workflow (/research or "
                "the owner's 'ultracode'). See docs/OPERATING_MODEL.md 'Models and effort'."),
        }}))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # fail open
        sys.exit(0)
