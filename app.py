# agent.py
# ============================================================
# Liar's Dice Agent Template
# ============================================================
# WHAT TO EDIT:
#   - CONFIG: change model, token limits, etc.
#   - SYSTEM_PROMPT: change the agent's strategy
#   - TOOLS: add tools the agent can call (intermediate)
#   - decide_action(): change the agent loop (advanced)
# ============================================================
#
# Run with:   uv run uvicorn template:app --reload
# ============================================================

import os
import re
import json
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# ============================================================
# CONFIG — edit model, limits, etc.
# ============================================================

CONFIG = {
    "agent_name": "My Liar's Dice Agent",
    "model": "gemini-2.5-flash-lite",
    "max_output_tokens": 1000,
    "version": "1.0.0",
}

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ============================================================
# SYSTEM PROMPT — edit this to change your agent's strategy
# ============================================================

SYSTEM_PROMPT = """
You are an agent playing Liar's Dice. Each player holds a secret serial number
from a dollar bill (8 digits, each digit 0–9). All serial numbers combined form
the "pool" of digits everyone is bidding over.

Rules:
- On your turn, either make a bid or challenge the previous bid.
- A bid is (quantity, digit): "there are at least <quantity> of digit <digit>
  across all players' serial numbers combined."
- A new bid must be strictly greater than the current bid:
    - same digit, higher quantity, OR
    - higher digit (any quantity >= 1), OR
    - higher quantity with higher digit.
  Note: 0 is considered the highest digit (after 9), following standard rules.
- If you challenge and the bid is false (actual count < quantity), the bidder
  may rebid (raise their bid) once. If the rebid is also challenged, we count.
- If you challenge and the bid is true (actual count >= quantity), you lose.
- The loser is eliminated from the game.

Rebid mechanic:
- When you are challenged (can_rebid=true in the game state), you may either:
    - REBID: raise your bid to a higher value (uses action_type "rebid")
    - ACCEPT: accept the challenge and go straight to a count (action_type "accept")
- A rebid cannot itself be rebid — if a rebid is challenged, we count immediately.

You will receive the game state as JSON, including your own serial number.
Use your serial number to inform your bids and challenges.

You may respond in simple shorthand OR as a JSON object.

Simple shorthand (preferred for clarity):
  3x5          — bid: 3 of digit 5
  challenge    — challenge the current bid
  rebid 4x5    — rebid: 4 of digit 5 (only when can_rebid=true)
  accept       — accept the challenge (only when can_rebid=true)

Full JSON (also accepted):
  { "action_type": "bid",       "quantity": 3, "digit": 5, "reasoning": "..." }
  { "action_type": "challenge", "reasoning": "..." }
  { "action_type": "rebid",     "quantity": 4, "digit": 5, "reasoning": "..." }
  { "action_type": "accept",    "reasoning": "..." }
"""

# ============================================================
# DATA MODELS — do not edit (shared schema with the harness)
# ============================================================

class PlayerState(BaseModel):
    player_id: str
    active: bool                     # False if eliminated

class Bid(BaseModel):
    player_id: str
    quantity: int
    digit: int                       # 0–9, where 0 is highest
    is_rebid: bool = False           # True if this bid was made as a rebid

class GameState(BaseModel):
    game_id: str
    round: int
    your_player_id: str
    your_serial_number: str          # your 8-digit serial number, e.g. "37291840"
    players: list[PlayerState]
    bid_history: list[Bid]           # bids so far this round, in order
    current_bid: Optional[Bid] = None
    total_digits: int                # total digits in play (8 × number of active players)
    can_rebid: bool = False          # True when you were just challenged and may rebid or accept

class Action(BaseModel):
    action_type: str                 # "bid", "challenge", "rebid", or "accept"
    quantity: Optional[int] = None   # required for "bid" and "rebid"
    digit: Optional[int] = None      # required for "bid" and "rebid" (0–9, 0 is highest)
    reasoning: Optional[str] = None

class ActionResponse(BaseModel):
    action: Action
    agent_version: str

class AgentInfo(BaseModel):
    agent_name: str
    model: str
    version: str

# ============================================================
# RESPONSE PARSING — parses both shorthand and JSON responses
# ============================================================

def parse_action(text: str) -> Action:
    """Parse the model's response as an Action.

    Accepts shorthand strings like "3x5", "challenge", "rebid 4x5", "accept",
    as well as full JSON objects (with optional markdown code fences).
    """
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Try JSON first
    try:
        data = json.loads(text)
        return Action(**data)
    except (json.JSONDecodeError, Exception):
        pass

    # Try simple shorthand formats
    lower = text.lower().strip()

    if lower in ("challenge", "c"):
        return Action(action_type="challenge")

    if lower in ("accept", "a"):
        return Action(action_type="accept")

    # "NxD" or "N x D" — bid with quantity N, digit D
    m = re.match(r'^(\d+)\s*[x×]\s*(\d+)$', lower)
    if m:
        return Action(action_type="bid", quantity=int(m.group(1)), digit=int(m.group(2)))

    # "rebid NxD" or "rebid N x D"
    m = re.match(r'^rebid\s+(\d+)\s*[x×]\s*(\d+)$', lower)
    if m:
        return Action(action_type="rebid", quantity=int(m.group(1)), digit=int(m.group(2)))

    raise ValueError(f"Could not parse model response as Action.\nResponse was: {text}")

# ============================================================
# AGENT LOGIC — edit this for advanced control of the agent loop
# ============================================================

def decide_action(state: GameState) -> Action:
    response = client.models.generate_content(
        model=CONFIG["model"],
        contents=state.model_dump_json(),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=CONFIG["max_output_tokens"],
        ),
    )

    return parse_action(response.text)

# ============================================================
# API — do not edit
# ============================================================

app = FastAPI()

@app.post("/action", response_model=ActionResponse)
async def action(state: GameState):
    try:
        act = decide_action(state)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ActionResponse(action=act, agent_version=CONFIG["version"])

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/info", response_model=AgentInfo)
async def info():
    return AgentInfo(
        agent_name=CONFIG["agent_name"],
        model=CONFIG["model"],
        version=CONFIG["version"],
    )
