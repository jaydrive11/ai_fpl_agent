"""Pull current player data from the public FPL API."""
from dataclasses import dataclass

import requests

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


@dataclass
class Player:
    id: int
    name: str
    team_id: int
    team_name: str
    position: int            # element_type: 1=GK, 2=DEF, 3=MID, 4=FWD
    price: int               # tenths of millions (55 -> £5.5m)
    xpts: float              # E[points] for this GW — drives XI selection
    ceiling_xpts: float | None = None  # P90 for this GW — drives captain choice
    horizon_xpts: float = 0.0  # Σ E[points] over next H-1 future GWs — drives squad selection


def fetch_bootstrap() -> dict:
    r = requests.get(BOOTSTRAP_URL, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_players(bootstrap: dict) -> list[Player]:
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    players: list[Player] = []
    for e in bootstrap["elements"]:
        # FPL's own next-GW expectation; fall back to season points-per-game
        ep = float(e.get("ep_next") or 0.0)
        if ep == 0.0:
            ep = float(e.get("points_per_game") or 0.0)
        players.append(Player(
            id=e["id"],
            name=e["web_name"],
            team_id=e["team"],
            team_name=teams[e["team"]],
            position=e["element_type"],
            price=e["now_cost"],
            xpts=ep,
        ))
    return players
