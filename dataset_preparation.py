import nflreadpy as nfl
import json
import polars as pl
# --- 1) Load weekly player stats ---
ps = nfl.load_player_stats(seasons=True, summary_level="week")

ps = ps.filter(ps["position_group"].is_in(["QB", "RB", "WR", "TE"]))

# --- 2) Load schedules ---
sch = nfl.load_schedules(seasons=True)

# Confirm date column
date_col = "gameday"  # confirmed from your error output

# --- 3) Build team/opponent/date key (home + away) ---
home_side = (
    sch.select([
        "season", "week", "game_type",
        "home_team", "away_team", date_col
    ])
    .rename({
        "home_team": "team",
        "away_team": "opponent_team",
        "game_type": "season_type",   # normalize naming
        date_col: "game_date"
    })
)

away_side = (
    sch.select([
        "season", "week", "game_type",
        "away_team", "home_team", date_col
    ])
    .rename({
        "away_team": "team",
        "home_team": "opponent_team",
        "game_type": "season_type",
        date_col: "game_date"
    })
)

sch_key = (
    home_side
    .vstack(away_side)
    .unique(["season", "week", "season_type", "team", "opponent_team"])
)

# --- 4) Join game_date onto player-week rows ---
ps = ps.join(
    sch_key,
    on=["season", "week", "season_type", "team", "opponent_team"],
    how="left"
)

# --- 5) Keep only the stats you want ---
keep_cols = [
    # identifiers
    "player_id", "player_name", "player_display_name",
    "position", "position_group",
    "season", "week", "season_type",
    "team", "opponent_team", "game_date",

    # QB
    "passing_yards", "completions", "attempts",
    "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",

    # WR / TE / RB
    "receiving_yards", "receptions", "receiving_tds",
]

keep_cols = [c for c in keep_cols if c in ps.columns]

dataset = ps.select(keep_cols)
# dataset = dataset.with_columns(
#     pl.col("game_date")
#       .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
#       .alias("game_date")
# )
dataset = dataset.sort(["player_id", "game_date"])

# last60 = (
#     dataset
#     .group_by("player_id", maintain_order=True)
#     .tail(60)
#     .sort(["player_id", "game_date"])
# )

dataset.write_json("nfl_player_gamelogs_raw.json")
with open("nfl_player_gamelogs_raw.json", "r") as f:
    data = json.load(f)

with open("nfl_player_gamelogs.json", "w") as f:
    json.dump(data, f, indent=2)
