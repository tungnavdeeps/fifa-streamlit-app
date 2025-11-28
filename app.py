import datetime
import pandas as pd
import streamlit as st
import gspread
import matplotlib.pyplot as plt
import numpy as np

from google.oauth2.service_account import Credentials

# ---- GLOBAL DARK THEME FOR CHARTS ----
plt.style.use("dark_background")
plt.rcParams.update({
    "text.color": "white",
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.edgecolor": "white",
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "legend.facecolor": "black",
    "legend.edgecolor": "white",
})

# =========================
# CONFIG – EDIT THESE
# =========================

# Full Google Sheets URL (NOT just the ID)
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1-82tJW2-y5mkt0b0qn4DPWj5sL-yOjKgCBKizUSzs9I/edit?gid=0#gid=0"

# Worksheet (tab) names – must match the tabs in your "FIFA Tracker" sheet
WORKSHEET_1V1 = "Matches_1v1"
WORKSHEET_2V2 = "Matches_2v2"
WORKSHEET_2V1 = "Matches_2v1"

# Game versions
GAME_OPTIONS = ["FIFA 24", "FIFA 25", "FIFA 26"]

# =========================
# GOOGLE SHEETS AUTH
# =========================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
    
@st.cache_resource
def get_gsheet_client():
    """
    Create and cache a gspread client using credentials stored in Streamlit secrets.
    """
    creds_info = st.secrets["gcp_service_account"]  # matches [gcp_service_account] in secrets.toml
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client


def load_sheet(worksheet_name: str) -> pd.DataFrame:
    """
    Load a worksheet by name from the configured Google Sheet, returning a DataFrame.

    If the sheet is empty or inaccessible, returns an empty DataFrame.
    """
    try:
        client = get_gsheet_client()
        # Use the full URL, not an ID
        sheet = client.open_by_url(SPREADSHEET_URL).worksheet(worksheet_name)
        records = sheet.get_all_records()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)
    except Exception as e:
        st.warning(f"⚠️ Could not load worksheet '{worksheet_name}': {e}")
        return pd.DataFrame()

def load_matches_1v1() -> pd.DataFrame:
    """
    Load 1v1 matches from the Matches_1v1 sheet and normalize columns/types.
    Expected columns:
        date, game, player1, team1, score1, xG1, result1,
        player2, team2, score2, xG2, result2
    """
    expected_cols = [
        "date",
        "game",
        "player1",
        "team1",
        "score1",
        "xG1",
        "result1",
        "player2",
        "team2",
        "score2",
        "xG2",
        "result2",
    ]

    df = load_sheet(WORKSHEET_1V1)
    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    # Ensure all expected columns exist
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    # Type conversions
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score1"] = pd.to_numeric(df["score1"], errors="coerce")
    df["score2"] = pd.to_numeric(df["score2"], errors="coerce")

    return df[expected_cols]


def load_matches_2v2() -> pd.DataFrame:
    """
    Load 2v2 matches from the Matches_2v2 sheet and normalize columns/types.
    Expected columns:
        date, game, team1_name, team1_players, score1, xG1, result1,
              team2_name, team2_players, score2, xG2, result2
    """
    expected_cols = [
        "date",
        "game",
        "team1_name",
        "team1_players",
        "score1",
        "xG1",
        "result1",
        "team2_name",
        "team2_players",
        "score2",
        "xG2",
        "result2",
    ]

    df = load_sheet(WORKSHEET_2V2)
    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score1"] = pd.to_numeric(df["score1"], errors="coerce")
    df["score2"] = pd.to_numeric(df["score2"], errors="coerce")

    return df[expected_cols]


def load_matches_2v1() -> pd.DataFrame:
    """
    Load 2v1 matches from the Matches_2v1 sheet and normalize columns/types.

    Convention:
    - team1_name / team1_players = the 2-player side (e.g. "Sue & Alex")
    - team2_name / team2_players = the solo side (e.g. "Jordan")
    """
    expected_cols = [
        "date",
        "game",
        "team1_name",
        "team1_players",
        "score1",
        "xG1",
        "result1",
        "team2_name",
        "team2_players",
        "score2",
        "xG2",
        "result2",
    ]

    df = load_sheet(WORKSHEET_2V1)
    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score1"] = pd.to_numeric(df["score1"], errors="coerce")
    df["score2"] = pd.to_numeric(df["score2"], errors="coerce")

    return df[expected_cols]

def append_match_1v1(
    date, game, player1, team1, score1, xG1, player2, team2, score2, xG2
):
    client = get_gsheet_client()
    sheet = client.open_by_url(SPREADSHEET_URL).worksheet(WORKSHEET_1V1)

    if score1 > score2:
        result1, result2 = "W", "L"
    elif score1 < score2:
        result1, result2 = "L", "W"
    else:
        result1 = result2 = "D"

    row = [
        str(date),
        game,
        player1,
        team1,
        int(score1),
        float(xG1),
        result1,
        player2,
        team2,
        int(score2),
        float(xG2),
        result2,
    ]
    sheet.append_row(row)



def append_match_2v2(
    date,
    game,
    team1_name,
    team1_players,
    score1,
    xG1,
    team2_name,
    team2_players,
    score2,
    xG2,
):
    client = get_gsheet_client()
    sheet = client.open_by_url(SPREADSHEET_URL).worksheet(WORKSHEET_2V2)

    if score1 > score2:
        result1, result2 = "W", "L"
    elif score1 < score2:
        result1, result2 = "L", "W"
    else:
        result1 = result2 = "D"

    row = [
        str(date),
        game,
        team1_name,
        team1_players,
        int(score1),
        float(xG1),
        result1,
        team2_name,
        team2_players,
        int(score2),
        float(xG2),
        result2,
    ]
    sheet.append_row(row)

def append_match_2v1(
    date,
    game,
    team1_name,
    team1_players,
    score1,
    xG1,
    team2_name,
    team2_players,
    score2,
    xG2,
):
    client = get_gsheet_client()
    sheet = client.open_by_url(SPREADSHEET_URL).worksheet(WORKSHEET_2V1)

    if score1 > score2:
        result1, result2 = "W", "L"
    elif score1 < score2:
        result1, result2 = "L", "W"
    else:
        result1 = result2 = "D"

    row = [
        str(date),
        game,
        team1_name,
        team1_players,
        int(score1),
        float(xG1),
        result1,
        team2_name,
        team2_players,
        int(score2),
        float(xG2),
        result2,
    ]
    sheet.append_row(row)

# =========================
# ELO RATING (1v1 only)
# =========================

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a, rating_b, score_a, k=20):
    exp_a = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b


def compute_ratings_1v1(df: pd.DataFrame, game: str, base_rating=1000):
    ratings = {}
    if df.empty:
        return ratings

    df_game = df[df["game"] == game].copy()
    if df_game.empty:
        return ratings

    df_game = df_game.sort_values(by="date", na_position="last")

    for _, row in df_game.iterrows():
        p1, p2 = row["player1"], row["player2"]
        s1, s2 = row["score1"], row["score2"]

        if pd.isna(p1) or pd.isna(p2) or pd.isna(s1) or pd.isna(s2):
            continue

        ratings.setdefault(p1, base_rating)
        ratings.setdefault(p2, base_rating)

        if s1 > s2:
            score_a = 1.0
        elif s1 < s2:
            score_a = 0.0
        else:
            score_a = 0.5

        new_p1, new_p2 = update_elo(ratings[p1], ratings[p2], score_a)
        ratings[p1], ratings[p2] = new_p1, new_p2

    return ratings

# =========================
# LEADERBOARDS & STATS
# =========================

def build_player_leaderboard_1v1(df: pd.DataFrame, game: str) -> pd.DataFrame:
    df_game = df[df["game"] == game].copy()
    if df_game.empty:
        return pd.DataFrame(
            columns=[
                "player",
                "games",
                "wins",
                "draws",
                "losses",
                "goals_for",
                "goals_against",
                "goal_diff",
                "avg_goals_for",
                "avg_goals_against",
                "win_pct",
                "elo_rating",
            ]
        )

    rows = []
    for _, row in df_game.iterrows():
        p1, p2 = row["player1"], row["player2"]
        s1, s2 = row["score1"], row["score2"]

        if pd.isna(p1) or pd.isna(p2) or pd.isna(s1) or pd.isna(s2):
            continue

        if s1 > s2:
            r1, r2 = "W", "L"
        elif s1 < s2:
            r1, r2 = "L", "W"
        else:
            r1 = r2 = "D"

        rows.append(
            {
                "player": p1,
                "goals_for": s1,
                "goals_against": s2,
                "result": r1,
            }
        )
        rows.append(
            {
                "player": p2,
                "goals_for": s2,
                "goals_against": s1,
                "result": r2,
            }
        )

    stats_df = pd.DataFrame(rows)
    grouped = stats_df.groupby("player").agg(
        games=("result", "count"),
        wins=("result", lambda x: (x == "W").sum()),
        draws=("result", lambda x: (x == "D").sum()),
        losses=("result", lambda x: (x == "L").sum()),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )

    grouped["goal_diff"] = grouped["goals_for"] - grouped["goals_against"]
    grouped["avg_goals_for"] = grouped["goals_for"] / grouped["games"]
    grouped["avg_goals_against"] = grouped["goals_against"] / grouped["games"]
    grouped["win_pct"] = grouped["wins"] / grouped["games"]

    ratings = compute_ratings_1v1(df, game)
    grouped["elo_rating"] = grouped.index.map(
        lambda p: round(ratings.get(p, 1000))
    )

    grouped = grouped.sort_values(
        by=["elo_rating", "wins"], ascending=False
    ).reset_index()

    return grouped


def build_team_leaderboard_2v2(df: pd.DataFrame, game: str) -> pd.DataFrame:
    df_game = df[df["game"] == game].copy()
    if df_game.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "players",
                "games",
                "wins",
                "draws",
                "losses",
                "goals_for",
                "goals_against",
                "goal_diff",
                "avg_goals_for",
                "avg_goals_against",
                "win_pct",
            ]
        )

    rows = []
    for _, row in df_game.iterrows():
        t1, t2 = row["team1_name"], row["team2_name"]
        p1_players, p2_players = row["team1_players"], row["team2_players"]
        s1, s2 = row["score1"], row["score2"]

        if pd.isna(t1) or pd.isna(t2) or pd.isna(s1) or pd.isna(s2):
            continue

        if s1 > s2:
            r1, r2 = "W", "L"
        elif s1 < s2:
            r1, r2 = "L", "W"
        else:
            r1 = r2 = "D"

        rows.append(
            {
                "team": t1,
                "players": p1_players,
                "goals_for": s1,
                "goals_against": s2,
                "result": r1,
            }
        )
        rows.append(
            {
                "team": t2,
                "players": p2_players,
                "goals_for": s2,
                "goals_against": s1,
                "result": r2,
            }
        )

    stats_df = pd.DataFrame(rows)
    grouped = stats_df.groupby("team").agg(
        players=("players", "first"),
        games=("result", "count"),
        wins=("result", lambda x: (x == "W").sum()),
        draws=("result", lambda x: (x == "D").sum()),
        losses=("result", lambda x: (x == "L").sum()),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )

    grouped["goal_diff"] = grouped["goals_for"] - grouped["goals_against"]
    grouped["avg_goals_for"] = grouped["goals_for"] / grouped["games"]
    grouped["avg_goals_against"] = grouped["goals_against"] / grouped["games"]
    grouped["win_pct"] = grouped["wins"] / grouped["games"]

    grouped = grouped.sort_values(
        by=["win_pct", "goal_diff"], ascending=False
    ).reset_index()

    return grouped


def head_to_head_1v1(df: pd.DataFrame, game: str, p1: str, p2: str) -> pd.DataFrame:
    df_game = df[df["game"] == game].copy()
    mask = (
        ((df_game["player1"] == p1) & (df_game["player2"] == p2))
        | ((df_game["player1"] == p2) & (df_game["player2"] == p1))
    )
    return df_game[mask].copy()


def _prediction_components_1v1(df: pd.DataFrame, game: str, player_a: str, player_b: str):
    """
    Internal helper: compute all components for the hybrid model.
    Returns a dict with:
      ra, rb, elo_prob, h2h_prob, xg_prob, final_prob
    """
    # -------- 1. ELO component (vs everyone in this game) --------
    ratings = compute_ratings_1v1(df, game)
    ra = ratings.get(player_a, 1000)
    rb = ratings.get(player_b, 1000)
    elo_prob = expected_score(ra, rb)  # ELO win chance for A

    # -------- 2. Head-to-head + recent form --------
    h2h_df = head_to_head_1v1(df, game, player_a, player_b)
    h2h_prob = None
    xg_prob = None

    if not h2h_df.empty:
        total_games = len(h2h_df)

        # Overall H2H (draw = 0.5)
        wins_a = (
            ((h2h_df["player1"] == player_a) & (h2h_df["score1"] > h2h_df["score2"])).sum()
            + ((h2h_df["player2"] == player_a) & (h2h_df["score2"] > h2h_df["score1"])).sum()
        )
        wins_b = (
            ((h2h_df["player1"] == player_b) & (h2h_df["score1"] > h2h_df["score2"])).sum()
            + ((h2h_df["player2"] == player_b) & (h2h_df["score2"] > h2h_df["score1"])).sum()
        )
        draws = (h2h_df["score1"] == h2h_df["score2"]).sum()

        if total_games > 0:
            h2h_prob_overall = (wins_a + 0.5 * draws) / total_games
        else:
            h2h_prob_overall = None

        # Recent form: last up to 5 H2H matches
        h2h_recent = h2h_df.sort_values("date").tail(min(5, total_games))
        if not h2h_recent.empty:
            recent_games = len(h2h_recent)
            r_wins_a = (
                ((h2h_recent["player1"] == player_a) & (h2h_recent["score1"] > h2h_recent["score2"])).sum()
                + ((h2h_recent["player2"] == player_a) & (h2h_recent["score2"] > h2h_recent["score1"])).sum()
            )
            r_draws = (h2h_recent["score1"] == h2h_recent["score2"]).sum()
            h2h_prob_recent = (r_wins_a + 0.5 * r_draws) / recent_games
        else:
            h2h_prob_recent = None

        if h2h_prob_overall is not None:
            if h2h_prob_recent is not None and total_games >= 3:
                h2h_prob = 0.5 * h2h_prob_overall + 0.5 * h2h_prob_recent
            else:
                h2h_prob = h2h_prob_overall

        # -------- 3. xG-based edge in this rivalry --------
        if "xG1" in h2h_df.columns and "xG2" in h2h_df.columns:
            diffs = []
            for _, row in h2h_df.iterrows():
                if row["player1"] == player_a:
                    xa = row.get("xG1")
                    xb = row.get("xG2")
                else:
                    xa = row.get("xG2")
                    xb = row.get("xG1")

                if pd.notna(xa) and pd.notna(xb):
                    diffs.append(float(xa) - float(xb))

            if diffs:
                avg_diff = float(np.mean(diffs))
                # logistic mapping from xG-goal advantage → probability
                xg_prob = 1.0 / (1.0 + np.exp(-avg_diff / 0.75))

    # -------- 4. Blend components --------
    components = [elo_prob]
    weights = [0.5]  # 50% ELO baseline

    if h2h_prob is not None:
        components.append(h2h_prob)
        weights.append(0.3)  # 30% H2H

    if xg_prob is not None:
        components.append(xg_prob)
        weights.append(0.2)  # 20% xG

    w_sum = sum(weights)
    final_prob = sum(c * w for c, w in zip(components, weights)) / w_sum
    final_prob = max(0.05, min(0.95, final_prob))  # clamp

    return {
        "ra": ra,
        "rb": rb,
        "elo_prob": elo_prob,
        "h2h_prob": h2h_prob,
        "xg_prob": xg_prob,
        "final_prob": final_prob,
    }

def predict_match_1v1(df: pd.DataFrame, game: str, player_a: str, player_b: str):
    """
    Public API used by the app.
    Keeps the old signature: returns (ra, rb, prob_a).
    """
    comps = _prediction_components_1v1(df, game, player_a, player_b)
    return comps["ra"], comps["rb"], comps["final_prob"]


def compute_goals_vs_opponent(h2h_df: pd.DataFrame, player: str):
    games = 0
    goals_for = 0
    goals_against = 0

    for _, row in h2h_df.iterrows():
        if row["player1"] == player:
            goals_for += row["score1"]
            goals_against += row["score2"]
            games += 1
        elif row["player2"] == player:
            goals_for += row["score2"]
            goals_against += row["score1"]
            games += 1

    return games, goals_for, goals_against


def summarize_team_stats_vs_opponent(h2h_df: pd.DataFrame, player: str) -> pd.DataFrame:
    """
    For a given player in this head-to-head, return per-team stats:
    games, goals_for, goals_against, xG_for, xG_against, and averages.
    """
    rows = []

    for _, row in h2h_df.iterrows():
        if row["player1"] == player:
            team = row.get("team1")
            gf = row.get("score1")
            ga = row.get("score2")
            xg_for = row.get("xG1")
            xg_against = row.get("xG2")
        elif row["player2"] == player:
            team = row.get("team2")
            gf = row.get("score2")
            ga = row.get("score1")
            xg_for = row.get("xG2")
            xg_against = row.get("xG1")
        else:
            continue

        rows.append(
            {
                "team": team,
                "goals_for": gf,
                "goals_against": ga,
                "xg_for": xg_for,
                "xg_against": xg_against,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "team",
                "games",
                "goals_for",
                "goals_against",
                "xg_for",
                "xg_against",
                "avg_goals_for",
                "avg_goals_against",
                "avg_xg_for",
                "avg_xg_against",
            ]
        )

    df = pd.DataFrame(rows)

    grouped = df.groupby("team", dropna=True).agg(
        games=("team", "count"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        xg_for=("xg_for", "sum"),
        xg_against=("xg_against", "sum"),
    )

    grouped["avg_goals_for"] = grouped["goals_for"] / grouped["games"]
    grouped["avg_goals_against"] = grouped["goals_against"] / grouped["games"]
    grouped["avg_xg_for"] = grouped["xg_for"] / grouped["games"]
    grouped["avg_xg_against"] = grouped["xg_against"] / grouped["games"]

    grouped = grouped.sort_values(by="goals_for", ascending=False).reset_index()

    return grouped

def team_win_rate_vs_opponent(h2h_df: pd.DataFrame, player: str, team_name: str) -> tuple[float, float]:
    """
    For a given head-to-head dataframe (one matchup, one game version),
    compute how this player performs when using a specific team.

    Returns (games_with_team, win_rate_with_team).
    win_rate is from the player's perspective.
    """
    if h2h_df.empty or not team_name:
        return 0.0, 0.0

    # only matches where this player used that team
    mask = (
        ((h2h_df["player1"] == player) & (h2h_df["team1"] == team_name))
        | ((h2h_df["player2"] == player) & (h2h_df["team2"] == team_name))
    )
    df_team = h2h_df[mask].copy()
    games = len(df_team)
    if games == 0:
        return 0.0, 0.0

    # wins from this player's point of view
    wins = (
        ((df_team["player1"] == player) & (df_team["score1"] > df_team["score2"]))
        | ((df_team["player2"] == player) & (df_team["score2"] > df_team["score1"]))
    ).sum()

    win_rate = wins / games
    return float(games), float(win_rate)

# =========================
# UTILS FOR INPUT UI
# =========================
def player_input_block(label, existing_players, key_prefix):
    """
    Helper for the Record Match page:
    - Shows a dropdown of existing players.
    - Also lets you type a brand new player name.
    - Returns the chosen player name (or "" if nothing is selected).
    """
    # Dropdown of known players
    options = ["-- Select existing --"] + sorted(existing_players)
    selected = st.selectbox(
        f"{label} (existing)",
        options,
        key=f"{key_prefix}_select",
    )

    # Text box for a new player
    new_name = st.text_input(
        f"{label} (new, if not in list)",
        key=f"{key_prefix}_new",
    ).strip()

    # If user typed a new name, use that
    if new_name:
        return new_name

    # Otherwise, if they picked an existing one, use that
    if selected != "-- Select existing --":
        return selected

    # Nothing chosen
    return ""
# ---------- LINEUP NORMALIZATION HELPERS (for 2v2 / 2v1) ----------
def normalize_player_list(players_string: str):
    """
    Turn 'Navdeep, Harman' into a sorted tuple ('harman', 'navdeep').
    Works for 1, 2, or more players and is order-independent.
    """
    if not isinstance(players_string, str) or not players_string.strip():
        return tuple()

    parts = [p.strip().lower() for p in players_string.split(",")]
    parts = [p for p in parts if p]  # drop empty
    return tuple(sorted(set(parts)))  # dedupe + sort


def lineup_key(players_string: str) -> str:
    """
    Human-readable key version of normalize_player_list.
    E.g. 'Navdeep, Harman' -> 'Harman + Navdeep'
    """
    names = normalize_player_list(players_string)
    if not names:
        return ""
    return " + ".join(n.title() for n in names)


def same_lineup(a: str, b: str) -> bool:
    """
    True if the two player-lists represent the same lineup (order doesn’t matter).
    """
    return normalize_player_list(a) == normalize_player_list(b)

# =========================
# STREAMLIT UI
# =========================

st.sidebar.markdown("### ⚙️ Settings")
selected_game = st.sidebar.selectbox("Game version", GAME_OPTIONS)
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Record Match", "Head-to-Head (1v1)", "Head-to-Head (2v2)", "Head-to-Head (2v1)", "All Data"],
)

# Refresh button to clear cached data
refresh_clicked = st.sidebar.button("🔄 Refresh data from Google Sheets")


# =========================
# CACHED DATA LOADER (REDUCE QUOTA)
# =========================
@st.cache_data(ttl=600)
def load_all_data():
    """
    Load all match types from Google Sheets, cached to reduce quota usage.
    """
    df1 = load_matches_1v1()
    df2 = load_matches_2v2()
    df3 = load_matches_2v1()
    return df1, df2, df3

# If refresh is clicked, clear the cache BEFORE loading
if refresh_clicked:
    load_all_data.clear()

# Use the cached loader once per run
df_1v1, df_2v2, df_2v1 = load_all_data()

# Extra safety on frames – ensure expected columns exist
EXPECTED_1V1_COLS = [
    "date",
    "game",
    "player1",
    "team1",
    "score1",
    "xG1",
    "result1",
    "player2",
    "team2",
    "score2",
    "xG2",
    "result2",
]
EXPECTED_2V2_COLS = [
    "date",
    "game",
    "team1_name",
    "team1_players",
    "score1",
    "xG1",
    "result1",
    "team2_name",
    "team2_players",
    "score2",
    "xG2",
    "result2",
]
EXPECTED_2V1_COLS = [
    "date",
    "game",
    "team1_name",
    "team1_players",
    "score1",
    "xG1",
    "result1",
    "team2_name",
    "team2_players",
    "score2",
    "xG2",
    "result2",
]

for col in EXPECTED_1V1_COLS:
    if col not in df_1v1.columns:
        df_1v1[col] = None

for col in EXPECTED_2V2_COLS:
    if col not in df_2v2.columns:
        df_2v2[col] = None

for col in EXPECTED_2V1_COLS:
    if col not in df_2v1.columns:
        df_2v1[col] = None

df_1v1 = df_1v1[EXPECTED_1V1_COLS]
df_2v2 = df_2v2[EXPECTED_2V2_COLS]
df_2v1 = df_2v1[EXPECTED_2V1_COLS]

df_1v1_game = df_1v1[df_1v1["game"] == selected_game].copy()
df_2v2_game = df_2v2[df_2v2["game"] == selected_game].copy()
df_2v1_game = df_2v1[df_2v1["game"] == selected_game].copy()

players_all = sorted(
    set(df_1v1["player1"].dropna().unique()).union(
        set(df_1v1["player2"].dropna().unique())
    )
)
players_game = sorted(
    set(df_1v1_game["player1"].dropna().unique()).union(
        set(df_1v1_game["player2"].dropna().unique())
    )
)

# ---------- PAGE: DASHBOARD ----------
if page == "Dashboard":
    st.subheader(f"🏠 Season Summary – {selected_game}")

    # --- Top KPI tiles ---
    colA, colB, colC, colD = st.columns(4)

    total_matches = len(df_1v1_game) + len(df_2v2_game)
    total_goals = 0
    if not df_1v1_game.empty:
        total_goals += df_1v1_game["score1"].sum() + df_1v1_game["score2"].sum()
    if not df_2v2_game.empty:
        total_goals += df_2v2_game["score1"].sum() + df_2v2_game["score2"].sum()

    leaderboard_players = build_player_leaderboard_1v1(df_1v1, selected_game)
    leaderboard_teams = build_team_leaderboard_2v2(df_2v2, selected_game)

    if not leaderboard_players.empty:
        top_player = leaderboard_players.iloc[0]["player"]
        top_player_elo = leaderboard_players.iloc[0]["elo_rating"]
    else:
        top_player = "N/A"
        top_player_elo = 0

    if not leaderboard_players.empty:
        best_attacker_row = leaderboard_players.sort_values(
            by="avg_goals_for", ascending=False
        ).iloc[0]
        best_attacker = best_attacker_row["player"]
        best_attacker_gpg = best_attacker_row["avg_goals_for"]
    else:
        best_attacker = "N/A"
        best_attacker_gpg = 0

    colA.metric("Total matches", total_matches)
    colB.metric("Total goals", int(total_goals))
    colC.metric("Top ELO player", top_player, f"ELO {int(top_player_elo)}")
    colD.metric(
        "Most goals per game",
        best_attacker,
        f"{best_attacker_gpg:.2f} goals/game" if best_attacker != "N/A" else "",
    )

    st.markdown("---")

    # --- 1v1 leaderboard – full width with Rank ---
    st.markdown("### 👤 1v1 Player Leaderboard")

    if leaderboard_players.empty:
        st.info(f"No 1v1 matches yet for {selected_game}.")
    else:
        # Ensure it’s sorted in your preferred order
        leaderboard_players = leaderboard_players.sort_values(
            by=["elo_rating", "wins", "goal_diff"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        # Add Rank column (1st, 2nd, 3rd, …)
        leaderboard_players.insert(0, "Rank", leaderboard_players.index + 1)

        display_cols = [
            "Rank",
            "player",
            "games",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "goal_diff",
            "avg_goals_for",
            "avg_goals_against",
            "win_pct",
            "elo_rating",
        ]

        st.dataframe(
            leaderboard_players[display_cols].style.format(
                {
                    "avg_goals_for": "{:.2f}",
                    "avg_goals_against": "{:.2f}",
                    "win_pct": "{:.1%}",
                    "elo_rating": "{:.0f}",
                }
            ),
            use_container_width=True,
        )

        # Optional compact ELO bar chart under the table
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.bar(leaderboard_players["player"], leaderboard_players["elo_rating"])
        ax.set_xlabel("Player")
        ax.set_ylabel("ELO Rating")
        ax.set_title(f"ELO – {selected_game}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # --- 2v2 leaderboard – full width with Rank ---
    st.markdown("### 👥 2v2 Team Leaderboard")

    if leaderboard_teams.empty:
        st.info(f"No 2v2 matches yet for {selected_game}.")
    else:
        # Make sure 2v2 leaderboard includes an ELO column in build_team_leaderboard_2v2
        leaderboard_teams = leaderboard_teams.sort_values(
            by=["win_pct", "goal_diff"],
            ascending=[False, False],
        ).reset_index(drop=True)

        leaderboard_teams.insert(0, "Rank", leaderboard_teams.index + 1)

        display_cols_t = [
            "Rank",
            "team",
            "players",
            "games",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "goal_diff",
            "avg_goals_for",
            "avg_goals_against",
            "win_pct",
        ]
        # If you have team ELO in the DF, optionally add it:
        if "elo_rating" in leaderboard_teams.columns:
            display_cols_t.append("elo_rating")

        st.dataframe(
            leaderboard_teams[display_cols_t].style.format(
                {
                    "avg_goals_for": "{:.2f}",
                    "avg_goals_against": "{:.2f}",
                    "win_pct": "{:.1%}",
                }
            ),
            use_container_width=True,
        )

        fig2, ax2 = plt.subplots(figsize=(6, 2.5))
        ax2.bar(leaderboard_teams["team"], leaderboard_teams["goals_for"])
        ax2.set_xlabel("Team")
        ax2.set_ylabel("Goals For")
        ax2.set_title(f"2v2 Goals – {selected_game}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("---")

        st.markdown("### 🏆 1v1 Player Awards & Titles")

        if leaderboard_players.empty:
            st.info("No 1v1 matches yet to calculate player titles.")
        else:
            import numpy as np
            from collections import defaultdict

            awards_1v1 = {}

            # ---------- Core: Golden Boot ----------
            golden_boot = leaderboard_players.sort_values(
                "goals_for", ascending=False
            ).iloc[0]
            awards_1v1["Golden Boot (Top Scorer)"] = (
                f"**{golden_boot['player']}** – {int(golden_boot['goals_for'])} goals"
            )

            # Players with enough games
            eligible = leaderboard_players[leaderboard_players["games"] >= 5].copy()

            # Best Defense
            if not eligible.empty:
                best_def = eligible.sort_values("avg_goals_against").iloc[0]
                awards_1v1["Brick Wall (Best Defense)"] = (
                    f"**{best_def['player']}** – {best_def['avg_goals_against']:.2f} avg conceded"
                )

                best_wr = eligible.sort_values("win_pct", ascending=False).iloc[0]
                awards_1v1["Consistent Winner (Highest Win Rate)"] = (
                    f"**{best_wr['player']}** – {best_wr['win_pct']:.1%} wins"
                )

            # Attack Threat
            top_attack = leaderboard_players.sort_values(
                "avg_goals_for", ascending=False
            ).iloc[0]
            awards_1v1["Most Feared Rival (Attack Threat)"] = (
                f"**{top_attack['player']}** – {top_attack['avg_goals_for']:.2f} avg goals per game"
            )

            # Clean Sheet King
            clean_sheets = {}
            for _, row in df_1v1_game.iterrows():
                if row["score2"] == 0:
                    clean_sheets[row["player1"]] = clean_sheets.get(row["player1"], 0) + 1
                if row["score1"] == 0:
                    clean_sheets[row["player2"]] = clean_sheets.get(row["player2"], 0) + 1

            if clean_sheets:
                cs_player = max(clean_sheets, key=clean_sheets.get)
                awards_1v1["Clean Sheet King"] = (
                    f"**{cs_player}** – {clean_sheets[cs_player]} clean sheets"
                )

            # ---------- Strongest All-Round Player ----------
            eligible_all = leaderboard_players[leaderboard_players["games"] >= 5].copy()
            if not eligible_all.empty:
                eligible_all["gd_per_game"] = eligible_all["goal_diff"] / eligible_all["games"]
                eligible_all["composite"] = (
                    eligible_all["win_pct"].rank(pct=True) * 0.5
                    + eligible_all["gd_per_game"].rank(pct=True) * 0.5
                )
                best_overall = eligible_all.sort_values("composite", ascending=False).iloc[0]
                awards_1v1["All-Round MVP (Strongest Overall)"] = (
                    f"**{best_overall['player']}** – "
                    f"{best_overall['win_pct']:.1%} wins, "
                    f"GD/game {best_overall['gd_per_game']:.2f}"
                )

                # Recent Form
                recent_results = defaultdict(list)
                df_sorted = df_1v1_game.sort_values("date")
                for _, row in df_sorted.iterrows():
                    p1, p2 = row["player1"], row["player2"]
                    s1, s2 = row["score1"], row["score2"]
                    if s1 > s2:
                        r1, r2 = 1.0, 0.0
                    elif s1 < s2:
                        r1, r2 = 0.0, 1.0
                    else:
                        r1 = r2 = 0.5
                    recent_results[p1].append(r1)
                    recent_results[p2].append(r2)

                streak_scores = {
                    p: sum(r == 1.0 for r in seq[-10:]) / len(seq[-10:])
                    for p, seq in recent_results.items()
                    if len(seq) >= 3
                }
                if streak_scores:
                    hot_player = max(streak_scores, key=streak_scores.get)
                    awards_1v1["Most Improved Form (Last 10 Games)"] = (
                        f"**{hot_player}** – "
                        f"{streak_scores[hot_player]:.1%} wins recently"
                    )

            # ---------- Luck (G − xG) ----------
            if "xG1" in df_1v1_game.columns:
                xg_for_map = {}
                for _, row in df_1v1_game.iterrows():
                    if not np.isnan(row.get("xG1", np.nan)):
                        xg_for_map[row["player1"]] = xg_for_map.get(row["player1"], 0) + row["xG1"]
                    if not np.isnan(row.get("xG2", np.nan)):
                        xg_for_map[row["player2"]] = xg_for_map.get(row["player2"], 0) + row["xG2"]

                leaderboard_players["xg_for"] = leaderboard_players["player"].map(
                    lambda p: xg_for_map.get(p, np.nan)
                )
                leaderboard_players["luck"] = (
                    leaderboard_players["goals_for"] - leaderboard_players["xg_for"]
                )

                eligible_xg = leaderboard_players[
                    leaderboard_players["xg_for"].notna()
                    & (leaderboard_players["games"] >= 5)
                ]
                if not eligible_xg.empty:
                    luckiest = eligible_xg.sort_values("luck", ascending=False).iloc[0]
                    unluckiest = eligible_xg.sort_values("luck", ascending=True).iloc[0]

                    awards_1v1["Luckiest Finisher"] = (
                        f"**{luckiest['player']}** – +{luckiest['luck']:.1f} goals vs xG"
                    )
                    awards_1v1["Unluckiest Finisher"] = (
                        f"**{unluckiest['player']}** – {unluckiest['luck']:.1f} goals vs xG"
                    )

            # ---------- Show all awards ----------
            for title, desc in awards_1v1.items():
                st.markdown(f"🏅 **{title}:** {desc}")



       st.markdown("### 🏆 2v2 Team Awards & Titles")

        if leaderboard_teams.empty:
            st.info("No 2v2 matches yet to calculate team titles.")
        else:
            import numpy as np
            from collections import defaultdict

            awards_2v2 = {}

        # ---------- Core titles ----------
        eligible_teams = leaderboard_teams[leaderboard_teams["games"] >= 3].copy()

        # Golden Duo – highest win %
        if not eligible_teams.empty:
            best_team = eligible_teams.sort_values("win_pct", ascending=False).iloc[0]
            awards_2v2["Golden Duo (Best 2v2 Team)"] = (
                f"**{best_team['team']}** – "
                f"{best_team['win_pct']:.1%} win rate "
                f"({int(best_team['games'])} games)"
            )

        # Attacking Duo – highest avg goals_for per game
        top_attack_team = leaderboard_teams.sort_values(
            "avg_goals_for", ascending=False
        ).iloc[0]
        awards_2v2["Attacking Duo (Most Goals per Game)"] = (
            f"**{top_attack_team['team']}** – "
            f"{top_attack_team['avg_goals_for']:.2f} avg goals scored"
        )

        # Fortress Duo – lowest avg goals_against per game (min 3 games)
        if not eligible_teams.empty:
            best_def_team = eligible_teams.sort_values(
                "avg_goals_against"
            ).iloc[0]
            awards_2v2["Fortress Duo (Best Defense)"] = (
                f"**{best_def_team['team']}** – "
                f"{best_def_team['avg_goals_against']:.2f} avg conceded"
            )

        # Clean Sheet Duo – most games with 0 conceded
        clean_sheets_team = {}
        for _, row in df_2v2_game.iterrows():
            if row["score2"] == 0:
                clean_sheets_team[row["team1_name"]] = clean_sheets_team.get(
                    row["team1_name"], 0
                ) + 1
            if row["score1"] == 0:
                clean_sheets_team[row["team2_name"]] = clean_sheets_team.get(
                    row["team2_name"], 0
                ) + 1

        if clean_sheets_team:
            cs_team = max(clean_sheets_team, key=clean_sheets_team.get)
            awards_2v2["Clean Sheet Duo"] = (
                f"**{cs_team}** – {clean_sheets_team[cs_team]} clean sheets"
            )

        # ---------- NEW: Strongest Overall Team ----------
        if not eligible_teams.empty:
            eligible_teams["gd_per_game"] = (
                eligible_teams["goal_diff"] / eligible_teams["games"]
            )
            eligible_teams["composite"] = (
                eligible_teams["win_pct"].rank(pct=True) * 0.5
                + eligible_teams["gd_per_game"].rank(pct=True) * 0.5
            )
            best_overall_team = eligible_teams.sort_values(
                "composite", ascending=False
            ).iloc[0]
            awards_2v2["Complete Squad (Strongest Overall)"] = (
                f"**{best_overall_team['team']}** – "
                f"{best_overall_team['win_pct']:.1%} wins, "
                f"GD/game {best_overall_team['gd_per_game']:.2f}"
            )

            # Recent form – last 10 games for each team
            recent_results_2v2 = defaultdict(list)
            df_sorted_2v2 = df_2v2_game.sort_values("date")

            for _, row in df_sorted_2v2.iterrows():
                a, b = row["team1_name"], row["team2_name"]
                s1, s2 = row["score1"], row["score2"]
                if pd.isna(a) or pd.isna(b) or pd.isna(s1) or pd.isna(s2):
                    continue
                if s1 > s2:
                    r1, r2 = 1.0, 0.0
                elif s1 < s2:
                    r1, r2 = 0.0, 1.0
                else:
                    r1 = r2 = 0.5
                recent_results_2v2[a].append(r1)
                recent_results_2v2[b].append(r2)

            streak_scores_2v2 = {}
            for team, seq in recent_results_2v2.items():
                if len(seq) >= 3:
                    last = seq[-10:]
                    streak_scores_2v2[team] = sum(
                        1 for r in last if r == 1.0
                    ) / len(last)

            if streak_scores_2v2:
                hot_team = max(streak_scores_2v2, key=streak_scores_2v2.get)
                games_count = min(10, len(recent_results_2v2[hot_team]))
                awards_2v2["Hottest Duo (Last 10 Games)"] = (
                    f"**{hot_team}** – "
                    f"{streak_scores_2v2[hot_team]:.1%} wins "
                    f"over their last {games_count} games"
                )

        # ---------- NEW: xG luck for teams (G − xG) ----------
        if "xG1" in df_2v2_game.columns and "xG2" in df_2v2_game.columns:
            xg_for_team = {}

            for _, row in df_2v2_game.iterrows():
                t_a, t_b = row["team1_name"], row["team2_name"]
                xg1 = row.get("xG1", np.nan)
                xg2 = row.get("xG2", np.nan)

                if pd.notna(t_a) and not np.isnan(xg1):
                    xg_for_team[t_a] = xg_for_team.get(t_a, 0.0) + xg1
                if pd.notna(t_b) and not np.isnan(xg2):
                    xg_for_team[t_b] = xg_for_team.get(t_b, 0.0) + xg2

            leaderboard_teams["xg_for"] = leaderboard_teams["team"].map(
                lambda t: xg_for_team.get(t, np.nan)
            )
            leaderboard_teams["luck_g_minus_xg"] = (
                leaderboard_teams["goals_for"] - leaderboard_teams["xg_for"]
            )

            eligible_xg_team = leaderboard_teams[
                leaderboard_teams["xg_for"].notna()
                & (leaderboard_teams["games"] >= 3)
            ].copy()

            if not eligible_xg_team.empty:
                luckiest_team = eligible_xg_team.sort_values(
                    "luck_g_minus_xg", ascending=False
                ).iloc[0]
                unluckiest_team = eligible_xg_team.sort_values(
                    "luck_g_minus_xg", ascending=True
                ).iloc[0]

                awards_2v2["Luckiest Duo (G > xG)"] = (
                    f"**{luckiest_team['team']}** – "
                    f"+{luckiest_team['luck_g_minus_xg']:.1f} goals vs xG"
                )
                awards_2v2["Unluckiest Duo (xG > G)"] = (
                    f"**{unluckiest_team['team']}** – "
                    f"{unluckiest_team['luck_g_minus_xg']:.1f} goals vs xG"
                )

        # ---------- Display all 2v2 titles ----------
        for title, text in awards_2v2.items():
            st.markdown(f"🏅 **{title}:** {text}")


    st.markdown("### 🔮 Quick 1v1 Prediction (for friendly wagers)")

    if len(players_game) < 2 or df_1v1_game.empty:
        st.info(
            f"Add a few 1v1 matches for {selected_game} to unlock meaningful predictions."
        )
    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            pred_p1 = st.selectbox("Player A", players_game, key="quick_pred_p1")
        with c2:
            pred_p2 = st.selectbox(
                "Player B",
                [p for p in players_game if p != pred_p1],
                key="quick_pred_p2",
            )
        with c3:
            st.write("")

        if pred_p1 and pred_p2:
            ra, rb, prob_a = predict_match_1v1(
                df_1v1, selected_game, pred_p1, pred_p2
            )
            st.write(
                f"Based on **{selected_game}** 1v1 history:\n\n"
                f"- {pred_p1} ELO: **{round(ra)}**\n"
                f"- {pred_p2} ELO: **{round(rb)}**\n"
            )
            st.write(
                f"Estimated win chance next game:\n"
                f"- **{pred_p1}: {prob_a:.1%}**\n"
                f"- **{pred_p2}: {(1 - prob_a):.1%}**"
            )

            if prob_a > 0.6:
                fav = pred_p1
                underdog = pred_p2
            elif prob_a < 0.4:
                fav = pred_p2
                underdog = pred_p1
            else:
                fav = None

            if fav:
                st.success(
                    f"Looks like **{fav}** is the favourite here. "
                    f"Perfect setup for {underdog} to pull an upset 👀"
                )
            else:
                st.info("Pretty even matchup. Flip a coin or take your chances 😅")


# ---------- PAGE: RECORD MATCH ----------
elif page == "Record Match":
    st.subheader(f"📝 Record a Match – {selected_game}")

    # Match type selector
    match_type = st.radio(
        "Match type",
        ["1v1", "2v2", "2v1"],       # NEW 2v1 option
        horizontal=True,
        key="match_type_radio",
    )

    # Match date + game selector
    date = st.date_input("Match date", value=datetime.date.today())
    game_for_entry = st.selectbox(
        "Game",
        GAME_OPTIONS,
        index=GAME_OPTIONS.index(selected_game),
    )

    st.markdown("#### Enter details")

        # -------------------------
    #          1v1
    # -------------------------
    if match_type == "1v1":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Player 1**")
            p1 = player_input_block("Player 1", players_all, key_prefix="p1_input")
            team1 = st.text_input("Team 1", key="team1_input").strip()
            score1 = st.number_input(
                "Goals scored by Player 1",
                min_value=0,
                step=1,
                key="score1_1v1",
            )
            xG1_1v1 = st.number_input(
                "Expected goals (xG) for Player 1",
                min_value=0.0,
                step=0.1,
                key="xg1_1v1",
            )

        with col2:
            st.markdown("**Player 2**")
            p2 = player_input_block("Player 2", players_all, key_prefix="p2_input")
            team2 = st.text_input("Team 2", key="team2_input").strip()
            score2 = st.number_input(
                "Goals scored by Player 2",
                min_value=0,
                step=1,
                key="score2_1v1",
            )
            xG2_1v1 = st.number_input(
                "Expected goals (xG) for Player 2",
                min_value=0.0,
                step=0.1,
                key="xg2_1v1",
            )

        if st.button("Save 1v1 match", use_container_width=True):
            if not p1 or not p2:
                st.error("Please fill in both player names (either existing or new).")
            elif p1 == p2:
                st.error("Players must be different.")
            elif not team1 or not team2:
                st.error("Please enter both Team 1 and Team 2.")
            else:
                append_match_1v1(
                    date,
                    game_for_entry,
                    p1,
                    team1,
                    score1,
                    xG1_1v1,
                    p2,
                    team2,
                    score2,
                    xG2_1v1,
                )
                st.success(f"Saved 1v1 match for {game_for_entry}! 🎉")
                load_all_data.clear()

        # -------------------------
    #          2v2
    # -------------------------
    elif match_type == "2v2":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Team 1**")
            team1_name = st.text_input("Team 1 name", key="team1_name").strip()
            team1_players = st.text_input(
                "Team 1 players (e.g. Sue & Alex)", key="team1_players"
            ).strip()
            score1_2v2 = st.number_input(
                "Goals scored by Team 1",
                min_value=0,
                step=1,
                key="score1_2v2",
            )
            xG1_2v2 = st.number_input(
                "Expected goals (xG) for Team 1",
                min_value=0.0,
                step=0.1,
                key="xg1_2v2",
            )

        with col2:
            st.markdown("**Team 2**")
            team2_name = st.text_input("Team 2 name", key="team2_name").strip()
            team2_players = st.text_input(
                "Team 2 players (e.g. Jordan & Max)", key="team2_players"
            ).strip()
            score2_2v2 = st.number_input(
                "Goals scored by Team 2",
                min_value=0,
                step=1,
                key="score2_2v2",
            )
            xG2_2v2 = st.number_input(
                "Expected goals (xG) for Team 2",
                min_value=0.0,
                step=0.1,
                key="xg2_2v2",
            )

        if st.button("Save 2v2 match", use_container_width=True):
            if not team1_name or not team2_name:
                st.error("Please fill in both team names.")
            elif team1_name == team2_name:
                st.error("Teams must be different.")
            else:
                append_match_2v2(
                    date,
                    game_for_entry,
                    team1_name,
                    team1_players,
                    score1_2v2,
                    xG1_2v2,
                    team2_name,
                    team2_players,
                    score2_2v2,
                    xG2_2v2,
                )
                st.success(f"Saved 2v2 match for {game_for_entry}! 🎉")
                load_all_data.clear()

    # -------------------------
    #          2v1   (NEW)
    # -------------------------
    elif match_type == "2v1":
        col1, col2 = st.columns(2)

        # Two-player side
        with col1:
            st.markdown("**Team 1 (2-player side)**")
            team1_name = st.text_input("Team 1 name", key="2v1_team1_name").strip()
            team1_players = st.text_input(
                "Team 1 players (e.g. Sue & Alex)",
                key="2v1_team1_players",
            ).strip()
            score1_2v1 = st.number_input(
                "Goals scored by Team 1",
                min_value=0,
                step=1,
                key="2v1_score1",
            )
            xG1_2v1 = st.number_input(
                "Expected goals (xG) for Team 1",
                min_value=0.0,
                step=0.1,
                key="xg1_2v1",
            )

        # Solo side
        with col2:
            st.markdown("**Team 2 (solo player)**")
            team2_name = st.text_input("Solo Player name", key="2v1_team2_name").strip()
            team2_players = st.text_input(
                "Solo player name",
                key="2v1_team2_players",
            ).strip()
            score2_2v1 = st.number_input(
                "Goals scored by Solo Player",
                min_value=0,
                step=1,
                key="2v1_score2",
            )
            xG2_2v1 = st.number_input(
                "Expected goals (xG) for Team 2",
                min_value=0.0,
                step=0.1,
                key="xg2_2v1",
            )

        if st.button("Save 2v1 match", use_container_width=True):
            if not team1_name or not team2_name:
                st.error("Please fill in both side names.")
            elif team1_name == team2_name:
                st.error("Team names must be different.")
            else:
                append_match_2v1(
                    date,
                    game_for_entry,
                    team1_name,
                    team1_players,
                    score1_2v1,
                    xG1_2v1,
                    team2_name,
                    team2_players,
                    score2_2v1,
                    xG2_2v1
                )
                st.success(f"Saved 2v1 match for {game_for_entry}! 🎉")
                load_all_data.clear()

# -----# ---------- PAGE: HEAD-TO-HEAD (1v1) ----------
elif page == "Head-to-Head (1v1)":
    st.subheader(f"🔍 1v1 Head-to-Head – {selected_game}")

    if len(players_game) < 2 or df_1v1_game.empty:
        st.info(
            f"Need at least 2 players and some 1v1 matches recorded for {selected_game}."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.selectbox("Player A", players_game, key="h2h_p1")
        with col2:
            p2 = st.selectbox(
                "Player B", [p for p in players_game if p != p1], key="h2h_p2"
            )

        if p1 and p2:
            h2h_df = head_to_head_1v1(df_1v1, selected_game, p1, p2)

            st.markdown(f"### {p1} vs {p2} – {selected_game}")

            if h2h_df.empty:
                st.info("No direct 1v1 matches between these two for this game yet.")
            else:
                # Sort chronologically for charts / last 5
                h2h_df = h2h_df.sort_values("date").reset_index(drop=True)

                # ----- CORE COUNTS & PER-MATCH ARRAYS -----
                wins_p1 = 0
                wins_p2 = 0
                draws = 0

                match_idx = []
                p1_goals = []
                p2_goals = []
                p1_xg_list = []
                p2_xg_list = []
                score_diffs = []
                goal_margins_abs = []

                for i, row in h2h_df.iterrows():
                    s1, s2 = row["score1"], row["score2"]
                    pl1, pl2 = row["player1"], row["player2"]

                    # win/draw/loss counts
                    if s1 > s2:
                        if pl1 == p1:
                            wins_p1 += 1
                        elif pl1 == p2:
                            wins_p2 += 1
                    elif s2 > s1:
                        if pl2 == p1:
                            wins_p1 += 1
                        elif pl2 == p2:
                            wins_p2 += 1
                    else:
                        draws += 1

                    # From p1 perspective for charts
                    if pl1 == p1:
                        g_p1 = s1
                        g_p2 = s2
                        xg_p1 = row.get("xG1", np.nan)
                        xg_p2 = row.get("xG2", np.nan)
                    else:
                        g_p1 = s2
                        g_p2 = s1
                        xg_p1 = row.get("xG2", np.nan)
                        xg_p2 = row.get("xG1", np.nan)

                    match_idx.append(i + 1)
                    p1_goals.append(g_p1)
                    p2_goals.append(g_p2)
                    p1_xg_list.append(xg_p1)
                    p2_xg_list.append(xg_p2)
                    score_diffs.append(g_p1 - g_p2)
                    if g_p1 != g_p2:
                        goal_margins_abs.append(abs(g_p1 - g_p2))

                total_games = len(h2h_df)

                # Goals aggregates
                gf_p1 = sum(p1_goals)
                ga_p1 = sum(p2_goals)
                gf_p2 = sum(p2_goals)
                ga_p2 = sum(p1_goals)

                avg_gf_p1 = gf_p1 / total_games if total_games else 0.0
                avg_ga_p1 = ga_p1 / total_games if total_games else 0.0
                avg_gf_p2 = gf_p2 / total_games if total_games else 0.0
                avg_ga_p2 = ga_p2 / total_games if total_games else 0.0

                # xG aggregates for text / radar
                p1_xg = np.array(p1_xg_list, dtype=float)
                p2_xg = np.array(p2_xg_list, dtype=float)

                avg_xg_for_p1 = float(np.nanmean(p1_xg)) if np.any(~np.isnan(p1_xg)) else 0.0
                avg_xg_for_p2 = float(np.nanmean(p2_xg)) if np.any(~np.isnan(p2_xg)) else 0.0
                avg_xg_against_p1 = float(np.nanmean(p2_xg)) if np.any(~np.isnan(p2_xg)) else 0.0
                avg_xg_against_p2 = float(np.nanmean(p1_xg)) if np.any(~np.isnan(p1_xg)) else 0.0

                # ----- TOP METRICS -----
                colL, colM, colR = st.columns(3)
                colL.metric(f"{p1} wins", wins_p1)
                colM.metric("Draws", draws)
                colR.metric(f"{p2} wins", wins_p2)

                # ----- RIVALRY SUMMARY + LAST 5 -----
                col_sum_left, col_sum_right = st.columns(2)

                with col_sum_left:
                    st.markdown("### Rivalry summary")
                    st.write(
                        f"- Total games: **{total_games}**\n"
                        f"- {p1}: **{wins_p1}W {draws}D {wins_p2}L**, "
                        f"{gf_p1} goals / {ga_p1} conceded "
                        f"(avg {avg_gf_p1:.2f} for, {avg_ga_p1:.2f} against, "
                        f"xG {avg_xg_for_p1:.2f} for, {avg_xg_against_p1:.2f} against)\n"
                        f"- {p2}: **{wins_p2}W {draws}D {wins_p1}L**, "
                        f"{gf_p2} goals / {ga_p2} conceded "
                        f"(avg {avg_gf_p2:.2f} for, {avg_ga_p2:.2f} against, "
                        f"xG {avg_xg_for_p2:.2f} for, {avg_xg_against_p2:.2f} against)"
                    )

                with col_sum_right:
                    st.markdown("### Last 5 games")
                    last5 = h2h_df.tail(5)
                    form_p1 = []
                    form_p2 = []

                    for _, row in last5.iterrows():
                        s1, s2 = row["score1"], row["score2"]
                        pl1, pl2 = row["player1"], row["player2"]

                        if s1 > s2:
                            # winner is player1
                            if pl1 == p1:
                                form_p1.append("W")
                                form_p2.append("L")
                            else:
                                form_p1.append("L")
                                form_p2.append("W")
                        elif s2 > s1:
                            # winner is player2
                            if pl2 == p1:
                                form_p1.append("W")
                                form_p2.append("L")
                            else:
                                form_p1.append("L")
                                form_p2.append("W")
                        else:
                            form_p1.append("D")
                            form_p2.append("D")

                    st.markdown(
                        f"**Last 5 form**  \n"
                        f"{p1}: `{'-'.join(form_p1)}`  \n"
                        f"{p2}: `{'-'.join(form_p2)}`"
                    )

                # ----- TEAMS USED (with xG columns) -----
                st.markdown("### Teams used in this matchup")

                team_stats_p1 = summarize_team_stats_vs_opponent(h2h_df, p1)
                team_stats_p2 = summarize_team_stats_vs_opponent(h2h_df, p2)

                cols_team_display = [
                    "team",
                    "games",
                    "goals_for",
                    "goals_against",
                    "xg_for",
                    "xg_against",
                    "avg_goals_for",
                    "avg_goals_against",
                    "avg_xg_for",
                    "avg_xg_against",
                ]
                fmt_team = {
                    "goals_for": "{:.0f}",
                    "goals_against": "{:.0f}",
                    "xg_for": "{:.2f}",
                    "xg_against": "{:.2f}",
                    "avg_goals_for": "{:.2f}",
                    "avg_goals_against": "{:.2f}",
                    "avg_xg_for": "{:.2f}",
                    "avg_xg_against": "{:.2f}",
                }

                colT1, colT2 = st.columns(2)
                with colT1:
                    st.markdown(f"**{p1} teams vs {p2}**")
                    if team_stats_p1.empty:
                        st.write("No team data recorded.")
                    else:
                        st.dataframe(
                            team_stats_p1[cols_team_display].style.format(fmt_team),
                            use_container_width=True,
                        )

                with colT2:
                    st.markdown(f"**{p2} teams vs {p1}**")
                    if team_stats_p2.empty:
                        st.write("No team data recorded.")
                    else:
                        st.dataframe(
                            team_stats_p2[cols_team_display].style.format(fmt_team),
                            use_container_width=True,
                        )

                # ----- MATCH HISTORY -----
                st.markdown("### Match history")
                st.dataframe(
                    h2h_df[
                        [
                            "date",
                            "player1",
                            "team1",
                            "xG1",
                            "score1",
                            "score2",
                            "xG2",
                            "team2",
                            "player2",
                        ]
                    ].sort_values(by="date", ascending=False),
                    use_container_width=True,
                )

                                # ----- CHARTS IN EXPANDER -----
                with st.expander("📊 Show charts"):
                    # 1) Wins / draws bar
                    c1, c2 = st.columns(2)
                    with c1:
                        fig, ax = plt.subplots(figsize=(4.0, 2.3))
                        ax.bar([p1, p2, "Draws"], [wins_p1, wins_p2, draws])
                        ax.set_ylabel("Games")
                        ax.set_title("Wins / draws")
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        st.markdown(
                            "<p style='font-size:0.7rem; text-align:right; opacity:0.7;'>"
                            "Counts of wins, losses, and draws in this head-to-head."
                            "</p>",
                            unsafe_allow_html=True,
                        )

                    # 2) Goal margin distribution
                    with c2:
                        fig2, ax2 = plt.subplots(figsize=(4.0, 2.3))
                        # bucket 1,2,3+ (non-draws)
                        ones = sum(1 for m in goal_margins_abs if m == 1)
                        twos = sum(1 for m in goal_margins_abs if m == 2)
                        big = sum(1 for m in goal_margins_abs if m >= 3)
                        labels = ["1 goal", "2 goals", "3+ goals"]
                        counts = [ones, twos, big]
                        ax2.bar(labels, counts)
                        ax2.set_ylabel("Games")
                        ax2.set_title("Scoreline spread")
                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)
                        st.markdown(
                            "<p style='font-size:0.7rem; text-align:right; opacity:0.7;'>"
                            "How often games are decided by 1, 2, or 3+ goals."
                            "</p>",
                            unsafe_allow_html=True,
                        )

                    # 3) Goals & xG per match over time
                    st.markdown("#### Goals and xG per match over time")
                    if match_idx:
                        fig3, ax3 = plt.subplots(figsize=(5.0, 2.4))
                        ax3.plot(match_idx, p1_goals, marker="o", label=f"{p1} goals")
                        ax3.plot(match_idx, p2_goals, marker="o", label=f"{p2} goals")

                        if np.any(~np.isnan(p1_xg)):
                            ax3.plot(
                                match_idx,
                                p1_xg,
                                marker="x",
                                linestyle="--",
                                label=f"{p1} xG",
                            )
                        if np.any(~np.isnan(p2_xg)):
                            ax3.plot(
                                match_idx,
                                p2_xg,
                                marker="x",
                                linestyle="--",
                                label=f"{p2} xG",
                            )

                        ax3.set_xlabel("Match # (chronological)")
                        ax3.set_ylabel("Goals / xG")
                        ax3.legend(fontsize=8)
                        st.pyplot(fig3, use_container_width=True)
                        plt.close(fig3)
                        st.markdown(
                            "<p style='font-size:0.7rem; text-align:right; opacity:0.7;'>"
                            "Match-by-match goals and expected goals (xG) for each player."
                            "</p>",
                            unsafe_allow_html=True,
                        )

                    # 4) Score difference trend
                    st.markdown(f"#### Score difference trend (positive = {p1} ahead)")
                    if match_idx:
                        fig4, ax4 = plt.subplots(figsize=(5.0, 2.4))
                        ax4.axhline(0, color="gray", linewidth=1)
                        ax4.plot(match_idx, score_diffs, marker="o")
                        ax4.set_xlabel("Match # (chronological)")
                        ax4.set_ylabel(f"{p1} − {p2}")
                        st.pyplot(fig4, use_container_width=True)
                        plt.close(fig4)
                        st.markdown(
                            "<p style='font-size:0.7rem; text-align:right; opacity:0.7;'>"
                            "Each point is one game: above 0 = win for Player A, below 0 = win for Player B."
                            "</p>",
                            unsafe_allow_html=True,
                        )

                    # 5) Radar chart – rivalry profile
                    st.markdown("#### Rivalry radar (relative per-game profile)")
                    if match_idx:
                        n_games = len(match_idx)
                        win_rate_p1 = wins_p1 / n_games if n_games else 0.0
                        win_rate_p2 = wins_p2 / n_games if n_games else 0.0

                        luck_p1 = (gf_p1 - np.nansum(p1_xg)) if np.any(~np.isnan(p1_xg)) else 0.0
                        luck_p2 = (gf_p2 - np.nansum(p2_xg)) if np.any(~np.isnan(p2_xg)) else 0.0

                        metrics = ["GF", "xG For", "GA", "xG Against", "Win %", "Luck (G−xG)"]
                        vals_p1 = [avg_gf_p1, avg_xg_for_p1, avg_ga_p1, avg_xg_against_p1, win_rate_p1, luck_p1]
                        vals_p2 = [avg_gf_p2, avg_xg_for_p2, avg_ga_p2, avg_xg_against_p2, win_rate_p2, luck_p2]

                        max_vals = [max(abs(v1), abs(v2), 1e-6) for v1, v2 in zip(vals_p1, vals_p2)]
                        norm_p1 = [v / m for v, m in zip(vals_p1, max_vals)]
                        norm_p2 = [v / m for v, m in zip(vals_p2, max_vals)]

                        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
                        angles = np.concatenate((angles, [angles[0]]))
                        norm_p1_loop = norm_p1 + [norm_p1[0]]
                        norm_p2_loop = norm_p2 + [norm_p2[0]]

                        fig_rad, ax_rad = plt.subplots(subplot_kw={"polar": True}, figsize=(5.0, 3.0))
                        ax_rad.plot(angles, norm_p1_loop, label=p1)
                        ax_rad.fill(angles, norm_p1_loop, alpha=0.1)
                        ax_rad.plot(angles, norm_p2_loop, label=p2)
                        ax_rad.fill(angles, norm_p2_loop, alpha=0.1)

                        ax_rad.set_xticks(angles[:-1])
                        ax_rad.set_xticklabels(metrics, fontsize=7)
                        ax_rad.set_yticklabels([])
                        ax_rad.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=8)
                        st.pyplot(fig_rad, use_container_width=True)
                        plt.close(fig_rad)
                        st.markdown(
                            "<p style='font-size:0.7rem; text-align:right; opacity:0.7;'>"
                            "Radar shows each player’s relative profile: scoring, xG, defence, win %, and luck."
                            "</p>",
                            unsafe_allow_html=True,
                        )

                            # ----- WIN PREDICTION (ELO) – OUTSIDE EXPANDER -----
            # ----- WIN PREDICTION (hybrid model) – OUTSIDE EXPANDER -----
            st.markdown("---")
            st.markdown("### Win Prediction (for next 1v1)")
            
            # Use the internal helper so we can see all components
            comps = _prediction_components_1v1(df_1v1, selected_game, p1, p2)
            ra = comps["ra"]
            rb = comps["rb"]
            prob_a = comps["final_prob"]
            elo_component = comps["elo_prob"]
            h2h_component = comps["h2h_prob"]
            xg_component = comps["xg_prob"]
            
            st.write("ELO rating (for this game only):")
            st.write(f"- {p1}: **{round(ra)}**")
            st.write(f"- {p2}: **{round(rb)}**")
            
            st.write("Estimated win chance next match (hybrid model):")
            st.write(f"- **{p1}: {prob_a:.1%}**")
            st.write(f"- **{p2}: {(1 - prob_a):.1%}**")
            
            # Favourite / underdog summary (same logic as before)
            if prob_a > 0.6:
                st.success(f"Favouring **{p1}** right now. Time for {p2} to prove the stats wrong.")
            elif prob_a < 0.4:
                st.success(f"Favouring **{p2}** right now. {p1}, you’re the underdog here.")
            else:
                st.info("This one’s tight. Either player could take it.")
            
            # ---- Explain the blend ----
            st.markdown("**Model blend:** 50% ELO, 30% head-to-head (incl. last 5 games), 20% xG edge.")
            
            lines = []
            lines.append(f"- ELO-only win chance for {p1}: **{elo_component:.1%}**")
            
            if h2h_component is not None:
                lines.append(f"- Head-to-head / recent-form chance: **{h2h_component:.1%}**")
            else:
                lines.append("- Head-to-head / recent-form chance: _n/a (not enough games yet)_")
            
            if xg_component is not None:
                lines.append(f"- xG-based chance (expected-goals edge): **{xg_component:.1%}**")
            else:
                lines.append("- xG-based chance: _n/a (no xG data yet)_")
            
            st.markdown("\n".join(lines))


            # ---------- Optional team effect overlay ----------
            st.markdown("### Optional: team-based adjustment")

            total_games = wins_p1 + wins_p2 + draws
            base_wr_p1 = wins_p1 / total_games if total_games else 0.0
            base_wr_p2 = wins_p2 / total_games if total_games else 0.0

            col_team1, col_team2 = st.columns(2)
            with col_team1:
                team_future_p1 = st.text_input(
                    f"{p1} planned team (optional)", key="pred_team_p1"
                ).strip()
            with col_team2:
                team_future_p2 = st.text_input(
                    f"{p2} planned team (optional)", key="pred_team_p2"
                ).strip()

            # Helper to show effect text
            def _describe_team_effect(player_name, base_wr, base_prob, team_name):
                games_t, wr_t = team_win_rate_vs_opponent(h2h_df, player_name, team_name)
                if games_t == 0:
                    st.write(f"No past games where **{player_name}** used **{team_name}** in this matchup.")
                    return

                delta_wr = wr_t - base_wr  # positive if this team historically better
                # convert win-rate delta into a small probability nudge
                # scale down so a huge +40% win-rate doesn't explode the prediction
                delta_prob = 0.25 * delta_wr  # 0.25 is a heuristic scale
                adj_prob = max(0.05, min(0.95, base_prob + delta_prob))

                sign = "+" if delta_prob >= 0 else "−"
                st.write(
                    f"Using **{team_name}**: {player_name} has "
                    f"a **{wr_t:.1%}** win rate in {int(games_t)} games "
                    f"(overall vs this opponent: {base_wr:.1%})."
                )
                st.write(
                    f"Team effect: {sign}{abs(delta_prob) * 100:.1f} pts → "
                    f"adjusted win chance ≈ **{adj_prob:.1%}** "
                    f"(base: {base_prob:.1%})."
                )

            # Show effect for Player A (if a future team is provided)
            if team_future_p1:
                _describe_team_effect(p1, base_wr_p1, prob_a, team_future_p1)

            # Show effect for Player B (if a future team is provided)
            if team_future_p2:
                base_prob_b = 1.0 - prob_a
                _describe_team_effect(p2, base_wr_p2, base_prob_b, team_future_p2)

elif page == "Head-to-Head (2v1)":
    st.subheader(f"⚔️ 2v1 Head-to-Head – {selected_game}")

    if df_2v1_game.empty:
        st.info(f"Need some 2v1 matches recorded for {selected_game}.")
    else:
        # TODO: adjust these columns to match your sheet
        attackers = sorted(df_2v1_game["attackers_name"].dropna().unique())
        defenders = sorted(df_2v1_game["defender_player"].dropna().unique())

        col1, col2 = st.columns(2)
        with col1:
            atk = st.selectbox("Attacking duo", attackers, key="h2h_2v1_atk")
        with col2:
            dfd = st.selectbox("Defender", defenders, key="h2h_2v1_dfd")

        if atk and dfd:
            # filter to this matchup
            h2h_2v1 = df_2v1_game[
                (df_2v1_game["attackers_name"] == atk)
                & (df_2v1_game["defender_player"] == dfd)
            ].copy().sort_values("date")

            st.markdown(f"### {atk} vs {dfd} – {selected_game}")

            if h2h_2v1.empty:
                st.info("No 2v1 games for this duo vs this defender yet.")
            else:
                # from attackers’ POV
                import numpy as np

                n = len(h2h_2v1)
                wins_atk = (h2h_2v1["score_attackers"] > h2h_2v1["score_defender"]).sum()
                wins_dfd = (h2h_2v1["score_attackers"] < h2h_2v1["score_defender"]).sum()
                draws_2v1 = (h2h_2v1["score_attackers"] == h2h_2v1["score_defender"]).sum()

                gf_atk = h2h_2v1["score_attackers"].sum()
                ga_atk = h2h_2v1["score_defender"].sum()
                xg_for_atk = h2h_2v1["xG1"].fillna(0).sum()
                xg_against_atk = h2h_2v1["xG2"].fillna(0).sum()

                avg_gf_atk = gf_atk / n
                avg_ga_atk = ga_atk / n
                avg_xgf_atk = xg_for_atk / n
                avg_xga_atk = xg_against_atk / n

                st.markdown(
                    f"{atk} vs {dfd}: {gf_atk} goals scored, {ga_atk} conceded in {n} games "
                    f"(_avg {avg_gf_atk:.2f} scored, {avg_ga_atk:.2f} conceded; "
                    f"xG {avg_xgf_atk:.2f} for, {avg_xga_atk:.2f} against_)."
                )

                # Match history table
                st.markdown("### Match history")
                st.dataframe(
                    h2h_2v1[
                        [
                            "date",
                            "attackers_name",
                            "attackers_players",
                            "score_attackers",
                            "xG1",
                            "score_defender",
                            "xG2",
                            "defender_player",
                            "defender_team",
                        ]
                    ].sort_values(by="date", ascending=False),
                    use_container_width=True,
                )

                # You can now re-use the same chart patterns as 1v1,
                # but using score_attackers / score_defender and xG1 / xG2.
                # (wins/draws bar, margin distribution, goals+xG over time, etc.)

# ---------- PAGE: HEAD-TO-HEAD (2v2) ----------
elif page == "Head-to-Head (2v2)":
    st.subheader(f"👥 2v2 Head-to-Head – {selected_game}")

    if df_2v2_game.empty:
        st.info(f"Need some 2v2 matches recorded for {selected_game}.")
    else:
        import re
        import numpy as np
        import pandas as pd

        # --- Build canonical, order-independent lineup keys from team1_players/team2_players ---
        def normalize_lineup(players_str: str) -> str:
            if not isinstance(players_str, str):
                return ""
            # Split on & , or / and sort names so order doesn't matter
            parts = re.split(r"[&,/]", players_str)
            names = [p.strip() for p in parts if p.strip()]
            return " & ".join(sorted(names))

        df2 = df_2v2_game.copy()
        df2["team1_lineup_key"] = df2["team1_players"].astype(str).apply(normalize_lineup)
        df2["team2_lineup_key"] = df2["team2_players"].astype(str).apply(normalize_lineup)

        # All unique lineups seen in this game
        lineups_all = sorted(
            set(df2["team1_lineup_key"].dropna().unique()).union(
                set(df2["team2_lineup_key"].dropna().unique())
            )
        )

        if len(lineups_all) < 2:
            st.info("Need at least two different 2v2 lineups to compare.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                lineup_A = st.selectbox("Lineup A", lineups_all, key="h2h_2v2_A")
            with col2:
                lineup_B = st.selectbox(
                    "Lineup B",
                    [l for l in lineups_all if l != lineup_A],
                    key="h2h_2v2_B",
                )

            if lineup_A and lineup_B:
                # --- Head-to-head slice: all matches where A vs B (order-independent) ---
                mask_ab = (
                    (df2["team1_lineup_key"] == lineup_A) & (df2["team2_lineup_key"] == lineup_B)
                )
                mask_ba = (
                    (df2["team1_lineup_key"] == lineup_B) & (df2["team2_lineup_key"] == lineup_A)
                )
                h2h_df_2v2 = df2[mask_ab | mask_ba].copy().sort_values("date")

                st.markdown(f"### {lineup_A} vs {lineup_B} – {selected_game}")

                if h2h_df_2v2.empty:
                    st.info("No 2v2 matches between these two lineups yet for this game.")
                else:
                    # ---------- Aggregate stats from each lineup's perspective ----------
                    def _aggregate_lineup(df, focus_lineup_key: str):
                        gf = ga = 0
                        xg_for = xg_against = 0.0
                        margins_signed = []
                        margins_abs = []
                        match_idx = []
                        goals_series = []
                        xg_series = []

                        for _, row in df.iterrows():
                            if row["team1_lineup_key"] == focus_lineup_key:
                                gf_match = row["score1"]
                                ga_match = row["score2"]
                                xg_for_match = row.get("xG1", np.nan)
                                xg_again_match = row.get("xG2", np.nan)
                            else:  # focus lineup must be team2
                                gf_match = row["score2"]
                                ga_match = row["score1"]
                                xg_for_match = row.get("xG2", np.nan)
                                xg_again_match = row.get("xG1", np.nan)

                            gf += gf_match
                            ga += ga_match

                            if not np.isnan(xg_for_match):
                                xg_for += xg_for_match
                            if not np.isnan(xg_again_match):
                                xg_against += xg_again_match

                            diff = gf_match - ga_match
                            margins_signed.append(diff)
                            margins_abs.append(abs(diff))
                            goals_series.append(gf_match)
                            xg_series.append(xg_for_match)
                            match_idx.append(len(match_idx) + 1)

                        n = len(df) if len(df) else 1
                        return {
                            "gf": gf,
                            "ga": ga,
                            "xg_for": xg_for,
                            "xg_against": xg_against,
                            "avg_gf": gf / n,
                            "avg_ga": ga / n,
                            "avg_xg_for": xg_for / n,
                            "avg_xg_against": xg_against / n,
                            "margins_signed": margins_signed,
                            "margins_abs": margins_abs,
                            "match_idx": match_idx,
                            "goals_series": goals_series,
                            "xg_series": xg_series,
                        }

                    stats_A = _aggregate_lineup(h2h_df_2v2, lineup_A)
                    stats_B = _aggregate_lineup(h2h_df_2v2, lineup_B)

                    # ---------- W/D/L counts ----------
                    wins_A = wins_B = draws_2v2 = 0
                    for _, row in h2h_df_2v2.iterrows():
                        # align goals so "A" is always lineup_A
                        if row["team1_lineup_key"] == lineup_A:
                            gA, gB = row["score1"], row["score2"]
                        else:
                            gA, gB = row["score2"], row["score1"]

                        if gA > gB:
                            wins_A += 1
                        elif gA < gB:
                            wins_B += 1
                        else:
                            draws_2v2 += 1

                    total_games_2v2 = wins_A + wins_B + draws_2v2
                    if total_games_2v2 == 0:
                        st.info("No valid match data for this rivalry.")
                        st.stop()

                    # ---------- Top-line metrics ----------
                    colL, colM, colR = st.columns(3)
                    colL.metric(f"{lineup_A} wins", wins_A)
                    colM.metric("Draws", draws_2v2)
                    colR.metric(f"{lineup_B} wins", wins_B)

                    # ---------- Rivalry summary for each lineup ----------
                    colS1, colS2 = st.columns(2)
                    with colS1:
                        st.markdown(f"**{lineup_A} rivalry summary**")
                        st.write(
                            f"- Games: {total_games_2v2}\n"
                            f"- Goals for: {stats_A['gf']} (avg {stats_A['avg_gf']:.2f})\n"
                            f"- Goals against: {stats_A['ga']} (avg {stats_A['avg_ga']:.2f})\n"
                            f"- xG for: {stats_A['xg_for']:.2f} (avg {stats_A['avg_xg_for']:.2f})\n"
                            f"- xG against: {stats_A['xg_against']:.2f} (avg {stats_A['avg_xg_against']:.2f})"
                        )

                    with colS2:
                        st.markdown(f"**{lineup_B} rivalry summary**")
                        st.write(
                            f"- Games: {total_games_2v2}\n"
                            f"- Goals for: {stats_B['gf']} (avg {stats_B['avg_gf']:.2f})\n"
                            f"- Goals against: {stats_B['ga']} (avg {stats_B['avg_ga']:.2f})\n"
                            f"- xG for: {stats_B['xg_for']:.2f} (avg {stats_B['avg_xg_for']:.2f})\n"
                            f"- xG against: {stats_B['xg_against']:.2f} (avg {stats_B['avg_xg_against']:.2f})"
                        )

                    # ---------- Last 5 meetings (from lineup_A perspective) ----------
                    last5 = h2h_df_2v2.tail(5)
                    form_symbols = []
                    for _, row in last5.iterrows():
                        if row["team1_lineup_key"] == lineup_A:
                            gA, gB = row["score1"], row["score2"]
                        else:
                            gA, gB = row["score2"], row["score1"]

                        if gA == gB:
                            form_symbols.append("D")
                        elif gA > gB:
                            form_symbols.append("W")
                        else:
                            form_symbols.append("L")

                    if form_symbols:
                        st.markdown(
                            f"**Last 5 (from {lineup_A} perspective):** "
                            + " · ".join(form_symbols[::-1])
                        )

                    # ---------- Line-ups table (just showing the player strings) ----------
                    def _lineup_table(df, focus_lineup_key: str):
                        records = []
                        for _, row in df.iterrows():
                            if row["team1_lineup_key"] == focus_lineup_key:
                                lineup = row.get("team1_players", "")
                                gf = row["score1"]
                                ga = row["score2"]
                            else:
                                lineup = row.get("team2_players", "")
                                gf = row["score2"]
                                ga = row["score1"]
                            records.append({"lineup": lineup, "gf": gf, "ga": ga})
                        if not records:
                            return pd.DataFrame()
                        tab = pd.DataFrame(records)
                        tab = (
                            tab.groupby("lineup")
                            .agg(
                                games=("gf", "count"),
                                goals_for=("gf", "sum"),
                                goals_against=("ga", "sum"),
                            )
                            .reset_index()
                        )
                        tab["avg_goals_for"] = tab["goals_for"] / tab["games"]
                        tab["avg_goals_against"] = tab["goals_against"] / tab["games"]
                        return tab

                    tab_A = _lineup_table(h2h_df_2v2, lineup_A)
                    tab_B = _lineup_table(h2h_df_2v2, lineup_B)

                    st.markdown("### Line-ups used in this matchup")
                    colT1, colT2 = st.columns(2)
                    with colT1:
                        st.markdown(f"**{lineup_A} variants vs {lineup_B}**")
                        if tab_A.empty:
                            st.write("No line-up data recorded.")
                        else:
                            st.dataframe(
                                tab_A[
                                    [
                                        "lineup",
                                        "games",
                                        "goals_for",
                                        "goals_against",
                                        "avg_goals_for",
                                        "avg_goals_against",
                                    ]
                                ].style.format(
                                    {
                                        "avg_goals_for": "{:.2f}",
                                        "avg_goals_against": "{:.2f}",
                                    }
                                ),
                                use_container_width=True,
                            )
                    with colT2:
                        st.markdown(f"**{lineup_B} variants vs {lineup_A}**")
                        if tab_B.empty:
                            st.write("No line-up data recorded.")
                        else:
                            st.dataframe(
                                tab_B[
                                    [
                                        "lineup",
                                        "games",
                                        "goals_for",
                                        "goals_against",
                                        "avg_goals_for",
                                        "avg_goals_against",
                                    ]
                                ].style.format(
                                    {
                                        "avg_goals_for": "{:.2f}",
                                        "avg_goals_against": "{:.2f}",
                                    }
                                ),
                                use_container_width=True,
                            )

                    # ---------- Match history ----------
                    st.markdown("### Match history")
                    st.dataframe(
                        h2h_df_2v2[
                            [
                                "date",
                                "team1_players",
                                "score1",
                                "xG1",
                                "score2",
                                "xG2",
                                "team2_players",
                            ]
                        ].sort_values(by="date", ascending=False),
                        use_container_width=True,
                    )

                    # ---------- Charts in expander ----------
                    with st.expander("📊 Show charts"):
                        goal_margins_abs = stats_A["margins_abs"]
                        match_idx = stats_A["match_idx"]
                        A_goals = stats_A["goals_series"]
                        B_goals = stats_B["goals_series"]
                        A_xg = np.array(stats_A["xg_series"], dtype=float)
                        B_xg = np.array(stats_B["xg_series"], dtype=float)
                        score_diffs = stats_A["margins_signed"]

                        # 1) Wins / draws bar
                        c1, c2 = st.columns(2)
                        with c1:
                            fig, ax = plt.subplots(figsize=(4.0, 2.3))
                            ax.bar([lineup_A, lineup_B, "Draws"], [wins_A, wins_B, draws_2v2])
                            ax.set_ylabel("Games")
                            ax.set_title("Wins / draws")
                            ax.text(
                                0.99,
                                0.02,
                                "Counts of wins/draws in this 2v2 rivalry.",
                                transform=ax.transAxes,
                                ha="right",
                                va="bottom",
                                fontsize=7,
                                color="#cccccc",
                                alpha=0.8,
                            )
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)

                        # 2) Goal margin distribution
                        with c2:
                            fig2, ax2 = plt.subplots(figsize=(4.0, 2.3))
                            ones = sum(1 for m in goal_margins_abs if m == 1)
                            twos = sum(1 for m in goal_margins_abs if m == 2)
                            big = sum(1 for m in goal_margins_abs if m >= 3)
                            labels = ["1 goal", "2 goals", "3+ goals"]
                            counts = [ones, twos, big]
                            ax2.bar(labels, counts)
                            ax2.set_ylabel("Games")
                            ax2.set_title("Scoreline spread")
                            ax2.text(
                                0.99,
                                0.02,
                                "How often games are decided by 1, 2, or 3+ goals.",
                                transform=ax2.transAxes,
                                ha="right",
                                va="bottom",
                                fontsize=7,
                                color="#cccccc",
                                alpha=0.8,
                            )
                            st.pyplot(fig2, use_container_width=True)
                            plt.close(fig2)

                        # 3) Goals & xG per match over time
                        st.markdown("#### Goals and xG per match over time")
                        if match_idx:
                            fig3, ax3 = plt.subplots(figsize=(6, 2.6))
                            ax3.plot(match_idx, A_goals, marker="o", label=f"{lineup_A} goals")
                            ax3.plot(match_idx, B_goals, marker="o", label=f"{lineup_B} goals")

                            if np.any(~np.isnan(A_xg)):
                                ax3.plot(
                                    match_idx,
                                    A_xg,
                                    marker="x",
                                    linestyle="--",
                                    label=f"{lineup_A} xG",
                                )
                            if np.any(~np.isnan(B_xg)):
                                ax3.plot(
                                    match_idx,
                                    B_xg,
                                    marker="x",
                                    linestyle="--",
                                    label=f"{lineup_B} xG",
                                )

                            ax3.set_xlabel("Match # (chronological)")
                            ax3.set_ylabel("Goals / xG")
                            ax3.legend(fontsize=8)
                            ax3.text(
                                0.99,
                                0.02,
                                "Match-by-match goals and expected goals.",
                                transform=ax3.transAxes,
                                ha="right",
                                va="bottom",
                                fontsize=7,
                                color="#cccccc",
                                alpha=0.8,
                            )
                            st.pyplot(fig3, use_container_width=True)
                            plt.close(fig3)

                        # 4) Score difference trend
                        st.markdown(f"#### Score difference trend (positive = {lineup_A} ahead)")
                        if match_idx:
                            fig4, ax4 = plt.subplots(figsize=(6, 2.6))
                            ax4.axhline(0, color="gray", linewidth=1)
                            ax4.plot(match_idx, score_diffs, marker="o")
                            ax4.set_xlabel("Match # (chronological)")
                            ax4.set_ylabel(f"{lineup_A} − {lineup_B}")
                            st.pyplot(fig4, use_container_width=True)
                            plt.close(fig4)
                            st.markdown(
                                "<p style='font-size:0.7rem; text-align:right; opacity:0.7;'>"
                                "Each point is one 2v2 game: above 0 = win for Lineup A, below 0 = win for Lineup B."
                                "</p>",
                                unsafe_allow_html=True,
                            )

                    # ---------- Upgraded prediction: win rate + xG edge ----------
                    st.markdown("---")
                    st.markdown("### Win Prediction (for next 2v2)")

                    base_wr_A = wins_A / total_games_2v2
                    base_wr_B = wins_B / total_games_2v2

                    avg_xg_for_A = stats_A["avg_xg_for"]
                    avg_xg_for_B = stats_B["avg_xg_for"]

                    # xG edge per game (A − B)
                    xg_edge = avg_xg_for_A - avg_xg_for_B

                    # Combine into a single score:
                    #   0.5 baseline
                    #   + 0.35 * (win-rate diff)
                    #   + 0.15 * (xG edge)
                    raw_score = 0.5 + 0.35 * (base_wr_A - base_wr_B) + 0.15 * xg_edge

                    # Clamp between 5% and 95%
                    prob_A = max(0.05, min(0.95, raw_score))
                    prob_B = 1.0 - prob_A

                    st.write("This prediction uses:")
                    st.write(
                        "- Historic 2v2 win rates in this rivalry\n"
                        "- Average xG advantage between the lineups"
                    )
                    st.write("Estimated win chance next match:")
                    st.write(f"- **{lineup_A}: {prob_A:.1%}**")
                    st.write(f"- **{lineup_B}: {prob_B:.1%}**")

                    if prob_A > 0.6:
                        st.success(
                            f"Stats favour **{lineup_A}** right now. "
                            f"Perfect chance for **{lineup_B}** to play spoiler 👀"
                        )
                    elif prob_A < 0.4:
                        st.success(
                            f"Stats favour **{lineup_B}** right now. "
                            f"**{lineup_A}** comes in as the underdog."
                        )
                    else:
                        st.info("This rivalry looks tight. Either lineup could take it.")

# ---------- PAGE: ALL DATA ----------
elif page == "All Data":
    st.subheader(f"📄 All Data – {selected_game}")

    st.markdown("#### 1v1 Matches")
    if df_1v1_game.empty:
        st.info(f"No 1v1 data yet for {selected_game}.")
    else:
        col_f1, _ = st.columns(2)
        with col_f1:
            player_filter = st.selectbox(
                "Filter 1v1 by player (optional)",
                ["(All players)"] + players_game,
                key="all_data_player_filter",
            )
        filtered_1v1 = df_1v1_game.copy()
        if player_filter != "(All players)":
            mask = (filtered_1v1["player1"] == player_filter) | (filtered_1v1["player2"] == player_filter)
            filtered_1v1 = filtered_1v1[mask]

        st.dataframe(
            filtered_1v1.sort_values(by="date", ascending=False),
            use_container_width=True,
        )

    st.markdown("#### 2v2 Matches")
    if df_2v2_game.empty:
        st.info(f"No 2v2 data yet for {selected_game}.")
    else:
        teams_in_game = sorted(
            set(df_2v2_game["team1_name"].dropna().unique()).union(
                set(df_2v2_game["team2_name"].dropna().unique())
            )
        )
        col_tf1, _ = st.columns(2)
        with col_tf1:
            team_filter = st.selectbox(
                "Filter 2v2 by team (optional)",
                ["(All teams)"] + teams_in_game,
                key="all_data_team_filter",
            )
        filtered_2v2 = df_2v2_game.copy()
        if team_filter != "(All teams)":
            mask_t = (filtered_2v2["team1_name"] == team_filter) | (filtered_2v2["team2_name"] == team_filter)
            filtered_2v2 = filtered_2v2[mask_t]

        st.dataframe(
            filtered_2v2.sort_values(by="date", ascending=False),
            use_container_width=True,
        )
