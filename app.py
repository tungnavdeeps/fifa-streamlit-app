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
    """
    try:
        client = get_gsheet_client()
        sheet = client.open_by_url(SPREADSHEET_URL).worksheet(worksheet_name)
        records = sheet.get_all_records()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)
    except Exception as e:
        st.warning(f"⚠️ Could not load worksheet '{worksheet_name}': {e}")
        return pd.DataFrame()

def load_matches_1v1() -> pd.DataFrame:
    expected_cols = [
        "date", "game", "player1", "team1", "score1", "xG1", "result1",
        "player2", "team2", "score2", "xG2", "result2",
    ]
    df = load_sheet(WORKSHEET_1V1)
    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score1"] = pd.to_numeric(df["score1"], errors="coerce")
    df["score2"] = pd.to_numeric(df["score2"], errors="coerce")
    return df[expected_cols]


def load_matches_2v2() -> pd.DataFrame:
    expected_cols = [
        "date", "game", "team1_name", "team1_players", "score1", "xG1", "result1",
        "team2_name", "team2_players", "score2", "xG2", "result2",
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
    expected_cols = [
        "date", "game", "team1_name", "team1_players", "score1", "xG1", "result1",
        "team2_name", "team2_players", "score2", "xG2", "result2",
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

def append_match_1v1(date, game, player1, team1, score1, xG1, player2, team2, score2, xG2):
    client = get_gsheet_client()
    sheet = client.open_by_url(SPREADSHEET_URL).worksheet(WORKSHEET_1V1)

    if score1 > score2:
        result1, result2 = "W", "L"
    elif score1 < score2:
        result1, result2 = "L", "W"
    else:
        result1 = result2 = "D"

    row = [str(date), game, player1, team1, int(score1), float(xG1), result1,
           player2, team2, int(score2), float(xG2), result2]
    sheet.append_row(row)

def append_match_2v2(date, game, team1_name, team1_players, score1, xG1, team2_name, team2_players, score2, xG2):
    client = get_gsheet_client()
    sheet = client.open_by_url(SPREADSHEET_URL).worksheet(WORKSHEET_2V2)

    if score1 > score2:
        result1, result2 = "W", "L"
    elif score1 < score2:
        result1, result2 = "L", "W"
    else:
        result1 = result2 = "D"

    row = [str(date), game, team1_name, team1_players, int(score1), float(xG1), result1,
           team2_name, team2_players, int(score2), float(xG2), result2]
    sheet.append_row(row)

def append_match_2v1(date, game, team1_name, team1_players, score1, xG1, team2_name, team2_players, score2, xG2):
    client = get_gsheet_client()
    sheet = client.open_by_url(SPREADSHEET_URL).worksheet(WORKSHEET_2V1)

    if score1 > score2:
        result1, result2 = "W", "L"
    elif score1 < score2:
        result1, result2 = "L", "W"
    else:
        result1 = result2 = "D"

    row = [str(date), game, team1_name, team1_players, int(score1), float(xG1), result1,
           team2_name, team2_players, int(score2), float(xG2), result2]
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
        return pd.DataFrame(columns=[
            "player", "games", "wins", "draws", "losses", "goals_for", "goals_against", 
            "goal_diff", "avg_goals_for", "avg_goals_against", "win_pct", "elo_rating"
        ])

    rows = []
    for _, row in df_game.iterrows():
        p1, p2 = row["player1"], row["player2"]
        s1, s2 = row["score1"], row["score2"]

        if pd.isna(p1) or pd.isna(p2) or pd.isna(s1) or pd.isna(s2):
            continue

        if s1 > s2: r1, r2 = "W", "L"
        elif s1 < s2: r1, r2 = "L", "W"
        else: r1 = r2 = "D"

        rows.append({"player": p1, "goals_for": s1, "goals_against": s2, "result": r1})
        rows.append({"player": p2, "goals_for": s2, "goals_against": s1, "result": r2})

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
    grouped["elo_rating"] = grouped.index.map(lambda p: round(ratings.get(p, 1000)))
    grouped = grouped.sort_values(by=["elo_rating", "wins"], ascending=False).reset_index()

    return grouped


def build_team_leaderboard_2v2(df: pd.DataFrame, game: str) -> pd.DataFrame:
    df_game = df[df["game"] == game].copy()
    if df_game.empty:
        return pd.DataFrame(columns=[
            "team", "players", "games", "wins", "draws", "losses", "goals_for", 
            "goals_against", "goal_diff", "avg_goals_for", "avg_goals_against", "win_pct"
        ])

    rows = []
    for _, row in df_game.iterrows():
        t1, t2 = row["team1_name"], row["team2_name"]
        p1_players, p2_players = row["team1_players"], row["team2_players"]
        s1, s2 = row["score1"], row["score2"]

        if pd.isna(t1) or pd.isna(t2) or pd.isna(s1) or pd.isna(s2):
            continue

        if s1 > s2: r1, r2 = "W", "L"
        elif s1 < s2: r1, r2 = "L", "W"
        else: r1 = r2 = "D"

        rows.append({"team": f"{lineup_key(p1_players)} ({t1})", "players": lineup_key(p1_players), "club": t1, "goals_for": s1, "goals_against": s2, "result": r1})
        rows.append({"team": f"{lineup_key(p2_players)} ({t2})", "players": lineup_key(p2_players), "club": t2, "goals_for": s2, "goals_against": s1, "result": r2})

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

    grouped = grouped.sort_values(by=["win_pct", "goal_diff"], ascending=False).reset_index()

    return grouped


def head_to_head_1v1(df: pd.DataFrame, game: str, p1: str, p2: str) -> pd.DataFrame:
    df_game = df[df["game"] == game].copy()
    mask = (
        ((df_game["player1"] == p1) & (df_game["player2"] == p2))
        | ((df_game["player1"] == p2) & (df_game["player2"] == p1))
    )
    return df_game[mask].copy()


def _prediction_components_1v1(df: pd.DataFrame, game: str, player_a: str, player_b: str):
    ratings = compute_ratings_1v1(df, game)
    ra = ratings.get(player_a, 1000)
    rb = ratings.get(player_b, 1000)
    elo_prob = expected_score(ra, rb)

    h2h_df = head_to_head_1v1(df, game, player_a, player_b)
    h2h_prob = None
    xg_prob = None

    if not h2h_df.empty:
        total_games = len(h2h_df)
        wins_a = (
            ((h2h_df["player1"] == player_a) & (h2h_df["score1"] > h2h_df["score2"])).sum()
            + ((h2h_df["player2"] == player_a) & (h2h_df["score2"] > h2h_df["score1"])).sum()
        )
        draws = (h2h_df["score1"] == h2h_df["score2"]).sum()

        if total_games > 0:
            h2h_prob_overall = (wins_a + 0.5 * draws) / total_games
        else:
            h2h_prob_overall = None

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
                xg_prob = 1.0 / (1.0 + np.exp(-avg_diff / 0.75))

    components = [elo_prob]
    weights = [0.5] 
    if h2h_prob is not None:
        components.append(h2h_prob)
        weights.append(0.3)
    if xg_prob is not None:
        components.append(xg_prob)
        weights.append(0.2)

    w_sum = sum(weights)
    final_prob = sum(c * w for c, w in zip(components, weights)) / w_sum
    final_prob = max(0.05, min(0.95, final_prob)) 

    return {"ra": ra, "rb": rb, "elo_prob": elo_prob, "h2h_prob": h2h_prob, "xg_prob": xg_prob, "final_prob": final_prob}

def predict_match_1v1(df: pd.DataFrame, game: str, player_a: str, player_b: str):
    comps = _prediction_components_1v1(df, game, player_a, player_b)
    return comps["ra"], comps["rb"], comps["final_prob"]


# =========================
# UTILS FOR INPUT UI
# =========================
def player_input_block(label, existing_players, key_prefix):
    options = ["-- Select existing --"] + sorted(existing_players)
    selected = st.selectbox(f"{label} (existing)", options, key=f"{key_prefix}_select")
    new_name = st.text_input(f"{label} (new, if not in list)", key=f"{key_prefix}_new").strip()

    if new_name:
        return new_name
    if selected != "-- Select existing --":
        return selected
    return ""

def team_input_block(label, existing_teams, key_prefix):
    """ Helper to show a dropdown of previously selected teams or type a new one. """
    options = ["-- Select existing --"] + sorted(existing_teams)
    selected = st.selectbox(f"{label} (existing)", options, key=f"{key_prefix}_select")
    new_team = st.text_input(f"{label} (new, if not in list)", key=f"{key_prefix}_new").strip()
    
    if new_team:
        return new_team
    if selected != "-- Select existing --":
        return selected
    return ""

# ---------- LINEUP NORMALIZATION HELPERS (for 2v2 / 2v1) ----------
def normalize_player_list(players_string: str):
    if not isinstance(players_string, str) or not players_string.strip():
        return tuple()
    parts = [p.strip().lower() for p in players_string.split(",")]
    parts = [p for p in parts if p] 
    return tuple(sorted(set(parts))) 


def lineup_key(players_string: str) -> str:
    names = normalize_player_list(players_string)
    if not names:
        return ""
    return " + ".join(n.title() for n in names)


def same_lineup(a: str, b: str) -> bool:
    return normalize_player_list(a) == normalize_player_list(b)


# =========================
# CACHED DATA LOADER (REDUCE QUOTA)
# =========================
@st.cache_data(ttl=600)
def load_all_data():
    df1 = load_matches_1v1()
    df2 = load_matches_2v2()
    df3 = load_matches_2v1()
    return df1, df2, df3

# =========================
# STREAMLIT UI & LOGIN LOGIC
# =========================

# Ensure data is loaded BEFORE setting up the sidebar so we have access to unique teams/players
df_1v1, df_2v2, df_2v1 = load_all_data()

# Extra safety on frames
EXPECTED_1V1_COLS = ["date", "game", "player1", "team1", "score1", "xG1", "result1", "player2", "team2", "score2", "xG2", "result2"]
EXPECTED_2V2_COLS = ["date", "game", "team1_name", "team1_players", "score1", "xG1", "result1", "team2_name", "team2_players", "score2", "xG2", "result2"]
EXPECTED_2V1_COLS = ["date", "game", "team1_name", "team1_players", "score1", "xG1", "result1", "team2_name", "team2_players", "score2", "xG2", "result2"]

for col in EXPECTED_1V1_COLS:
    if col not in df_1v1.columns: df_1v1[col] = None
for col in EXPECTED_2V2_COLS:
    if col not in df_2v2.columns: df_2v2[col] = None
for col in EXPECTED_2V1_COLS:
    if col not in df_2v1.columns: df_2v1[col] = None

df_1v1 = df_1v1[EXPECTED_1V1_COLS]
df_2v2 = df_2v2[EXPECTED_2V2_COLS]
df_2v1 = df_2v1[EXPECTED_2V1_COLS]

st.sidebar.markdown("### ⚙️ Settings")
selected_game = st.sidebar.selectbox("Game version", GAME_OPTIONS)

# =========================
# GOOGLE LOGIN (SIDEBAR)
# =========================
st.sidebar.markdown("### 🔐 Access")

# Safely check login status to prevent crashes if secrets are missing
is_logged_in = getattr(st.user, "is_logged_in", False)

if not is_logged_in:
    st.sidebar.button("Log in with Google", on_click=st.login)
    st.sidebar.caption("Log in to record matches. Guests can view stats.")
else:
    st.sidebar.success(f"Logged in as: {st.user.name}")
    st.sidebar.button("Log out", on_click=st.logout)

st.sidebar.markdown("---")
refresh_clicked = st.sidebar.button("🔄 Refresh data from Google Sheets")
if refresh_clicked:
    load_all_data.clear()

# =========================
# NAVIGATION LOGIC
# =========================
nav_pages = ["Dashboard", "Head-to-Head (1v1)", "Head-to-Head (2v2)", "Head-to-Head (2v1)", "All Data"]

if is_logged_in:
    nav_pages.insert(1, "Record Match")

page = st.sidebar.radio("Go to", nav_pages)

# Data filtering for UI
df_1v1_game = df_1v1[df_1v1["game"] == selected_game].copy()
df_2v2_game = df_2v2[df_2v2["game"] == selected_game].copy()
df_2v1_game = df_2v1[df_2v1["game"] == selected_game].copy()

# Extract unique players
players_all = sorted(set(df_1v1["player1"].dropna().unique()).union(set(df_1v1["player2"].dropna().unique())))
players_game = sorted(set(df_1v1_game["player1"].dropna().unique()).union(set(df_1v1_game["player2"].dropna().unique())))

# Extract unique teams globally for the new dropdown
teams_1v1 = set(df_1v1["team1"].dropna().unique()).union(set(df_1v1["team2"].dropna().unique()))
teams_2v2 = set(df_2v2["team1_name"].dropna().unique()).union(set(df_2v2["team2_name"].dropna().unique()))
teams_2v1 = set(df_2v1["team1_name"].dropna().unique()).union(set(df_2v1["team2_name"].dropna().unique()))
teams_all = sorted(teams_1v1.union(teams_2v2).union(teams_2v1))


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
        top_player_row = leaderboard_players.iloc[0]
        top_player = top_player_row["player"]
        top_player_elo = top_player_row["elo_rating"]
        
        best_attacker_row = leaderboard_players.sort_values(by="avg_goals_for", ascending=False).iloc[0]
        best_attacker = best_attacker_row["player"]
        best_attacker_gpg = best_attacker_row["avg_goals_for"]
    else:
        top_player, top_player_elo, best_attacker, best_attacker_gpg = "N/A", 0, "N/A", 0

    colA.metric("Total matches", int(total_matches))
    colB.metric("Total goals", int(total_goals))
    colC.metric("Top ELO player", top_player, f"ELO {int(top_player_elo)}")
    colD.metric("Most goals per game", best_attacker, f"{best_attacker_gpg:.2f} goals/game" if best_attacker != "N/A" else "")

    st.markdown("---")

    # ==========================================
    # 1V1 SECTION (AWARDS + LEADERBOARD)
    # ==========================================
    st.markdown("### 🏆 1v1 Player Awards & Titles")

    if leaderboard_players.empty:
        st.info("No 1v1 matches yet to calculate player titles.")
    else:
        from collections import defaultdict
        awards_1v1 = {}

        # Golden Boot – total goals
        golden_boot = leaderboard_players.sort_values("goals_for", ascending=False).iloc[0]
        awards_1v1["Golden Boot (Top Scorer)"] = f"**{golden_boot['player']}** – {int(golden_boot['goals_for'])} goals"

        eligible = leaderboard_players[leaderboard_players["games"] >= 5].copy()

        if not eligible.empty:
            best_def = eligible.sort_values("avg_goals_against").iloc[0]
            awards_1v1["Brick Wall (Best Defense)"] = f"**{best_def['player']}** – {best_def['avg_goals_against']:.2f} avg conceded"

            best_wr = eligible.sort_values("win_pct", ascending=False).iloc[0]
            awards_1v1["Consistent Winner (Highest Win Rate)"] = f"**{best_wr['player']}** – {best_wr['win_pct']:.1%} wins"

        # Attack threat
        top_attack = leaderboard_players.sort_values("avg_goals_for", ascending=False).iloc[0]
        awards_1v1["Most Feared Rival (Attack Threat)"] = f"**{top_attack['player']}** – {top_attack['avg_goals_for']:.2f} avg goals per game"

        # Show all 1v1 awards
        for title, desc in awards_1v1.items():
            st.markdown(f"🏅 **{title}:** {desc}")

    st.markdown("#### 👤 1v1 Player Leaderboard")
    if leaderboard_players.empty:
        st.info(f"No 1v1 matches yet for {selected_game}.")
    else:
        leaderboard_players = leaderboard_players.sort_values(
            by=["elo_rating", "wins", "goal_diff"], ascending=[False, False, False]
        ).reset_index(drop=True)
        leaderboard_players.insert(0, "Rank", leaderboard_players.index + 1)

        display_cols = ["Rank", "player", "games", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff", "avg_goals_for", "avg_goals_against", "win_pct", "elo_rating"]
        st.dataframe(
            leaderboard_players[display_cols].style.format({
                "avg_goals_for": "{:.2f}", "avg_goals_against": "{:.2f}", "win_pct": "{:.1%}", "elo_rating": "{:.0f}"
            }), 
            use_container_width=True
        )

    st.markdown("---")

    # ==========================================
    # 2V2 SECTION (AWARDS + LEADERBOARD)
    # ==========================================
    st.markdown("### 🏆 2v2 Team Awards & Titles")

    if leaderboard_teams.empty:
        st.info("No 2v2 matches yet to calculate team titles.")
    else:
        awards_2v2 = {}
        eligible_teams = leaderboard_teams[leaderboard_teams["games"] >= 3].copy()

        if not eligible_teams.empty:
            best_team = eligible_teams.sort_values("win_pct", ascending=False).iloc[0]
            awards_2v2["Golden Duo (Best 2v2 Team)"] = f"**{best_team['team']}** – {best_team['win_pct']:.1%} win rate ({int(best_team['games'])} games)"

            best_def_team = eligible_teams.sort_values("avg_goals_against").iloc[0]
            awards_2v2["Fortress Duo (Best Defense)"] = f"**{best_def_team['team']}** – {best_def_team['avg_goals_against']:.2f} avg conceded"

        top_attack_team = leaderboard_teams.sort_values("avg_goals_for", ascending=False).iloc[0]
        awards_2v2["Attacking Duo (Most Goals per Game)"] = f"**{top_attack_team['team']}** – {top_attack_team['avg_goals_for']:.2f} avg goals scored"

        for title, text in awards_2v2.items():
            st.markdown(f"🏅 **{title}:** {text}")

    st.markdown("#### 👥 2v2 Team Leaderboard")
    if leaderboard_teams.empty:
        st.info(f"No 2v2 matches yet for {selected_game}.")
    else:
        # FIXED: Removed 'elo_rating' from the sort parameters since 2v2 doesn't compute ELO
        leaderboard_teams = leaderboard_teams.sort_values(
            by=["win_pct", "goal_diff"], ascending=[False, False]
        ).reset_index(drop=True)
        
        if "Rank" not in leaderboard_teams.columns:
            leaderboard_teams.insert(0, "Rank", leaderboard_teams.index + 1)

        # FIXED: Removed 'elo_rating' from desired_cols_t
        desired_cols_t = ["Rank", "team", "players", "games", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff", "avg_goals_for", "avg_goals_against", "win_pct"]
        display_cols_t = [c for c in desired_cols_t if c in leaderboard_teams.columns]

        fmt_team = {"avg_goals_for": "{:.2f}", "avg_goals_against": "{:.2f}", "win_pct": "{:.1%}"}
        fmt_team = {k: v for k, v in fmt_team.items() if k in display_cols_t}

        st.dataframe(leaderboard_teams[display_cols_t].style.format(fmt_team), use_container_width=True)


# ---------- PAGE: RECORD MATCH ----------
elif page == "Record Match" and st.user.is_logged_in:
    st.subheader(f"📝 Record a Match – {selected_game}")

    match_type = st.radio("Match type", ["1v1", "2v2", "2v1"], horizontal=True, key="match_type_radio")
    date = st.date_input("Match date", value=datetime.date.today())
    game_for_entry = st.selectbox("Game", GAME_OPTIONS, index=GAME_OPTIONS.index(selected_game))
    
    # Setup score options dropdown range (0 to 50 goals)
    score_options = list(range(0, 51))

    st.markdown("#### Enter details")

    # -------------------------
    #          1v1
    # -------------------------
    if match_type == "1v1":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Player 1**")
            p1 = player_input_block("Player 1", players_all, key_prefix="p1_input")
            team1 = team_input_block("Team 1", teams_all, key_prefix="team1_input")
            score1 = st.selectbox("Goals scored by Player 1", options=score_options, key="score1_1v1")
            xG1_1v1 = st.number_input("Expected goals (xG) for Player 1", min_value=0.0, step=0.1, key="xg1_1v1")

        with col2:
            st.markdown("**Player 2**")
            p2 = player_input_block("Player 2", players_all, key_prefix="p2_input")
            team2 = team_input_block("Team 2", teams_all, key_prefix="team2_input")
            score2 = st.selectbox("Goals scored by Player 2", options=score_options, key="score2_1v1")
            xG2_1v1 = st.number_input("Expected goals (xG) for Player 2", min_value=0.0, step=0.1, key="xg2_1v1")

        if st.button("Save 1v1 match", use_container_width=True):
            if not p1 or not p2: st.error("Please fill in both player names.")
            elif p1 == p2: st.error("Players must be different.")
            elif not team1 or not team2: st.error("Please enter both Team 1 and Team 2.")
            else:
                append_match_1v1(date, game_for_entry, p1, team1, score1, xG1_1v1, p2, team2, score2, xG2_1v1)
                st.success(f"Saved 1v1 match for {game_for_entry}! 🎉")
                load_all_data.clear()

    # -------------------------
    #          2v2
    # -------------------------
    elif match_type == "2v2":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Team 1**")
            team1_name = team_input_block("Team 1 name", teams_all, key_prefix="team1_name_2v2")
            team1_players = st.text_input("Team 1 players (e.g. Sue & Alex)", key="team1_players").strip()
            score1_2v2 = st.selectbox("Goals scored by Team 1", options=score_options, key="score1_2v2")
            xG1_2v2 = st.number_input("Expected goals (xG) for Team 1", min_value=0.0, step=0.1, key="xg1_2v2")

        with col2:
            st.markdown("**Team 2**")
            team2_name = team_input_block("Team 2 name", teams_all, key_prefix="team2_name_2v2")
            team2_players = st.text_input("Team 2 players (e.g. Jordan & Max)", key="team2_players").strip()
            score2_2v2 = st.selectbox("Goals scored by Team 2", options=score_options, key="score2_2v2")
            xG2_2v2 = st.number_input("Expected goals (xG) for Team 2", min_value=0.0, step=0.1, key="xg2_2v2")

        if st.button("Save 2v2 match", use_container_width=True):
            if not team1_name or not team2_name: st.error("Please fill in both team names.")
            elif team1_name == team2_name: st.error("Teams must be different.")
            else:
                append_match_2v2(date, game_for_entry, team1_name, team1_players, score1_2v2, xG1_2v2, team2_name, team2_players, score2_2v2, xG2_2v2)
                st.success(f"Saved 2v2 match for {game_for_entry}! 🎉")
                load_all_data.clear()

    # -------------------------
    #          2v1
    # -------------------------
    elif match_type == "2v1":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Team 1 (2-player side)**")
            team1_name = team_input_block("Team 1 name", teams_all, key_prefix="team1_name_2v1")
            team1_players = st.text_input("Team 1 players (e.g. Sue & Alex)", key="2v1_team1_players").strip()
            score1_2v1 = st.selectbox("Goals scored by Team 1", options=score_options, key="2v1_score1")
            xG1_2v1 = st.number_input("Expected goals (xG) for Team 1", min_value=0.0, step=0.1, key="xg1_2v1")

        with col2:
            st.markdown("**Team 2 (solo player)**")
            team2_name = team_input_block("Solo Player Club", teams_all, key_prefix="team2_name_2v1")
            team2_players = player_input_block("Solo player name", players_all, key_prefix="2v1_team2_players")
            score2_2v1 = st.selectbox("Goals scored by Solo Player", options=score_options, key="2v1_score2")
            xG2_2v1 = st.number_input("Expected goals (xG) for Team 2", min_value=0.0, step=0.1, key="xg2_2v1")

        if st.button("Save 2v1 match", use_container_width=True):
            if not team1_name or not team2_name: st.error("Please fill in both side names.")
            elif team1_name == team2_name: st.error("Team names must be different.")
            else:
                append_match_2v1(date, game_for_entry, team1_name, team1_players, score1_2v1, xG1_2v1, team2_name, team2_players, score2_2v1, xG2_2v1)
                st.success(f"Saved 2v1 match for {game_for_entry}! 🎉")
                load_all_data.clear()

# ---------- PAGE: HEAD-TO-HEAD (1v1) ----------
elif page == "Head-to-Head (1v1)":
    st.subheader(f"🔍 1v1 Head-to-Head – {selected_game}")

    if len(players_game) < 2 or df_1v1_game.empty:
        st.info(f"Need at least 2 players and some 1v1 matches recorded for {selected_game}.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.selectbox("Player A", players_game, key="h2h_p1")
        with col2:
            p2 = st.selectbox("Player B", [p for p in players_game if p != p1], key="h2h_p2")

        if p1 and p2:
            h2h_df = df_1v1[
                (df_1v1["game"] == selected_game)
                & (
                    ((df_1v1["player1"] == p1) & (df_1v1["player2"] == p2))
                    | ((df_1v1["player1"] == p2) & (df_1v1["player2"] == p1))
                )
            ].copy()

            st.markdown(f"### {p1} vs {p2} – {selected_game}")

            if h2h_df.empty:
                st.info("No direct 1v1 matches between these two for this game yet.")
            else:
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
                score_diffs = []       # from p1 perspective
                goal_margins_abs = []  # absolute margin

                total_xg_p1 = 0.0
                total_xg_p2 = 0.0

                for i, row in h2h_df.iterrows():
                    s1, s2 = row["score1"], row["score2"]
                    pl1, pl2 = row["player1"], row["player2"]
                    x1 = row.get("xG1", 0.0) or 0.0
                    x2 = row.get("xG2", 0.0) or 0.0

                    if pl1 == p1:
                        g_for, g_against = s1, s2
                        xg_for, xg_against = x1, x2
                    else:
                        g_for, g_against = s2, s1
                        xg_for, xg_against = x2, x1

                    if g_for > g_against: wins_p1 += 1
                    elif g_for < g_against: wins_p2 += 1
                    else: draws += 1

                    match_idx.append(i + 1)
                    p1_goals.append(g_for)
                    p2_goals.append(g_against)
                    p1_xg_list.append(xg_for)
                    p2_xg_list.append(xg_against)

                    score_diff = g_for - g_against
                    score_diffs.append(score_diff)
                    goal_margins_abs.append(abs(score_diff))

                    total_xg_p1 += xg_for
                    total_xg_p2 += xg_against

                total_matches = len(h2h_df)
                total_goals_p1 = sum(p1_goals)
                total_goals_p2 = sum(p2_goals)

                # Last 5 from p1 perspective
                last5 = h2h_df.tail(5)
                last5_results = []
                for _, row in last5.iterrows():
                    s1, s2 = row["score1"], row["score2"]
                    pl1 = row["player1"]
                    if pl1 == p1: g_for, g_against = s1, s2
                    else: g_for, g_against = s2, s1

                    if g_for > g_against: last5_results.append("W")
                    elif g_for < g_against: last5_results.append("L")
                    else: last5_results.append("D")

                last5_str = " ".join(last5_results) if last5_results else "N/A"

                # ----- SUMMARY TILES -----
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(f"{p1} record", f"{wins_p1}W–{draws}D–{wins_p2}L", help="From Player A perspective")
                with c2:
                    st.metric(f"{p1} goals vs {p2}", f"{total_goals_p1} – {total_goals_p2}", help="Total goals for and against in this matchup")
                with c3:
                    st.metric("Last 5 (from Player A side)", last5_str, help="Most recent 5 results from Player A perspective")

                st.markdown("---")

                # ----- TEAMS USED TABLE -----
                def summarize_teams_for_player(hdf, focus_player):
                    rows = []
                    for _, r in hdf.iterrows():
                        if r["player1"] == focus_player:
                            team = r.get("team1", "")
                            g_for, g_against = r["score1"], r["score2"]
                            xg_for = r.get("xG1", 0.0) or 0.0
                            xg_against = r.get("xG2", 0.0) or 0.0
                        else:
                            team = r.get("team2", "")
                            g_for, g_against = r["score2"], r["score1"]
                            xg_for = r.get("xG2", 0.0) or 0.0
                            xg_against = r.get("xG1", 0.0) or 0.0
                        rows.append({"team": team, "goals_for": g_for, "goals_against": g_against, "xg_for": xg_for, "xg_against": xg_against})
                    
                    if not rows:
                        return pd.DataFrame(columns=["team", "games", "goals_for", "goals_against", "xg_for", "xg_against", "avg_goals_for", "avg_goals_against", "avg_xg_for", "avg_xg_against"])
                    
                    tmp = pd.DataFrame(rows)
                    grouped = tmp.groupby("team").agg(
                        games=("goals_for", "count"),
                        goals_for=("goals_for", "sum"),
                        goals_against=("goals_against", "sum"),
                        xg_for=("xg_for", "sum"),
                        xg_against=("xg_against", "sum"),
                    )
                    grouped["avg_goals_for"] = grouped["goals_for"] / grouped["games"]
                    grouped["avg_goals_against"] = grouped["goals_against"] / grouped["games"]
                    grouped["avg_xg_for"] = grouped["xg_for"] / grouped["games"]
                    grouped["avg_xg_against"] = grouped["xg_against"] / grouped["games"]
                    return grouped.reset_index()

                st.markdown("### Line-ups and clubs used in this matchup")
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown(f"**{p1} – clubs & record**")
                    st.dataframe(summarize_teams_for_player(h2h_df, p1), use_container_width=True)
                with col_right:
                    st.markdown(f"**{p2} – clubs & record**")
                    st.dataframe(summarize_teams_for_player(h2h_df, p2), use_container_width=True)

                st.markdown("---")

                # ----- MATCH HISTORY TABLE -----
                st.markdown("### Match history")
                display_cols = ["date", "player1", "team1", "score1", "xG1", "player2", "team2", "score2", "xG2"]
                mh = h2h_df[display_cols].copy()
                mh["date"] = pd.to_datetime(mh["date"]).dt.strftime("%Y-%m-%d")
                st.dataframe(mh, use_container_width=True)

                # ----------------- CHARTS IN EXPANDER -----------------
                with st.expander("📊 Show charts"):
                    fig1, ax1 = plt.subplots()
                    ax1.bar(["Wins " + p1, "Draws", "Wins " + p2], [wins_p1, draws, wins_p2])
                    ax1.set_title("Win / draw counts")
                    ax1.set_ylabel("Matches")
                    st.pyplot(fig1)

                    fig2, ax2 = plt.subplots()
                    ax2.hist(goal_margins_abs, bins=range(0, max(goal_margins_abs) + 2))
                    ax2.set_title("Goal margin distribution (absolute)")
                    ax2.set_xlabel("Goal difference")
                    ax2.set_ylabel("Matches")
                    st.pyplot(fig2)

                    fig3, ax3 = plt.subplots()
                    ax3.scatter(p1_xg_list, p1_goals, label=f"{p1} goals", marker="o")
                    ax3.scatter(p2_xg_list, p2_goals, label=f"{p2} goals", marker="x")
                    ax3.set_xlabel("xG in match")
                    ax3.set_ylabel("Goals scored")
                    ax3.set_title("Goals vs xG per match")
                    ax3.legend()
                    st.pyplot(fig3)

                    fig4, ax4 = plt.subplots()
                    ax4.plot(match_idx, score_diffs, marker="o")
                    ax4.axhline(0, linestyle="--")
                    ax4.set_xlabel("Match (chronological)")
                    ax4.set_ylabel("Score difference (P1 - P2)")
                    ax4.set_title("Score difference over time (Player A perspective)")
                    st.pyplot(fig4)

                # ---------- ADVANCED HYBRID PREDICTION ----------
                st.markdown("### 🔮 Advanced prediction for next 1v1")
                ra, rb, prob_elo = predict_match_1v1(df_1v1, selected_game, p1, p2)

                if total_matches > 0:
                    h2h_points_p1 = wins_p1 + 0.5 * draws
                    h2h_points_p2 = wins_p2 + 0.5 * draws
                    prob_h2h_p1 = h2h_points_p1 / (h2h_points_p1 + h2h_points_p2)
                else:
                    prob_h2h_p1 = 0.5

                if (total_xg_p1 + total_xg_p2) > 0:
                    prob_xg_p1 = total_xg_p1 / (total_xg_p1 + total_xg_p2)
                else:
                    prob_xg_p1 = 0.5

                w_elo, w_h2h, w_xg = 0.5, 0.3, 0.2
                prob_hybrid_p1 = (w_elo * prob_elo + w_h2h * prob_h2h_p1 + w_xg * prob_xg_p1)
                prob_hybrid_p2 = 1 - prob_hybrid_p1

                st.write(f"Model blend: **50% ELO**, **30% head-to-head**, **20% xG share**.")
                st.write(f"- ELO model favours **{p1}** at {prob_elo:.1%} (ELO {round(ra)} vs {round(rb)})")
                st.write(f"- Head-to-head points give **{p1}** about {prob_h2h_p1:.1%} chance.")
                st.write(f"- xG share across these matches gives **{p1}** about {prob_xg_p1:.1%} chance.")

                st.markdown("**Blended prediction for next match:**")
                st.write(f"- **{p1}: {prob_hybrid_p1:.1%}**")
                st.write(f"- **{p2}: {prob_hybrid_p2:.1%}**")

                if prob_hybrid_p1 > 0.6: st.success(f"Overall, **{p1}** is the favourite in this matchup.")
                elif prob_hybrid_p1 < 0.4: st.success(f"Overall, **{p2}** is the favourite in this matchup.")
                else: st.info("This looks like a pretty even matchup on all models.")

# ---------- PAGE: HEAD-TO-HEAD (2v2) ----------
elif page == "Head-to-Head (2v2)":
    st.subheader(f"👥 2v2 Head-to-Head – {selected_game}")

    if df_2v2_game.empty:
        st.info(f"No 2v2 matches yet for {selected_game}.")
    else:
        all_lineups = sorted(set(df_2v2_game["team1_players"].dropna().unique()).union(set(df_2v2_game["team2_players"].dropna().unique())))

        col1, col2 = st.columns(2)
        with col1:
            team1_players = st.selectbox("Team A line-up", all_lineups, key="h2h2v2_a")
        with col2:
            team2_players = st.selectbox("Team B line-up", [t for t in all_lineups if t != team1_players], key="h2h2v2_b")

        if team1_players and team2_players:
            h2h_df_2v2 = df_2v2_game[
                (
                    (df_2v2_game["team1_players"] == team1_players) & (df_2v2_game["team2_players"] == team2_players)
                ) | (
                    (df_2v2_game["team1_players"] == team2_players) & (df_2v2_game["team2_players"] == team1_players)
                )
            ].copy()

            st.markdown(f"### {team1_players} vs {team2_players}")

            if h2h_df_2v2.empty:
                st.info("No 2v2 matches recorded for this exact line-up matchup yet.")
            else:
                h2h_df_2v2 = h2h_df_2v2.sort_values("date").reset_index(drop=True)

                wins_a = 0
                wins_b = 0
                draws = 0
                goal_margins_abs = []
                idx = []
                diff_trend = []

                for i, row in h2h_df_2v2.iterrows():
                    s1, s2 = row["score1"], row["score2"]
                    if row["team1_players"] == team1_players: g_a, g_b = s1, s2
                    else: g_a, g_b = s2, s1

                    if g_a > g_b: wins_a += 1
                    elif g_a < g_b: wins_b += 1
                    else: draws += 1

                    margin = g_a - g_b
                    goal_margins_abs.append(abs(margin))
                    idx.append(i + 1)
                    diff_trend.append(margin)

                total_matches = len(h2h_df_2v2)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(f"{team1_players} record", f"{wins_a}W–{draws}D–{wins_b}L", help="From Team A line-up perspective")
                with c2:
                    st.metric("Total matches", total_matches)
                with c3:
                    st.metric("Average goal margin (A - B)", f"{np.mean(diff_trend):.2f}")

                st.markdown("---")

                # Line-ups + clubs used table
                st.markdown("### Line-ups and clubs used in this matchup")
                def _lineup_table(df, focus_lineup):
                    rows = []
                    for _, r in df.iterrows():
                        if r["team1_players"] == focus_lineup:
                            club = r.get("team1_name", "")
                            g_for, g_against = r["score1"], r["score2"]
                        else:
                            club = r.get("team2_name", "")
                            g_for, g_against = r["score2"], r["score1"]
                        xg_for = r.get("xG1", 0.0) or 0.0
                        xg_against = r.get("xG2", 0.0) or 0.0
                        rows.append({"club": club, "goals_for": g_for, "goals_against": g_against, "xg_for": xg_for, "xg_against": xg_against})

                    if not rows:
                        return pd.DataFrame(columns=["club", "games", "goals_for", "goals_against", "xg_for", "xg_against"])
                    
                    tmp = pd.DataFrame(rows)
                    grouped = tmp.groupby("club").agg(
                        games=("goals_for", "count"),
                        goals_for=("goals_for", "sum"),
                        goals_against=("goals_against", "sum"),
                        xg_for=("xg_for", "sum"),
                        xg_against=("xg_against", "sum"),
                    )
                    return grouped.reset_index()

                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown(f"**{team1_players} – clubs & record**")
                    st.dataframe(_lineup_table(h2h_df_2v2, team1_players), use_container_width=True)
                with col_r:
                    st.markdown(f"**{team2_players} – clubs & record**")
                    st.dataframe(_lineup_table(h2h_df_2v2, team2_players), use_container_width=True)

                st.markdown("---")

                # Match history
                st.markdown("### Match history")
                cols_mh = ["date", "team1_name", "team1_players", "score1", "xG1", "team2_name", "team2_players", "score2", "xG2"]
                mh2 = h2h_df_2v2[cols_mh].copy()
                mh2["date"] = pd.to_datetime(mh2["date"]).dt.strftime("%Y-%m-%d")
                st.dataframe(mh2, use_container_width=True)

                with st.expander("📊 Show charts"):
                    fig1, ax1 = plt.subplots()
                    ax1.bar([f"Wins\n{team1_players}", "Draws", f"Wins\n{team2_players}"], [wins_a, draws, wins_b])
                    ax1.set_ylabel("Matches")
                    ax1.set_title("2v2 win / draw counts")
                    st.pyplot(fig1)

                    fig2, ax2 = plt.subplots()
                    ax2.hist(goal_margins_abs, bins=range(0, max(goal_margins_abs) + 2))
                    ax2.set_xlabel("Absolute goal margin")
                    ax2.set_ylabel("Matches")
                    ax2.set_title("Goal margin distribution")
                    st.pyplot(fig2)

# ---------- PAGE: HEAD-TO-HEAD (2v1) ----------
elif page == "Head-to-Head (2v1)":
    st.subheader(f"⚔️ 2v1 Head-to-Head – {selected_game}")

    if df_2v1_game.empty:
        st.info(f"No 2v1 matches yet for {selected_game}.")
    else:
        attackers = sorted(df_2v1_game["team1_players"].dropna().unique())
        defenders = sorted(df_2v1_game["team2_players"].dropna().unique())

        col1, col2 = st.columns(2)
        with col1:
            atk = st.selectbox("Attacking duo", attackers, key="h2h2v1_atk")
        with col2:
            dfd = st.selectbox("Solo defender", [x for x in defenders if x], key="h2h2v1_def")

        if atk and dfd:
            h2h_df_2v1 = df_2v1_game[
                (df_2v1_game["team1_players"] == atk) & (df_2v1_game["team2_players"] == dfd)
            ].copy()

            st.markdown(f"### {atk} (2-player side) vs {dfd} (solo)")

            if h2h_df_2v1.empty:
                st.info("No 2v1 matches yet for this exact duo vs solo combo.")
            else:
                h2h_df_2v1 = h2h_df_2v1.sort_values("date").reset_index(drop=True)

                wins_atk = (h2h_df_2v1["result1"] == "W").sum()
                wins_def = (h2h_df_2v1["result2"] == "W").sum()
                draws = (h2h_df_2v1["result1"] == "D").sum()

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(f"{atk} record", f"{wins_atk}W–{draws}D–{wins_def}L")
                with c2:
                    st.metric("Total matches", len(h2h_df_2v1))
                with c3:
                    st.metric("Avg goals for duo", f"{h2h_df_2v1['score1'].mean():.2f}")

                st.markdown("---")

                st.markdown("### Match history")
                cols_2v1 = ["date", "team1_name", "team1_players", "score1", "xG1", "team2_name", "team2_players", "score2", "xG2"]
                mh_2v1 = h2h_df_2v1[cols_2v1].copy()
                mh_2v1["date"] = pd.to_datetime(mh_2v1["date"]).dt.strftime("%Y-%m-%d")
                st.dataframe(mh_2v1, use_container_width=True)
