import datetime
import pandas as pd
import streamlit as st
import gspread
import time 
import matplotlib.pyplot as plt

# =========================
# CONFIG – EDIT THESE
# =========================
SPREADSHEET_ID = "1-82tJW2-y5mkt0b0qn4DPj5sL-yOjKgCBKizUSzs9I" 
WORKSHEET_1V1 = "Matches_1v1"
WORKSHEET_2V2 = "Matches_2v2"

GAME_OPTIONS = ["FIFA 24", "FIFA 25", "FIFA 26"]


# =========================
# GOOGLE SHEETS HELPERS (ULTRA-ROBUST KEY CLEANUP)
# =========================
@st.cache_resource(ttl=600)
def get_gsheet_client():
    """
    Initializes and caches the gspread client.
    Performs the most aggressive manual cleanup possible of the private_key string 
    to fix Streamlit's corruption and ensure valid Base64 decoding.
    """
    SECRET_KEY = "gcp_service_account"
    
    if SECRET_KEY not in st.secrets:
        st.error("🛑 **Secret Key Missing!** Please configure the secret named `gcp_service_account`.")
        st.stop()

    sa_info = dict(st.secrets[SECRET_KEY])
    
    if "private_key" in sa_info:
        raw_key = sa_info["private_key"]
        
        # 1. Fix escaped newlines that Streamlit adds
        private_key_fixed = raw_key.replace('\\n', '\n')
        
        begin = '-----BEGIN PRIVATE KEY-----'
        end = '-----END PRIVATE KEY-----'
        
        if begin in private_key_fixed and end in private_key_fixed:
            # Split the string to isolate the raw Base64 content
            content = private_key_fixed.split(begin)[1].split(end)[0]
            
            # Remove ALL whitespace and newlines from the Base64 content string
            clean_content = "".join(content.split())
            
            # Reconstruct the final key string exactly with expected markers and explicit newlines
            final_key = f"{begin}\n{clean_content}\n{end}" # Note: removed the trailing \n
            
            # Apply a final, global strip to remove any lingering outside whitespace
            sa_info["private_key"] = final_key.strip()
        else:
            # Fallback (shouldn't happen with the correct secret structure)
            sa_info["private_key"] = private_key_fixed.strip()
    
    try:
        # Use the cleaned dictionary
        client = gspread.service_account_from_dict(sa_info)
        return client
    except Exception as e:
        # Re-raising the error with a custom message to guide the user
        st.error(f"❌ **Authentication Failed!** Credentials rejected by Google.")
        st.warning(
            "Final Action Required: If this error persists, the *only* remaining issue is that the service account email is **not an Editor** on the Google Sheet."
        )
        st.exception(e)
        st.stop()
        

def load_sheet(worksheet_name: str) -> pd.DataFrame:
    """Loads data from a specified worksheet with retry logic."""
    client = get_gsheet_client() 
    
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            sheet = client.open_by_key(SPREADSHEET_ID).worksheet(worksheet_name)
            records = sheet.get_all_records()
            
            if not records:
                return pd.DataFrame() 
            return pd.DataFrame(records)
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2) 
            else:
                st.error(f"Failed to connect to Google Sheet after {MAX_RETRIES} attempts.")
                raise e
    
    return pd.DataFrame() 


# =========================
# DATA LOADING AND CLEANING
# =========================

def load_matches_1v1() -> pd.DataFrame:
    """Loads, cleans, and prepares 1v1 data."""
    df = load_sheet(WORKSHEET_1V1)
    if df.empty: return df
    df['date'] = pd.to_datetime(df.get('date', pd.Series()), errors='coerce')
    df['score1'] = pd.to_numeric(df.get('score1', pd.Series()), errors='coerce').fillna(0).astype(int)
    df['score2'] = pd.to_numeric(df.get('score2', pd.Series()), errors='coerce').fillna(0).astype(int)
    
    required_cols = ['date', 'game', 'player1', 'player2', 'score1', 'score2', 'team1', 'team2']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df.dropna(subset=['date', 'game', 'player1', 'player2'], inplace=True)
    df = df[df['game'].isin(GAME_OPTIONS)]
    
    df['winner'] = df.apply(
        lambda row: row['player1'] if row['score1'] > row['score2'] else 
                    (row['player2'] if row['score2'] > row['score1'] else 'Draw'),
        axis=1
    )
    return df.sort_values(by='date', ascending=True).reset_index(drop=True)


def load_matches_2v2() -> pd.DataFrame:
    """Loads, cleans, and prepares 2v2 data."""
    df = load_sheet(WORKSHEET_2V2)
    if df.empty: return df
    df['date'] = pd.to_datetime(df.get('date', pd.Series()), errors='coerce')
    df['score1'] = pd.to_numeric(df.get('score1', pd.Series()), errors='coerce').fillna(0).astype(int)
    df['score2'] = pd.to_numeric(df.get('score2', pd.Series()), errors='coerce').fillna(0).astype(int)

    required_cols = ['date', 'game', 'player1', 'player2', 'player3', 'player4', 'score1', 'score2', 'team1_name', 'team2_name']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df.dropna(subset=['date', 'game', 'score1', 'score2', 'team1_name', 'team2_name'], inplace=True)
    df = df[df['game'].isin(GAME_OPTIONS)]
    
    df['winner_team'] = df.apply(
        lambda row: row['team1_name'] if row['score1'] > row['score2'] else 
                    (row['team2_name'] if row['score2'] > row['score1'] else 'Draw'),
        axis=1
    )
    return df.sort_values(by='date', ascending=True).reset_index(drop=True)


# --- MATCH APPEND FUNCTIONS ---

def append_match_1v1(date, game, player1, team1, score1, player2, team2, score2):
    client = get_gsheet_client()
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_1V1)

    if score1 > score2:
        result1, result2 = "W", "L"
    elif score1 < score2:
        result1, result2 = "L", "W"
    else:
        result1 = result2 = "D"

    row = [
        str(date), game, player1, team1, int(score1), result1, 
        player2, team2, int(score2), result2,
    ]
    sheet.append_row(row, value_input_option='USER_ENTERED')


def append_match_2v2(date, game, team1_name, team1_players, score1, team2_name, team2_players, score2):
    client = get_gsheet_client()
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_2V2)

    if score1 > score2:
        result1, result2 = "W", "L"
    elif score1 < score2:
        result1, result2 = "L", "W"
    else:
        result1 = result2 = "D"

    row = [
        str(date), game, team1_name, team1_players, int(score1), result1, 
        team2_name, team2_players, int(score2), result2,
    ]
    sheet.append_row(row, value_input_option='USER_ENTERED')


# --- ELO RATING AND LEADERBOARD FUNCTIONS ---

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, score_a, k=20):
    exp_a = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b

def compute_ratings_1v1(df: pd.DataFrame, game: str, base_rating=1000):
    ratings = {}
    if df.empty: return ratings
    df_game = df[df["game"] == game].copy().sort_values(by="date", na_position="last")
    for _, row in df_game.iterrows():
        p1, p2 = row["player1"], row["player2"]
        s1, s2 = row["score1"], row["score2"]
        if pd.isna(p1) or pd.isna(p2) or pd.isna(s1) or pd.isna(s2): continue
        ratings.setdefault(p1, base_rating)
        ratings.setdefault(p2, base_rating)
        score_a = 1.0 if s1 > s2 else (0.0 if s1 < s2 else 0.5)
        new_p1, new_p2 = update_elo(ratings[p1], ratings[p2], score_a)
        ratings[p1], ratings[p2] = new_p1, new_p2
    return ratings

def build_player_leaderboard_1v1(df: pd.DataFrame, game: str) -> pd.DataFrame:
    df_game = df[df["game"] == game].copy()
    if df_game.empty: return pd.DataFrame()
    rows = []
    for _, row in df_game.iterrows():
        p1, p2 = row["player1"], row["player2"]
        s1, s2 = row["score1"], row["score2"]
        if s1 > s2: r1, r2 = "W", "L"
        elif s1 < s2: r1, r2 = "L", "W"
        else: r1 = r2 = "D"
        rows.append({"player": p1, "goals_for": s1, "goals_against": s2, "result": r1})
        rows.append({"player": p2, "goals_for": s2, "goals_against": s1, "result": r2})

    stats_df = pd.DataFrame(rows).groupby("player").agg(
        games=("result", "count"),
        wins=("result", lambda x: (x == "W").sum()),
        draws=("result", lambda x: (x == "D").sum()),
        losses=("result", lambda x: (x == "L").sum()),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )
    stats_df["goal_diff"] = stats_df["goals_for"] - stats_df["goals_against"]
    stats_df["win_pct"] = stats_df["wins"] / stats_df["games"]
    ratings = compute_ratings_1v1(df, game)
    stats_df["elo_rating"] = stats_df.index.map(lambda p: round(ratings.get(p, 1000)))
    return stats_df.sort_values(by=["elo_rating", "wins"], ascending=False).reset_index()


def build_team_leaderboard_2v2(df: pd.DataFrame, game: str) -> pd.DataFrame:
    df_game = df[df["game"] == game].copy()
    if df_game.empty: return pd.DataFrame()
    rows = []
    for _, row in df_game.iterrows():
        t1, t2 = row["team1_name"], row["team2_name"]
        s1, s2 = row["score1"], row["score2"]
        if s1 > s2: r1, r2 = "W", "L"
        elif s1 < s2: r1, r2 = "L", "W"
        else: r1 = r2 = "D"
        rows.append({"team": t1, "goals_for": s1, "goals_against": s2, "result": r1})
        rows.append({"team": t2, "goals_for": s2, "goals_against": s1, "result": r2})

    stats_df = pd.DataFrame(rows).groupby("team").agg(
        games=("result", "count"),
        wins=("result", lambda x: (x == "W").sum()),
        draws=("result", lambda x: (x == "D").sum()),
        losses=("result", lambda x: (x == "L").sum()),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )
    stats_df["goal_diff"] = stats_df["goals_for"] - stats_df["goals_against"]
    stats_df["win_pct"] = stats_df["wins"] / stats_df["games"]
    return stats_df.sort_values(by=["win_pct", "goal_diff"], ascending=False).reset_index()


def player_input_block(label, existing_players, key_prefix):
    options = ["-- Select existing --"] + sorted(existing_players)
    selected = st.selectbox(f"{label} (existing)", options, key=f"{key_prefix}_select")
    new_name = st.text_input(f"{label} (new, if not in list)", key=f"{key_prefix}_new").strip()
    if new_name: return new_name
    if selected != "-- Select existing --": return selected
    return ""


# =========================
# STREAMLIT UI 
# =========================
st.set_page_config(page_title="FIFA Squad Tracker", layout="wide")
st.title("🎮 FIFA Squad Tracker & Predictor")

# Sidebar
st.sidebar.markdown("### ⚙️ Settings")
selected_game = st.sidebar.selectbox("Game version", GAME_OPTIONS)
page = st.sidebar.radio("Go to", ["Dashboard", "Record Match", "All Data"])

if st.sidebar.button("Clear Streamlit Data Cache"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()


# Load Data (The authentication happens here)
try:
    df_1v1 = load_matches_1v1()
    df_2v2 = load_matches_2v2()
except Exception:
    # Stop execution if loading fails (handled inside load_sheet)
    st.stop()

df_1v1_game = df_1v1[df_1v1["game"] == selected_game].copy()
df_2v2_game = df_2v2[df_2v2["game"] == selected_game].copy()
players_all = sorted(set(df_1v1["player1"].dropna().unique()).union(set(df_1v1["player2"].dropna().unique())))
players_game = sorted(set(df_1v1_game["player1"].dropna().unique()).union(set(df_1v1_game["player2"].dropna().unique())))


# --- PAGE: DASHBOARD ---
if page == "Dashboard":
    st.subheader(f"🏠 Season Summary – {selected_game}")

    leaderboard_players = build_player_leaderboard_1v1(df_1v1, selected_game)
    leaderboard_teams = build_team_leaderboard_2v2(df_2v2, selected_game)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 1v1 Player Leaderboard")
        if leaderboard_players.empty:
            st.info(f"No 1v1 matches yet for {selected_game}.")
        else:
            display_cols = ["player", "games", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff", "win_pct", "elo_rating"]
            st.dataframe(
                leaderboard_players[display_cols].style.format({"win_pct": "{:.1%}", "elo_rating": "{:.0f}"}),
                use_container_width=True,
            )

    with col2:
        st.markdown("### 👥 2v2 Team Leaderboard")
        if leaderboard_teams.empty:
            st.info(f"No 2v2 matches yet for {selected_game}.")
        else:
            display_cols_t = ["team", "games", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff", "win_pct"]
            st.dataframe(
                leaderboard_teams[display_cols_t].style.format({"win_pct": "{:.1%}"}),
                use_container_width=True,
            )


# --- PAGE: RECORD MATCH ---
elif page == "Record Match":
    st.subheader(f"📝 Record a Match – {selected_game}")
    match_type = st.radio("Match type", ["1v1", "2v2"], horizontal=True, key="match_type_radio")
    date = st.date_input("Match date", value=datetime.date.today())
    game_for_entry = st.selectbox("Game", GAME_OPTIONS, index=GAME_OPTIONS.index(selected_game))

    st.markdown("#### Enter details")

    if match_type == "1v1":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Player 1**")
            p1 = player_input_block("Player 1", players_all, key_prefix="p1_input")
            team1 = st.text_input("Team 1 (optional)", key="team1_input").strip()
            score1 = st.number_input("Goals scored by Player 1", min_value=0, step=1, key="score1_1v1")
        with col2:
            st.markdown("**Player 2**")
            p2 = player_input_block("Player 2", players_all, key_prefix="p2_input")
            team2 = st.text_input("Team 2 (optional)", key="team2_input").strip()
            score2 = st.number_input("Goals scored by Player 2", min_value=0, step=1, key="score2_1v1")

        if st.button("Save 1v1 match", use_container_width=True):
            if not p1 or not p2: st.error("Please fill in both player names.")
            elif p1 == p2: st.error("Players must be different.")
            else:
                append_match_1v1(date, game_for_entry, p1, team1, score1, p2, team2, score2)
                st.success(f"Saved 1v1 match for {game_for_entry}! 🎉")
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()

    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Team 1**")
            team1_name = st.text_input("Team 1 name", key="team1_name").strip()
            team1_players = st.text_input("Team 1 players (e.g. Sue & Alex)", key="team1_players").strip()
            score1_2v2 = st.number_input("Goals scored by Team 1", min_value=0, step=1, key="score1_2v2")
        with col2:
            st.markdown("**Team 2**")
            team2_name = st.text_input("Team 2 name", key="team2_name").strip()
            team2_players = st.text_input("Team 2 players (e.g. Jordan & Max)", key="team2_players").strip()
            score2_2v2 = st.number_input("Goals scored by Team 2", min_value=0, step=1, key="score2_2v2")

        if st.button("Save 2v2 match", use_container_width=True):
            if not team1_name or not team2_name: st.error("Please fill in both team names.")
            elif team1_name == team2_name: st.error("Teams must be different.")
            else:
                append_match_2v2(date, game_for_entry, team1_name, team1_players, score1_2v2, team2_name, team2_players, score2_2v2)
                st.success(f"Saved 2v2 match for {game_for_entry}! 🎉")
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()


# --- PAGE: ALL DATA ---
elif page == "All Data":
    st.subheader(f"📄 All Data – {selected_game}")
    st.markdown("#### 1v1 Matches")
    if df_1v1_game.empty:
        st.info(f"No 1v1 data yet for {selected_game}.")
    else:
        player_filter = st.selectbox("Filter 1v1 by player (optional)", ["(All players)"] + players_game, key="all_data_player_filter")
        filtered_1v1 = df_1v1_game.copy()
        if player_filter != "(All players)":
            mask = (filtered_1v1["player1"] == player_filter) | (filtered_1v1["player2"] == player_filter)
            filtered_1v1 = filtered_1v1[mask]
        st.dataframe(filtered_1v1.sort_values(by="date", ascending=False), use_container_width=True)

    st.markdown("#### 2v2 Matches")
    if df_2v2_game.empty:
        st.info(f"No 2v2 data yet for {selected_game}.")
    else:
        teams_in_game = sorted(set(df_2v2_game["team1_name"].dropna().unique()).union(set(df_2v2_game["team2_name"].dropna().unique())))
        team_filter = st.selectbox("Filter 2v2 by team (optional)", ["(All teams)"] + teams_in_game, key="all_data_team_filter")
        filtered_2v2 = df_2v2_game.copy()
        if team_filter != "(All teams)":
            mask_t = (filtered_2v2["team1_name"] == team_filter) | (filtered_2v2["team2_name"] == team_filter)
            filtered_2v2 = filtered_2v2[mask_t]
        st.dataframe(filtered_2v2.sort_values(by="date", ascending=False), use_container_width=True)
