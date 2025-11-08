import datetime
import pandas as pd
import streamlit as st
import gspread
import matplotlib.pyplot as plt
import time # Needed for retry logic

# =========================
# CONFIG – EDIT THESE
# =========================
# Ensure this ID is correct for your Google Sheet
SPREADSHEET_ID = "1-82tJW2-y5mkt0b0qn4DPWj5sL-yOjKgCBKizUSzs9I" 
WORKSHEET_1V1 = "Matches_1v1"
WORKSHEET_2V2 = "Matches_2v2"

# You can change or add versions here
GAME_OPTIONS = ["FIFA 24", "FIFA 25", "FIFA 26"]


# =========================
# GOOGLE SHEETS HELPERS (FINAL WORKAROUND - SIMPLEST CODE)
# =========================
@st.cache_data(ttl=60)
def get_gsheet_client(_cache_buster=None):
    """
    Initializes and caches the gspread client using Streamlit Secrets.
    Uses the simplest code path to avoid parsing corruption issues.
    """
    SECRET_KEY = "gcp_service_account"
    
    if SECRET_KEY not in st.secrets:
        st.error(
            "🛑 **Secret Key Missing!** Please ensure you have configured the Streamlit Cloud secret named "
            f"`{SECRET_KEY}`."
        )
        st.stop()

    try:
        # Use the most direct method to load the service account
        client = gspread.service_account_from_dict(st.secrets[SECRET_KEY])
        return client
    except Exception as e:
        # Catch all final authentication failures from Google's side
        st.error(f"❌ **Authentication Failed!** Could not connect using the `{SECRET_KEY}` credentials.")
        st.warning("Action Required: **Delete and re-paste the entire secret block** in Streamlit Cloud to clear corruption.")
        st.stop()


def load_sheet(worksheet_name: str) -> pd.DataFrame:
    """
    Loads data from a specified worksheet with retry logic for intermittent connection issues.
    """
    # This is the function the NameError was referring to. It must be defined BEFORE load_matches_1v1.
    client = get_gsheet_client(_cache_buster=1) 
    
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
                st.warning(f"Connection attempt {attempt + 1} failed for {worksheet_name}. Retrying in 2 seconds...")
                time.sleep(2) 
            else:
                st.error(f"Failed to connect to Google Sheet after {MAX_RETRIES} attempts.")
                st.exception(e)
                raise e
    
    return pd.DataFrame() 


# =========================
# DATA LOADING AND CLEANING
# =========================

def load_matches_1v1() -> pd.DataFrame:
    """Loads, cleans, and prepares 1v1 data."""
    df = load_sheet(WORKSHEET_1V1)

    if df.empty:
        return df

    # Data Cleaning
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df.dropna(subset=['date', 'game', 'player1', 'player2', 'score1', 'score2'], inplace=True)
    df = df[df['game'].isin(GAME_OPTIONS)]
    
    # Ensure scores are integers
    df['score1'] = pd.to_numeric(df['score1'], errors='coerce').fillna(0).astype(int)
    df['score2'] = pd.to_numeric(df['score2'], errors='coerce').fillna(0).astype(int)
    
    # Determine the winner based on score
    df['winner'] = df.apply(
        lambda row: row['player1'] if row['score1'] > row['score2'] else 
                    (row['player2'] if row['score2'] > row['score1'] else 'Draw'),
        axis=1
    )
    
    return df.sort_values(by='date', ascending=True).reset_index(drop=True)


def load_matches_2v2() -> pd.DataFrame:
    """Loads, cleans, and prepares 2v2 data."""
    df = load_sheet(WORKSHEET_2V2)

    if df.empty:
        return df

    # Data Cleaning (similar to 1v1 but for 2v2 structure)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df.dropna(subset=['date', 'game', 'player1', 'player2', 'player3', 'player4', 'score1', 'score2'], inplace=True)
    df = df[df['game'].isin(GAME_OPTIONS)]
    
    # Create team names
    df['team1_name'] = df['player1'] + ' & ' + df['player2']
    df['team2_name'] = df['player3'] + ' & ' + df['player4']
    
    df['score1'] = pd.to_numeric(df['score1'], errors='coerce').fillna(0).astype(int)
    df['score2'] = pd.to_numeric(df['score2'], errors='coerce').fillna(0).astype(int)

    # Determine the winning team name
    df['winner_team'] = df.apply(
        lambda row: row['team1_name'] if row['score1'] > row['score2'] else 
                    (row['team2_name'] if row['score2'] > row['score1'] else 'Draw'),
        axis=1
    )
    
    return df.sort_values(by='date', ascending=True).reset_index(drop=True)


# =========================
# ELO RATING CALCULATION (STARTING POINT)
# =========================

INITIAL_ELO = 1000
K_FACTOR = 32

# Function to calculate expected score (Ea)
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10**((rating_b - rating_a) / 400))

# Function to calculate new ELO rating
def calculate_new_elo(current_elo, expected_score, actual_score, k_factor):
    return current_elo + k_factor * (actual_score - expected_score)

# Placeholder function to process all 1v1 matches and calculate ELO
def calculate_elo_1v1(df_1v1: pd.DataFrame) -> pd.DataFrame:
    if df_1v1.empty:
        return pd.DataFrame()
        
    # Placeholder logic for ELO calculation
    return pd.DataFrame({'Player': ['WIP'], 'ELO': [INITIAL_ELO]})
    
# Placeholder function for 2v2 ELO
def calculate_elo_2v2(df_2v2: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()


# =========================
# STREAMLIT UI CODE STARTS HERE
# =========================

st.set_page_config(
    page_title="FIFA Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ FIFA Match Tracker")

# Sidebar for controls
st.sidebar.header("Controls")

# Game selector
selected_game = st.sidebar.selectbox(
    "Select Game Version", 
    GAME_OPTIONS, 
    index=0
)

# Cache clearing button
if st.sidebar.button("Clear Streamlit Data Cache"):
    st.cache_data.clear()
    st.rerun()

# Load and filter data by selected game
df_1v1_raw = load_matches_1v1()
df_2v2_raw = load_matches_2v2()

df_1v1_game = df_1v1_raw[df_1v1_raw['game'] == selected_game]
df_2v2_game = df_2v2_raw[df_2v2_raw['game'] == selected_game]


# --- Main Tabs ---
tab_dashboard, tab_add_match, tab_all_data = st.tabs(
    ["📊 Dashboard", "➕ Add Match", "📜 All Match Data"]
)

# ======================================================
# TAB 1: DASHBOARD (ELO and Stats will go here)
# ======================================================
with tab_dashboard:
    st.header(f"Dashboard for {selected_game}")
    
    st.markdown("#### ELO Ratings (WIP)")
    
    df_elo_1v1 = calculate_elo_1v1(df_1v1_game)
    
    if df_elo_1v1.empty:
        st.info("No ELO data yet (or ELO calculation logic is pending).")
    else:
        st.dataframe(df_elo_1v1)


# ======================================================
# TAB 2: ADD MATCH
# ======================================================
with tab_add_match:
    st.header("Record a New Match")
    st.info("Match submission form goes here.")


# ======================================================
# TAB 3: ALL MATCH DATA
# ======================================================
with tab_all_data:
    st.header("Full Match History")

    # 1v1
    st.markdown("#### 1v1 Matches")
    if df_1v1_game.empty:
        st.info(f"No 1v1 data yet for {selected_game}.")
    else:
        players_game = sorted(
            set(df_1v1_game["player1"].unique()).union(
                set(df_1v1_game["player2"].unique())
            )
        )
        
        col_pf1, _ = st.columns(2)
        with col_pf1:
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

    # 2v2
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
            mask = (filtered_2v2["team1_name"] == team_filter) | (filtered_2v2["team2_name"] == team_filter)
            filtered_2v2 = filtered_2v2[mask]

        st.dataframe(
            filtered_2v2.sort_values(by="date", ascending=False),
            use_container_width=True,
        )
