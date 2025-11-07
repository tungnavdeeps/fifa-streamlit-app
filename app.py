import datetime
import pandas as pd
import streamlit as st
import gspread
import matplotlib.pyplot as plt

# =========================
# CONFIG – EDIT THESE
# =========================
SPREADSHEET_ID = "1-82tJW2-y5mkt0b0qn4DPWj5sL-yOjKgCBKizUSzs9I"  # 👈 REPLACE THIS

WORKSHEET_1V1 = "Matches_1v1"
WORKSHEET_2V2 = "Matches_2v2"

# You can change or add versions here
GAME_OPTIONS = ["FIFA 24", "FIFA 25", "FIFA 26"]


# =========================
# GOOGLE SHEETS HELPERS (UPDATED FOR SECRETS)
# =========================
@st.cache_data(ttl=60)
def get_gsheet_client():
    client = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
    return client


def load_sheet(worksheet_name: str) -> pd.DataFrame:
    client = get_gsheet_client()
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(worksheet_name)
    records = sheet.get_all_records()
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_matches_1v1() -> pd.DataFrame:
    """
    1v1 sheet columns:
    date, game, player1, team1, score1, result1, player2, team2, score2, result2
    """
    df = load_sheet(WORKSHEET_1V1)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "game",
                "player1",
                "team1",
                "score1",
                "result1",
                "player2",
                "team2",
                "score2",
                "result2",
            ]
        )

    for col in [
        "date",
        "game",
        "player1",
        "team1",
        "score1",
        "result1",
        "player2",
        "team2",
        "score2",
        "result2",
    ]:
        if col not in df.columns:
            df[col] = None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score1"] = pd.to_numeric(df["score1"], errors="coerce")
    df["score2"] = pd.to_numeric(df["score2"], errors="coerce")

    return df[
        [
            "date",
            "game",
            "player1",
            "team1",
            "score1",
            "result1",
            "player2",
            "team2",
            "score2",
            "result2",
        ]
    ]


def load_matches_2v2() -> pd.DataFrame:
    """
    2v2 sheet columns:
    date, game, team1_name, team1_players, score1, result1,
          team2_name, team2_players, score2, result2
    """
    df = load_sheet(WORKSHEET_2V2)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "game",
                "team1_name",
                "team1_players",
                "score1",
                "result1",
                "team2_name",
                "team2_players",
                "score2",
                "result2",
            ]
        )

    for col in [
        "date",
        "game",
        "team1_name",
        "team1_players",
        "score1",
        "result1",
        "team2_name",
        "team2_players",
        "score2",
        "result2",
    ]:
        if col not in df.columns:
            df[col] = None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score1"] = pd.to_numeric(df["score1"], errors="coerce")
    df["score2"] = pd.to_numeric(df["score2"], errors="coerce")

    return df[
        [
            "date",
            "game",
            "team1_name",
            "team1_players",
            "score1",
            "result1",
            "team2_name",
            "team2_players",
            "score2",
            "result2",
        ]
    ]


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
        str(date),
        game,
        player1,
        team1,
        int(score1),
        result1,
        player2,
        team2,
        int(score2),
        result2,
    ]
    sheet.append_row(row)


def append_match_2v2(
    date, game, team1_name, team1_players, score1, team2_name, team2_players, score2
):
    client = get_gsheet_client()
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_2V2)

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
        result1,
        team2_name,
        team2_players,
        int(score2),
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


def predict_match_1v1(df: pd.DataFrame, game: str, player_a: str, player_b: str):
    ratings = compute_ratings_1v1(df, game)
    ra = ratings.get(player_a, 1000)
    rb = ratings.get(player_b, 1000)
    prob_a = expected_score(ra, rb)
    return ra, rb, prob_a


def compute_goals_vs_opponent(h2h_df: pd.DataFrame, player: str):
    """Total games, goals for, goals against for one player vs a specific opponent."""
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
    games, goals_for, goals_against, averages.
    """
    rows = []

    for _, row in h2h_df.iterrows():
        if row["player1"] == player:
            team = row.get("team1")
            gf = row.get("score1")
            ga = row.get("score2")
        elif row["player2"] == player:
            team = row.get("team2")
            gf = row.get("score2")
            ga = row.get("score1")
        else:
            continue

        rows.append(
            {"team": team, "goals_for": gf, "goals_against": ga}
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "team",
                "games",
                "goals_for",
                "goals_against",
                "avg_goals_for",
                "avg_goals_against",
            ]
        )

    df = pd.DataFrame(rows)

    grouped = df.groupby("team", dropna=True).agg(
        games=("team", "count"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )

    grouped["avg_goals_for"] = grouped["goals_for"] / grouped["games"]
    grouped["avg_goals_against"] = grouped["goals_against"] / grouped["games"]

    grouped = grouped.sort_values(
        by="goals_for", ascending=False
    ).reset_index()

    return grouped


# =========================
# UTILS FOR INPUT UI
# =========================
def player_input_block(label, existing_players, key_prefix):
    """
    Helper: dropdown of existing players + optional "new player" text box.
    Returns the chosen player name (or empty string).
    """
    options = ["-- Select existing --"] + sorted(existing_players)
    selected = st.selectbox(
        f"{label} (existing)", options, key=f"{key_prefix}_select"
    )
    new_name = st.text_input(
        f"{label} (new, if not in list)", key=f"{key_prefix}_new"
    ).strip()

    if new_name:
        return new_name
    if selected != "-- Select existing --":
        return selected
    return ""


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="FIFA Squad Tracker", layout="wide")
st.title("🎮 FIFA Squad Tracker & Predictor")

# Game filter in sidebar
st.sidebar.markdown("### ⚙️ Settings")
selected_game = st.sidebar.selectbox("Game version", GAME_OPTIONS)
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Record Match", "Head-to-Head (1v1)", "All Data"],
)

# Load all data once
df_1v1 = load_matches_1v1()
df_2v2 = load_matches_2v2()

df_1v1_game = df_1v1[df_1v1["game"] == selected_game].copy()
df_2v2_game = df_2v2[df_2v2["game"] == selected_game].copy()

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

    # Leaderboards side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 1v1 Player Leaderboard")
        if leaderboard_players.empty:
            st.info(f"No 1v1 matches yet for {selected_game}.")
        else:
            display_cols = [
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

            st.markdown("**ELO Ratings (1v1)**")
            fig, ax = plt.subplots()
            ax.bar(leaderboard_players["player"], leaderboard_players["elo_rating"])
            ax.set_xlabel("Player")
            ax.set_ylabel("ELO Rating")
            ax.set_title(f"ELO – {selected_game}")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

    with col2:
        st.markdown("### 👥 2v2 Team Leaderboard")
        if leaderboard_teams.empty:
            st.info(f"No 2v2 matches yet for {selected_game}.")
        else:
            display_cols_t = [
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

            st.markdown("**Goals Scored by Team (2v2)**")
            fig2, ax2 = plt.subplots()
            ax2.bar(leaderboard_teams["team"], leaderboard_teams["goals_for"])
            ax2.set_xlabel("Team")
            ax2.set_ylabel("Goals For")
            ax2.set_title(f"2v2 Goals – {selected_game}")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig2)

    st.markdown("---")

    # ---------- TITLES / AWARDS FOR 1v1 AND 2v2 ----------
    st.markdown("### 🏆 1v1 Player Awards & Titles")

    if leaderboard_players.empty:
        st.info("No 1v1 matches yet to calculate player titles.")
    else:
        awards_1v1 = {}

        # Golden Boot – most total goals scored
        golden_boot = leaderboard_players.sort_values("goals_for", ascending=False).iloc[0]
        awards_1v1["Golden Boot (Top Scorer)"] = (
            f"**{golden_boot['player']}** – {int(golden_boot['goals_for'])} goals"
        )

        # Eligible players for 'serious' awards (at least 5 games)
        eligible = leaderboard_players[leaderboard_players["games"] >= 5].copy()

        # Brick Wall – lowest avg goals conceded
        if not eligible.empty:
            best_def = eligible.sort_values("avg_goals_against").iloc[0]
            awards_1v1["Brick Wall (Best Defense)"] = (
                f"**{best_def['player']}** – {best_def['avg_goals_against']:.2f} avg conceded"
            )

            # Consistent Winner – highest win %
            best_wr = eligible.sort_values("win_pct", ascending=False).iloc[0]
            awards_1v1["Consistent Winner (Highest Win %)"] = (
                f"**{best_wr['player']}** – {best_wr['win_pct']:.1%} wins"
            )

        # Most Feared Rival – highest avg goals per game
        top_attacker = leaderboard_players.sort_values("avg_goals_for", ascending=False).iloc[0]
        awards_1v1["Most Feared Rival (Attack Threat)"] = (
            f"**{top_attacker['player']}** – {top_attacker['avg_goals_for']:.2f} avg goals per game"
        )

        # Clean Sheet King – most games where opponent scored 0
        clean_sheets = {}
        for _, row in df_1v1_game.iterrows():
            # player1 kept a clean sheet
            if row["score2"] == 0:
                clean_sheets[row["player1"]] = clean_sheets.get(row["player1"], 0) + 1
            # player2 kept a clean sheet
            if row["score1"] == 0:
                clean_sheets[row["player2"]] = clean_sheets.get(row["player2"], 0) + 1

        if clean_sheets:
            cs_player = max(clean_sheets, key=clean_sheets.get)
            awards_1v1["Clean Sheet King"] = (
                f"**{cs_player}** – {clean_sheets[cs_player]} matches with 0 goals conceded"
            )

        # Display 1v1 awards
        for title, text in awards_1v1.items():
            st.markdown(f"🏅 **{title}:** {text}")

    st.markdown("### 🏆 2v2 Team Awards & Titles")

    if leaderboard_teams.empty:
        st.info("No 2v2 matches yet to calculate team titles.")
    else:
        awards_2v2 = {}

        # Golden Duo – highest win % (min 3 games)
        eligible_teams = leaderboard_teams[leaderboard_teams["games"] >= 3].copy()
        if not eligible_teams.empty:
            best_team = eligible_teams.sort_values("win_pct", ascending=False).iloc[0]
            awards_2v2["Golden Duo (Best 2v2 Team)"] = (
                f"**{best_team['team']}** – {best_team['win_pct']:.1%} win rate "
                f"({int(best_team['games'])} games)"
            )

        # Attacking Duo – highest avg goals_for per game
        top_attack_team = leaderboard_teams.sort_values("avg_goals_for", ascending=False).iloc[0]
        awards_2v2["Attacking Duo (Most Goals per Game)"] = (
            f"**{top_attack_team['team']}** – {top_attack_team['avg_goals_for']:.2f} avg goals scored"
        )

        # Fortress Duo – lowest avg goals_against per game (min 3 games)
        if not eligible_teams.empty:
            best_def_team = eligible_teams.sort_values("avg_goals_against").iloc[0]
            awards_2v2["Fortress Duo (Best Defense)"] = (
                f"**{best_def_team['team']}** – {best_def_team['avg_goals_against']:.2f} avg conceded"
            )

        # Clean Sheet Duo – most matches with 0 goals conceded
        clean_sheets_team = {}
        for _, row in df_2v2_game.iterrows():
            if row["score2"] == 0:
                clean_sheets_team[row["team1_name"]] = clean_sheets_team.get(row["team1_name"], 0) + 1
            if row["score1"] == 0:
                clean_sheets_team[row["team2_name"]] = clean_sheets_team.get(row["team2_name"], 0) + 1

        if clean_sheets_team:
            cs_team = max(clean_sheets_team, key=clean_sheets_team.get)
            awards_2v2["Clean Sheet Duo"] = (
                f"**{cs_team}** – {clean_sheets_team[cs_team]} clean sheets"
            )

        # Display 2v2 awards
        for title, text in awards_2v2.items():
            st.markdown(f"🏅 **{title}:** {text}")

    st.markdown("---")

    # Quick prediction widget
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

    match_type = st.radio(
        "Match type", ["1v1", "2v2"], horizontal=True, key="match_type_radio"
    )
    date = st.date_input("Match date", value=datetime.date.today())
    game_for_entry = st.selectbox(
        "Game", GAME_OPTIONS, index=GAME_OPTIONS.index(selected_game)
    )

    st.markdown("#### Enter details")

    if match_type == "1v1":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Player 1**")
            p1 = player_input_block(
                "Player 1", players_all, key_prefix="p1_input"
            )
            team1 = st.text_input("Team 1 (optional)", key="team1_input").strip()
            score1 = st.number_input(
                "Goals scored by Player 1",
                min_value=0,
                step=1,
                key="score1_1v1",
            )

        with col2:
            st.markdown("**Player 2**")
            p2 = player_input_block(
                "Player 2", players_all, key_prefix="p2_input"
            )
            team2 = st.text_input("Team 2 (optional)", key="team2_input").strip()
            score2 = st.number_input(
                "Goals scored by Player 2",
                min_value=0,
                step=1,
                key="score2_1v1",
            )

        if st.button("Save 1v1 match", use_container_width=True):
            if not p1 or not p2:
                st.error("Please fill in both player names (either existing or new).")
            elif p1 == p2:
                st.error("Players must be different.")
            else:
                append_match_1v1(
                    date, game_for_entry, p1, team1, score1, p2, team2, score2
                )
                st.success(f"Saved 1v1 match for {game_for_entry}! 🎉")
                st.cache_data.clear()

    else:
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
                    team2_name,
                    team2_players,
                    score2_2v2,
                )
                st.success(f"Saved 2v2 match for {game_for_entry}! 🎉")
                st.cache_data.clear()


# ---------- PAGE: HEAD-TO-HEAD (1v1) ----------
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
                wins_p1 = (
                    ((h2h_df["player1"] == p1) & (h2h_df["score1"] > h2h_df["score2"])).sum()
                    + ((h2h_df["player2"] == p1) & (h2h_df["score2"] > h2h_df["score1"])).sum()
                )
                wins_p2 = (
                    ((h2h_df["player1"] == p2) & (h2h_df["score1"] > h2h_df["score2"])).sum()
                    + ((h2h_df["player2"] == p2) & (h2h_df["score2"] > h2h_df["score1"])).sum()
                )
                draws = (h2h_df["score1"] == h2h_df["score2"]).sum()

                colL, colM, colR = st.columns(3)
                colL.metric(f"{p1} wins", wins_p1)
                colM.metric("Draws", draws)
                colR.metric(f"{p2} wins", wins_p2)

                # Goals vs opponent
                games_p1, gf_p1, ga_p1 = compute_goals_vs_opponent(h2h_df, p1)
                games_p2, gf_p2, ga_p2 = compute_goals_vs_opponent(h2h_df, p2)

                if games_p1 > 0 and games_p2 > 0:
                    st.markdown("#### Goals in this matchup")
                    colG1, colG2 = st.columns(2)
                    with colG1:
                        st.write(
                            f"**{p1} vs {p2}:** {gf_p1} goals scored, {ga_p1} conceded "
                            f"in {games_p1} games "
                            f"(_avg {gf_p1/games_p1:.2f} scored, {ga_p1/games_p1:.2f} conceded_)."
                        )
                    with colG2:
                        st.write(
                            f"**{p2} vs {p1}:** {gf_p2} goals scored, {ga_p2} conceded "
                            f"in {games_p2} games "
                            f"(_avg {gf_p2/games_p2:.2f} scored, {ga_p2/games_p2:.2f} conceded_)."
                        )

                # Team breakdown tables
                st.markdown("#### Teams used in this matchup")

                team_stats_p1 = summarize_team_stats_vs_opponent(h2h_df, p1)
                team_stats_p2 = summarize_team_stats_vs_opponent(h2h_df, p2)

                colT1, colT2 = st.columns(2)
                with colT1:
                    st.markdown(f"**{p1} teams vs {p2}**")
                    if team_stats_p1.empty:
                        st.write("No team data recorded.")
                    else:
                        st.dataframe(
                            team_stats_p1[
                                [
                                    "team",
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
                    st.markdown(f"**{p2} teams vs {p1}**")
                    if team_stats_p2.empty:
                        st.write("No team data recorded.")
                    else:
                        st.dataframe(
                            team_stats_p2[
                                [
                                    "team",
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

                st.markdown("#### Match history")
                st.dataframe(
                    h2h_df[
                        [
                            "date",
                            "player1",
                            "team1",
                            "score1",
                            "score2",
                            "team2",
                            "player2",
                        ]
                    ].sort_values(by="date", ascending=False),
                    use_container_width=True,
                )

            st.markdown("---")
            st.markdown("### Win Prediction (for next 1v1)")

            ra, rb, prob_a = predict_match_1v1(
                df_1v1, selected_game, p1, p2
            )
            st.write("ELO rating (for this game only):")
            st.write(f"- {p1}: **{round(ra)}**")
            st.write(f"- {p2}: **{round(rb)}**")

            st.write("Estimated win chance next match:")
            st.write(f"- **{p1}: {prob_a:.1%}**")
            st.write(f"- **{p2}: {(1 - prob_a):.1%}**")

            if prob_a > 0.6:
                st.success(f"Favouring **{p1}** right now. Time for {p2} to prove the stats wrong.")
            elif prob_a < 0.4:
                st.success(f"Favouring **{p2}** right now. {p1}, you’re the underdog here.")
            else:
                st.info("This one’s tight. Either player could take it.")

            st.markdown("---")
            st.markdown("### Optional: team-specific performance")

            colOpt1, colOpt2 = st.columns(2)
            with colOpt1:
                p1_team = st.text_input(
                    f"Analyze {p1} using team (optional)", key="p1_team_analysis"
                ).strip()
            with colOpt2:
                p2_team = st.text_input(
                    f"Analyze {p2} using team (optional)", key="p2_team_analysis"
                ).strip()

            # Player 1 with a specific team
            if p1_team:
                mask_p1_team = (
                    ((h2h_df["player1"] == p1) & (h2h_df["team1"] == p1_team))
                    | ((h2h_df["player2"] == p1) & (h2h_df["team2"] == p1_team))
                )
                df_p1_team = h2h_df[mask_p1_team].copy()

                st.markdown(f"**{p1} using {p1_team} vs {p2}**")
                if df_p1_team.empty:
                    st.write(f"No matches found where {p1} used {p1_team} vs {p2}.")
                else:
                    games_t, gf_t, ga_t = compute_goals_vs_opponent(df_p1_team, p1)
                    avg_gf_t = gf_t / games_t
                    avg_ga_t = ga_t / games_t

                    if games_p1 > 0:
                        avg_gf_all = gf_p1 / games_p1
                        avg_ga_all = ga_p1 / games_p1
                    else:
                        avg_gf_all = avg_ga_all = 0

                    st.write(
                        f"- Games: {games_t}\n"
                        f"- Avg goals scored: **{avg_gf_t:.2f}** "
                        f"(overall vs {p2}: {avg_gf_all:.2f})\n"
                        f"- Avg goals conceded: **{avg_ga_t:.2f}** "
                        f"(overall vs {p2}: {avg_ga_all:.2f})"
                    )

                    verdict = []
                    if avg_gf_t > avg_gf_all:
                        verdict.append("scores **more** than usual")
                    elif avg_gf_t < avg_gf_all:
                        verdict.append("scores **less** than usual")

                    if avg_ga_t < avg_ga_all:
                        verdict.append("concedes **fewer** goals")
                    elif avg_ga_t > avg_ga_all:
                        verdict.append("concedes **more** goals")

                    if verdict:
                        st.success(
                            f"When {p1} uses **{p1_team}**, they " + " and ".join(verdict) + f" vs {p2}."
                        )

            # Player 2 with a specific team
            if p2_team:
                mask_p2_team = (
                    ((h2h_df["player1"] == p2) & (h2h_df["team1"] == p2_team))
                    | ((h2h_df["player2"] == p2) & (h2h_df["team2"] == p2_team))
                )
                df_p2_team = h2h_df[mask_p2_team].copy()

                st.markdown(f"**{p2} using {p2_team} vs {p1}**")
                if df_p2_team.empty:
                    st.write(f"No matches found where {p2} used {p2_team} vs {p1}.")
                else:
                    games_t2, gf_t2, ga_t2 = compute_goals_vs_opponent(df_p2_team, p2)
                    avg_gf_t2 = gf_t2 / games_t2
                    avg_ga_t2 = ga_t2 / games_t2

                    if games_p2 > 0:
                        avg_gf_all2 = gf_p2 / games_p2
                        avg_ga_all2 = ga_p2 / games_p2
                    else:
                        avg_gf_all2 = avg_ga_all2 = 0

                    st.write(
                        f"- Games: {games_t2}\n"
                        f"- Avg goals scored: **{avg_gf_t2:.2f}** "
                        f"(overall vs {p1}: {avg_gf_all2:.2f})\n"
                        f"- Avg goals conceded: **{avg_ga_t2:.2f}** "
                        f"(overall vs {p1}: {avg_ga_all2:.2f})"
                    )

                    verdict2 = []
                    if avg_gf_t2 > avg_gf_all2:
                        verdict2.append("scores **more** than usual")
                    elif avg_gf_t2 < avg_gf_all2:
                        verdict2.append("scores **less** than usual")

                    if avg_ga_t2 < avg_ga_all2:
                        verdict2.append("concedes **fewer** goals")
                    elif avg_gf_t2 > avg_ga_all2:
                        verdict2.append("concedes **more** goals")

                    if verdict2:
                        st.success(
                            f"When {p2} uses **{p2_team}**, they " + " and ".join(verdict2) + f" vs {p1}."
                        )


# ---------- PAGE: ALL DATA ----------
elif page == "All Data":
    st.subheader(f"📄 All Data – {selected_game}")

    # 1v1
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
            mask_t = (filtered_2v2["team1_name"] == team_filter) | (filtered_2v2["team2_name"] == team_filter)
            filtered_2v2 = filtered_2v2[mask_t]

        st.dataframe(
            filtered_2v2.sort_values(by="date", ascending=False),
            use_container_width=True,
        )
    
