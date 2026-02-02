import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import base64

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aegis for Injury",
    page_icon="logo.png", #🛡️
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- 1. GLOBAL SETTINGS ---
SETTINGS_FILE = 'app_config.json'

def load_settings():
    default = {
        "manager_can_simulate": True,
        "manager_can_see_raw_data": True,
        "player_can_see_Risk": True,
        "player_can_see_detailed_metrics": True
    }
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return default

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)

app_settings = load_settings()

# --- 2. ADVANCED ML LOGIC (Gradient Boosting) ---

# Helper for Exponential Weighted Moving Average
def get_ewma(series, span):
    return series.ewm(span=span, adjust=False).mean()

@st.cache_resource
def train_and_process(df_raw):
    # 1. CLEANING & PREPROCESSING
    df = df_raw.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['player_name', 'date'])
    
    # Fill specific columns logic from notebook
    cols_to_fix = ['heart_rate_mean', 'heart_rate_max', 'speed_mean', 'daily_load', 'fatigue', 'soreness']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            # Fill with player median, then global median, then 0
            df[col] = df[col].fillna(df.groupby('player_name')[col].transform('median'))
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
            df[col] = df[col].fillna(0)

    # 2. FEATURE ENGINEERING
    grouped = df.groupby('player_name')['daily_load']
    
    # EWMA & ACWR
    df['atl_ewma'] = grouped.transform(lambda x: get_ewma(x, 7))
    df['ctl_ewma'] = grouped.transform(lambda x: get_ewma(x, 28))
    df['acwr_ewma'] = df['atl_ewma'] / (df['ctl_ewma'] + 1)
    
    # Monotony & Strain
    roll_mean = grouped.transform(lambda x: x.rolling(7).mean())
    roll_std = grouped.transform(lambda x: x.rolling(7).std())
    df['monotony_calc'] = roll_mean / (roll_std + 0.1)
    df['strain_calc'] = df['daily_load'] * df['monotony_calc']
    
    # Daily Jump
    df['daily_jump'] = df['daily_load'] / (grouped.shift(1) + 1)

    # Lagging Features (Context from yesterday)
    features_to_lag = ['daily_load', 'fatigue', 'soreness', 'acwr_ewma', 'monotony_calc', 'strain_calc', 'daily_jump']
    for col in features_to_lag:
        df[f'{col}_lag1'] = df.groupby('player_name')[col].shift(1)

    # Drop early rows where lags are NaN (so model doesn't train on empty history)
    df_clean = df.dropna(subset=['acwr_ewma_lag1']).copy()

    # 3. TRAINING
    if 'target' not in df_clean.columns:
        # Fallback if target isn't present, we just return processed data
        return None, 0.5, df_clean
    
    cols_exclude = ['player_name', 'date', 'target', 'has_injury', 'has_illness', 
                    'timestamp_x', 'timestamp_y', 'problems_x', 'problems_y', 
                    'type_x', 'type_y', 'target_x', 'target_y', 'year']
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in cols_exclude]

    X = df_clean[features]
    y = df_clean['target']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Weights: Safety First (15x weight for injuries)
    sample_weight = np.where(y_train == 1, 15, 1)

    model = HistGradientBoostingClassifier(
        learning_rate=0.03, 
        max_depth=5, 
        l2_regularization=0.5, 
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # 4. OPTIMIZE THRESHOLD (F4 Score)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_proba_test)
    
    beta = 4
    numerator = (1 + beta**2) * (precision_curve * recall_curve)
    denominator = (beta**2 * precision_curve) + recall_curve + 1e-10
    f_beta_scores = numerator / denominator
    
    # Find best threshold
    best_thresh = thresholds[np.argmax(f_beta_scores)]

    # 5. PREDICT ON FULL DATASET
    # We apply the model back to the full dataset to get risk scores for everyone
    full_probs = model.predict_proba(df_clean[features])[:, 1]
    df_clean['injury_probability'] = full_probs

    return model, best_thresh, df_clean

# --- 3. DATA LOADING ---
@st.cache_data
def load_dataset_final_v17():
    file_path = "outputC.csv"
    if not os.path.exists(file_path):
        for root, dirs, files in os.walk("."):
            if "outputC.csv" in files:
                file_path = os.path.join(root, "outputC.csv")
                break
    if not os.path.exists(file_path):
        return None, None, None, None

    try:
        df_raw = pd.read_csv(file_path)
    except:
        return None, None, None, None

    if 'player_name' not in df_raw.columns:
        return None, None, None, None
    
    # Run the Pipeline
    model, threshold, df_processed = train_and_process(df_raw)
    
    # Filter Team A
    df_team_a = df_processed[df_processed['player_name'].str.startswith("TeamA", na=False)].copy()
    if df_team_a.empty:
        return pd.DataFrame(), pd.DataFrame(), None, None

    # Anonymize Names
    unique_ids = df_team_a['player_name'].unique()
    id_map = {real_id: f"Player {i+1}" for i, real_id in enumerate(unique_ids)}
    df_team_a['Name'] = df_team_a['player_name'].map(id_map)
    
    # Create "Risk" column (0-100 scale)
    if 'injury_probability' in df_team_a.columns:
        df_team_a['Risk'] = df_team_a['injury_probability'] * 100
    else:
        df_team_a['Risk'] = 0.0

    df_latest = df_team_a.groupby('Name').tail(1).reset_index(drop=True)
    
    return df_latest, df_team_a, model, threshold


if df_latest is None:
    st.error("❌ 'outputC.csv' not found.")
    st.stop()

# --- 4. AUTHENTICATION ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.user_id = None
if 'selected_player_view' not in st.session_state:
    st.session_state.selected_player_view = None

def login_logic(username, password):
    if username == "admin" and password == "password123":
        st.session_state.role = "Admin"
        st.session_state.user_id = "Admin"
        st.session_state.logged_in = True
    elif username == "manager" and password == "manager123":
        st.session_state.role = "Manager"
        st.session_state.user_id = "Manager"
        st.session_state.logged_in = True
    elif username in df_latest['Name'].values:
        if password == "1234":
            st.session_state.role = "Player"
            st.session_state.user_id = username
            st.session_state.logged_in = True
        else: st.error("Incorrect password.")
    else: st.error("User not found.")
    if st.session_state.logged_in: st.rerun()

def logout():
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.selected_player_view = None
    st.rerun()

# --- 5. VISUALIZATIONS ---
def render_radar_chart(row):
    categories = ['Mood', 'Sleep Qual.', 'Readiness', 'Low Fatigue', 'Low Stress']
    # Note: Using .get() with defaults to avoid errors if columns missing
    # Normalizing 1-5 scales to 0-100 for Radar
    
    # 1. Mood (1-5) -> *20
    val_mood = row.get('mood', 3) * 20 
    # 2. Sleep Quality (1-5) -> *20
    val_sleep = row.get('sleep_quality', 3) * 20
    # 3. Readiness (1-10 usually, or 1-5? Assuming 1-10 based on common data)
    # If readiness is 1-5:
    val_readiness = row.get('readiness', 5) * 20 
    
    # 4. Low Fatigue (1 is good, 5 is bad. Invert.)
    # (6 - 5) * 20 = 20 (Bad) | (6 - 1) * 20 = 100 (Good)
    val_fatigue = (6 - row.get('fatigue', 3)) * 20 
    
    # 5. Low Soreness/Stress
    val_soreness = (6 - row.get('soreness', 3)) * 20

    values = [max(0, min(v, 100)) for v in [val_mood, val_sleep, val_readiness, val_fatigue, val_soreness]]
    values = [0 if pd.isna(x) else x for x in values]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Wellness', line_color='#2ECC71'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=320, margin=dict(t=20, b=20, l=40, r=40))
    return fig

def render_history_chart(player_history, threshold_percent):
    # Dynamic Scale
    max_risk = player_history['Risk'].max()
    y_max = min(100, max_risk * 1.2 if max_risk > 0 else 10)
    
    fig = px.line(player_history, x='date', y='Risk', title="Injury Risk Trend", labels={'date': 'Date', 'Risk': 'Risk (%)'}, markers=True)
    fig.update_traces(line_color='#e74c3c', line_width=2)
    
    # Add the Calculated Threshold Line
    fig.add_hline(y=threshold_percent, line_dash="dash", line_color="orange", annotation_text=f"Threshold ({threshold_percent:.1f}%)")
    
    fig.update_yaxes(range=[0, 100]) # Keep 0-100 for context
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(buttons=list([
            dict(count=7, label="1W", step="day", stepmode="backward"),
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(step="all", label="All")
        ]))
    )
    return fig

def render_load_fatigue_chart(player_history):
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar for Load
    fig.add_trace(go.Bar(
        x=player_history['date'], 
        y=player_history['daily_load'], 
        name="Daily Load", 
        marker_color='#3498db', 
        opacity=0.5
    ), secondary_y=False)
    
    # Line for Fatigue
    fig.add_trace(go.Scatter(
        x=player_history['date'], 
        y=player_history['fatigue'], 
        name="Fatigue", 
        line=dict(color='#f1c40f', width=2)
    ), secondary_y=True)
    
    fig.update_layout(title_text="Workload vs. Fatigue")
    fig.update_yaxes(title_text="Load", secondary_y=False)
    # Fatigue strictly 1-5
    fig.update_yaxes(title_text="Fatigue (1-5)", secondary_y=True, range=[1, 5])
    
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(buttons=list([
            dict(count=7, label="1W", step="day", stepmode="backward"),
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(step="all", label="All")
        ]))
    )
    return fig

# --- 6. VIEW: LOGIN ---
if not st.session_state.logged_in:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        # --- LOGO & TITLE ALIGNMENT ---
        # 1. Helper function to read image as Base64 string
        def get_base64_image(image_path):
            try:
                with open(image_path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode()
            except FileNotFoundError:
                return None

        # 2. Get the encoded string
        img_b64 = get_base64_image("logo.png")

        # 3. Render HTML with Flexbox (Centers everything)
        if img_b64:
            st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: center;">
                    <img src="data:image/png;base64,{img_b64}" style="width: 150px; margin-right: 15px;">
                    <h1 style="margin: 0; padding: 0;">Aegis for Injury</h1>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Fallback if logo.png is missing
            st.title("Aegis for Injury")
        
        # Add a little spacer
        st.write("")
        
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log In"):
                login_logic(u, p)

# --- 7. VIEW: MAIN APP ---
else:
    with st.sidebar:
        st.image("logo.png", width=150)
        st.title("Team A")
        st.write(f"User: **{st.session_state.user_id}**")
        if st.session_state.role == "Manager" and st.session_state.selected_player_view:
            if st.button("⬅️ Back to Squad"):
                st.session_state.selected_player_view = None
                st.rerun()
        st.divider()
        if st.button("Log Out"): logout()

    # ADMIN
    if st.session_state.role == "Admin":
        st.title("Config")
        m_sim = st.checkbox("Enable Simulation", app_settings['manager_can_simulate'])
        m_raw = st.checkbox("Show Raw Data", app_settings['manager_can_see_raw_data'])
        if st.button("Save"):
            save_settings({"manager_can_simulate": m_sim, "manager_can_see_raw_data": m_raw, "player_can_see_Risk": True, "player_can_see_detailed_metrics": True})
            st.success("Saved.")

    # MANAGER
    elif st.session_state.role == "Manager":
        
        # --- GLOBAL DATE STATE ---
        unique_dates = df_team_history['date'].dt.date.unique()
        unique_dates.sort()
        if 'view_date' not in st.session_state: st.session_state.view_date = unique_dates[-1]
        
        # === PLAYER DETAILS VIEW ===
        if st.session_state.selected_player_view:
            pid = st.session_state.selected_player_view
            
            current_view_date = st.session_state.view_date
            
            full_history_snapshot = df_team_history[
                (df_team_history['Name'] == pid) & 
                (df_team_history['date'].dt.date <= current_view_date)
            ].sort_values('date')
            
            if full_history_snapshot.empty:
                st.error("No data available for this player on selected date.")
                if st.button("Back"):
                    st.session_state.selected_player_view = None
                    st.rerun()
                st.stop()

            player_snapshot = full_history_snapshot.iloc[-1]

            st.title(f"Profile: {pid}")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Status on Date")
                d_str = player_snapshot['date'].strftime('%Y-%m-%d')
                st.markdown(f"**📅 Viewing:** `{d_str}`")
                
                # RISK DISPLAY
                risk_pct = player_snapshot['Risk']
                # Determine color based on Calculated Threshold
                thresh_pct = best_threshold * 100
                
                if risk_pct > thresh_pct: color = "red"
                elif risk_pct > (thresh_pct / 2): color = "orange"
                else: color = "green"
                
                st.markdown(f"<h1 style='color:{color}; margin-top:0'>{risk_pct:.1f}%</h1>", unsafe_allow_html=True)
                
                st.metric("Load", f"{player_snapshot.get('daily_load', 0):.0f}")
                st.metric("Fatigue", f"{player_snapshot.get('fatigue', 0):.1f}")
                st.metric("ACWR", f"{player_snapshot.get('acwr_ewma', 0):.2f}")
                
            with c2:
                st.plotly_chart(render_radar_chart(player_snapshot), use_container_width=True)

            st.divider()
            st.subheader("Analysis (Up to Selected Date)")
            t1, t2 = st.tabs(["Risk Trend", "Load vs Fatigue"])
            with t1: st.plotly_chart(render_history_chart(full_history_snapshot, best_threshold*100), use_container_width=True)
            with t2: st.plotly_chart(render_load_fatigue_chart(full_history_snapshot), use_container_width=True)

            if app_settings['manager_can_simulate']:
                st.divider()
                st.subheader("Simulation (Next Session)")
                st.caption("Plan tomorrow's Load to see the impact on Risk.")
                st.info("Coming soon")

        # === DASHBOARD VIEW ===
        else:
            st.title(f"Team A Overview")
            
            if 'slider_key' not in st.session_state: st.session_state.slider_key = st.session_state.view_date
            if 'date_input_key' not in st.session_state: st.session_state.date_input_key = st.session_state.view_date

            def update_from_slider():
                st.session_state.view_date = st.session_state.slider_key
                st.session_state.date_input_key = st.session_state.view_date

            def update_from_date_input():
                user_date = st.session_state.date_input_key
                valid_dates = [d for d in unique_dates if d <= user_date]
                new_date = valid_dates[-1] if valid_dates else unique_dates[0]
                st.session_state.view_date = new_date
                st.session_state.slider_key = new_date

            selected_date_slider = st.select_slider("View Team Status on Date", options=unique_dates, key='slider_key', on_change=update_from_slider)
            selected_date_input = st.date_input("Or insert date:", key='date_input_key', on_change=update_from_date_input)
            
            final_view_date = st.session_state.view_date
            df_snapshot = df_team_history[df_team_history['date'].dt.date <= final_view_date]
            df_view = df_snapshot.groupby('Name').tail(1).reset_index(drop=True)
            
            st.divider()

            # --- SNAPSHOT METRICS ---
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Squad Size", len(df_view))
            
            avg_fatigue = df_view['fatigue'].mean()
            avg_fatigue = 0 if pd.isna(avg_fatigue) else avg_fatigue
            k2.metric("Avg Fatigue", f"{avg_fatigue:.2f}/5.0")
            
            avg_acwr = df_view['acwr_ewma'].mean()
            avg_acwr = 0 if pd.isna(avg_acwr) else avg_acwr
            k3.metric("Avg ACWR", f"{avg_acwr:.2f}")
            
            thresh_pct = best_threshold * 100
            at_risk_count = len(df_view[df_view['Risk'] >= thresh_pct])
            k4.metric("Players at risk", at_risk_count, delta_color="inverse")
            
            st.write("")
            sort_by = st.selectbox("Sort By", ["Risk", "Fatigue", "Name"])
            if "Risk" in sort_by: df_view = df_view.sort_values('Risk', ascending=False)
            elif "Fatigue" in sort_by: df_view = df_view.sort_values('fatigue', ascending=False)
            else:
                df_view['sort_num'] = df_view['Name'].str.extract('(\d+)').astype(float)
                df_view = df_view.sort_values('sort_num')

            # --- TABLE ---
            cols = st.columns([2.5, 2, 1, 1, 1, 1])
            cols[0].write("**Player**")
            cols[1].write("**Last Update**")
            cols[2].write("**Load**")
            cols[3].write("**Fatigue**")
            cols[4].write("**Risk**")
            cols[5].write("**View**")
            st.markdown("---")
            
            for i, row in df_view.iterrows():
                cols = st.columns([2.5, 2, 1, 1, 1, 1])
                risk_val = row['Risk']
                
                # Check against dynamic threshold
                if risk_val >= thresh_pct: 
                    cols[0].markdown(f"🔴 **{row['Name']}**")
                    rc = "red"
                elif risk_val >= (thresh_pct / 2): 
                    cols[0].write(f"**{row['Name']}**")
                    rc = "orange"
                else: 
                    cols[0].write(f"**{row['Name']}**")
                    rc = "green"
                
                cols[1].write(f"`{row['date'].strftime('%Y-%m-%d')}`")
                
                l_val = row.get('daily_load', 0)
                f_val = row.get('fatigue', 0)
                
                cols[2].write(f"{l_val:.0f}")
                cols[3].write(f"{f_val:.1f}")
                
                cols[4].markdown(f":{rc}[**{risk_val:.1f}%**]")
                
                if cols[5].button("View", key=f"btn_{i}"):
                    st.session_state.selected_player_view = row['Name']
                    st.rerun()
                st.markdown("<hr style='margin:0.2em 0; opacity:0.1'>", unsafe_allow_html=True)

    # PLAYER
    elif st.session_state.role == "Player":
        st.title(f"Hello, {st.session_state.user_id}")
        try:
            me_latest = df_latest[df_latest['Name'] == st.session_state.user_id].iloc[0]
            me_full = df_team_history[df_team_history['Name'] == st.session_state.user_id].sort_values('date')
        except:
            st.error("Error."); st.stop()
            
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Today's Status")
            st.markdown(f"**Date:** `{me_latest['date'].strftime('%Y-%m-%d')}`")
            r = me_latest['Risk']
            thresh_pct = best_threshold * 100
            
            st.metric("Risk", f"{r:.1f}%")
            if r >= thresh_pct: st.error("High Risk")
            else: st.success("Low Risk")
            
            s_val = me_latest.get('sleep_duration', 0)
            if pd.isna(s_val): s_val = 0
            st.write(f"**Sleep:** {s_val}h")
            
        with c2: st.plotly_chart(render_radar_chart(me_latest), use_container_width=True)
            
        st.divider()
        st.subheader("My Trends")

        st.plotly_chart(render_history_chart(me_full, best_threshold*100), use_container_width=True)
