import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import numpy as np
import base64
import plotly.graph_objects as go

# ================= CONFIG =================
HOST = "digital.iservices.rte-france.com"
DATA_ENDPOINT_PATH = "/open_api/wholesale_market/v3/france_power_exchanges"
DATA_API_URL = f"https://{HOST}{DATA_ENDPOINT_PATH}"
TOKEN_API_URL = f"https://{HOST}/token/oauth/"

CLIENT_ID = "a13b48af-637a-42cc-b332-65335bc3cdea"
CLIENT_SECRET = "8b1f7961-84b9-426b-b439-bea5a20a7f10"

st.set_page_config(page_title="Prix électricité RTE", layout="wide")
st.title("📈 Prix de l'électricité – Historique & Prédiction")

# ================= AUTH =================
@st.cache_data(ttl=3600)
def get_access_token():
    auth_string = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {"grant_type": "client_credentials"}

    response = requests.post(TOKEN_API_URL, headers=headers, data=payload)
    response.raise_for_status()
    return response.json()["access_token"]

# ================= DATA =================
@st.cache_data(ttl=600)
def get_historical_data(token, start_date, end_date):
    headers = {"Authorization": f"Bearer {token}"}
    params = {"start_date": start_date, "end_date": end_date}

    response = requests.get(DATA_API_URL, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    values = data["france_power_exchanges"][0]["values"]
    df = pd.DataFrame(values)

    df["start_date"] = pd.to_datetime(df["start_date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df.dropna().sort_values("start_date")

# ================= OPTIMIZATION =================
def find_optimal_batches(df, num_batches, batch_duration=2):
    df = df.copy()
    df['hour'] = df['start_date'].dt.hour
    windows = []
    for start in range(24 - batch_duration + 1):
        end = start + batch_duration
        avg_price = df[(df['hour'] >= start) & (df['hour'] < end)]['price'].mean()
        windows.append((start, end, avg_price))
    windows.sort(key=lambda x: x[2])  # sort by avg_price ascending
    selected = []
    for start, end, price in windows:
        if not any(s < end and start < e for s, e in selected):
            selected.append((start, end))
            if len(selected) == num_batches:
                break
    return selected

# ================= ANALYSE & PLOT =================
def plot_streamlit(df, batch_times):
    # ----- Modèle simple -----
    df["time_feature"] = (df["start_date"] - df["start_date"].min()).dt.total_seconds()
    X = df[["time_feature"]]
    y = df["price"]

    model = LinearRegression()
    model.fit(X, y)

    # Prédiction heure suivante
    next_date = df["start_date"].max() + timedelta(hours=1)
    next_tf = (next_date - df["start_date"].min()).total_seconds()
    X_next = pd.DataFrame({"time_feature": [next_tf]})
    predicted_price = model.predict(X_next)[0]

    # =========================
    # PLOTLY INTERACTIF
    # =========================
    fig = go.Figure()

    # Courbe principale
    fig.add_trace(go.Scatter(
        x=df["start_date"],
        y=df["price"],
        mode="lines+markers",
        name="Prix observé",
        hovertemplate=
        "<b>Heure</b>: %{x|%H:%M}<br>"
        "<b>Prix</b>: %{y:.2f} €/MWh<br>"
        "<extra></extra>"
    ))

    # Point de prédiction
    fig.add_trace(go.Scatter(
        x=[next_date],
        y=[predicted_price],
        mode="markers",
        marker=dict(size=12, color="red"),
        name="Prédiction",
        hovertemplate=
        "<b>Heure</b>: %{x|%H:%M}<br>"
        "<b>Prix prédit</b>: %{y:.2f} €/MWh<br>"
        "<extra></extra>"
    ))

    # Zones de batches
    day_start = df["start_date"].dt.normalize().iloc[0]
    for start_h, end_h in batch_times:
        fig.add_vrect(
            x0=day_start + pd.Timedelta(hours=start_h),
            x1=day_start + pd.Timedelta(hours=end_h),
            fillcolor="green",
            opacity=0.25,
            layer="below",
            line_width=0
        )

    # Mise en forme axes
    fig.update_layout(
        title="Prix de l'électricité – Profil journalier (interactif)",
        xaxis_title="Heure de la journée (UTC)",
        yaxis_title="Prix (€/MWh)",
        hovermode="x unified",
        xaxis=dict(
            tickformat="%Hh",
            dtick=2 * 3600000,  # 2 heures en ms
            range=[
                day_start,
                day_start + pd.Timedelta(hours=24)
            ]
        )
    )

    st.plotly_chart(fig, width="100%", use_container_width=True)


    st.success(
        f"🔮 Prix prédit pour {next_date.strftime('%Y-%m-%d %H:%M')} UTC : "
        f"**{predicted_price:.2f} €/MWh**"
    )


# ================= MAIN =================
hours = 24

end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(hours=hours)

start_dt = start_dt.strftime("%Y-%m-%dT%H:00:00Z")
end_dt = end_dt.strftime("%Y-%m-%dT%H:00:00Z")

token = get_access_token()
df = get_historical_data(token, start_dt, end_dt)

num_batches = st.number_input("Nombre de batches", min_value=1, max_value=8, value=1, step=1)
batch_times = find_optimal_batches(df, num_batches)

plot_streamlit(df, batch_times)

# Calendrier
st.subheader("Calendrier des batches")
batch_hours = set()
for s, e in batch_times:
    batch_hours.update(range(s, e))

html = '<div style="display: flex; flex-wrap: wrap; gap: 5px;">'
for h in range(24):
    color = 'green' if h in batch_hours else 'lightgray'
    html += f'<div style="width: 60px; height: 60px; background-color: {color}; border: 1px solid black; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: bold;">{h:02d}h</div>'
html += '</div>'
st.markdown(html, unsafe_allow_html=True)
