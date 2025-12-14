import test  # On importe test.py
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta, timezone

# --- 1. Obtenir le token ---
token = test.get_access_token()
if not token:
    raise RuntimeError("Impossible de récupérer le token.")

# --- 2. Définir la période pour récupérer les données ---
end_dt_raw = datetime.now(timezone.utc)
end_dt = end_dt_raw.strftime("%Y-%m-%dT%H:00:00Z")
hours_to_fetch = 48  # Par exemple, récupérer les 48 dernières heures
start_dt = (end_dt_raw - timedelta(hours=hours_to_fetch)).strftime("%Y-%m-%dT%H:00:00Z")

# --- 3. Récupérer les données historiques ---
data_df = test.get_historical_data(token, start_dt, end_dt)

if data_df.empty:
    raise RuntimeError("Aucune donnée récupérée.")

# --- 4. Tracer la courbe ---
plt.figure(figsize=(12,6))
plt.plot(pd.to_datetime(data_df['start_date']), data_df['price'], marker='o', linestyle='-')
plt.title("Prix de l'électricité en fonction des heures")
plt.xlabel("Date/Heure")
plt.ylabel("Prix (€/MWh)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
