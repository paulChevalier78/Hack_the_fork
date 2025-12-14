import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import base64

# --- Configuration API ---
HOST = "digital.iservices.rte-france.com"
DATA_ENDPOINT_PATH = "/open_api/wholesale_market/v3/france_power_exchanges"
DATA_API_URL = f"https://{HOST}{DATA_ENDPOINT_PATH}"
TOKEN_API_URL = f"https://{HOST}/token/oauth/"

CLIENT_ID = "a13b48af-637a-42cc-b332-65335bc3cdea"
CLIENT_SECRET = "8b1f7961-84b9-426b-b439-bea5a20a7f10"

# --- Obtenir token d'accès ---
def get_access_token():
    auth_string = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    
    headers = {
        'Authorization': f'Basic {encoded_auth}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    payload = {'grant_type': 'client_credentials'}
    response = requests.post(TOKEN_API_URL, headers=headers, data=payload)
    response.raise_for_status()
    return response.json().get('access_token')

# --- Récupérer les données historiques ---
def get_historical_data(token, start_date, end_date):
    headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
    params = {'start_date': start_date, 'end_date': end_date}
    response = requests.get(DATA_API_URL, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    
    if isinstance(data, list) and len(data) > 0:
        first_item = data[0]
        values = first_item.get('values', [])
        if not values:
            raise ValueError("Aucun point de données dans 'values'.")
        # Créer DataFrame
        processed = [{'start_date': v['start_date'], 'price': v['price']} for v in values]
        df = pd.DataFrame(processed)
        df['start_date'] = pd.to_datetime(df['start_date'])
        df = df.sort_values('start_date')
        return df
    else:
        raise ValueError("Données API vides ou mal formatées.")

# --- Prédiction simple ---
def predict_next_hours(df, hours=3):
    df = df.dropna(subset=['price'])
    df['time_feature'] = (df['start_date'] - df['start_date'].min()).dt.total_seconds()
    
    X = df[['time_feature']]
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    
    last_time = df['start_date'].max()
    predictions = []
    for i in range(1, hours+1):
        next_time = last_time + timedelta(hours=i)
        t_feature = (next_time - df['start_date'].min()).total_seconds()
        pred_price = model.predict(np.array([[t_feature]]))[0]
        predictions.append((next_time, pred_price))
    return predictions

# --- Affichage Matplotlib ---
def plot_prices(df, predictions):
    plt.figure(figsize=(12,6))
    plt.plot(df['start_date'], df['price'], marker='o', label='Prix observé')
    
    pred_dates = [p[0] for p in predictions]
    pred_prices = [p[1] for p in predictions]
    plt.scatter(pred_dates, pred_prices, color='green', s=120, label='Prédiction')
    
    plt.title("Prix de l'électricité en France (Historique + Prédiction)")
    plt.xlabel("Date/Heure")
    plt.ylabel("Prix (€/MWh)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Exécution ---
if __name__ == "__main__":
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=6)  # dernières 6 heures
    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    token = get_access_token()
    df_prices = get_historical_data(token, start_str, end_str)
    predictions = predict_next_hours(df_prices, hours=3)
    
    print("Prédictions pour les 3 prochaines heures :")
    for t, p in predictions:
        print(f"{t}: {p:.2f} €/MWh")
    
    plot_prices(df_prices, predictions)
