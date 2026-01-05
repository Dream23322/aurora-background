import os
import pandas as pd
import numpy as np
import joblib
import requests
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler

# ===== CONFIG =====
MODEL_PATH = "models/cheat_detection_lstm_model.h5"
SCALER_PATH = "models/scaler.pkl"
INPUT_DIR = "data/check/processed"
API_KEY = "81F7CFCFAF256132B497EE4D4F655879"
CHEAT_THRESHOLD = 0.5

# ===== LOAD MODEL & SCALER =====
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===== UTIL: FETCH STEAM NAMES =====
def get_steam_name(steamid):
    if not API_KEY:
        print("No Steam API key found")
        return None
    try:
        params = {"key": API_KEY, "steamids": steamid}
        r = requests.get("https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/", params=params, timeout=5)
        r.raise_for_status()
        players = r.json().get("response", {}).get("players", [])
        if players:
            return players[0].get("personaname")
    except Exception as e:
        print(f"Failed to fetch steam name for {steamid}: {e}")
    return None

# ===== PROCESS ONE FILE =====
def predict_file(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["tick","label"], errors="ignore")
    X = scaler.transform(df.values)
    
    X_lstm = X.reshape(1, X.shape[0], X.shape[1])
    
    y_pred = model.predict(X_lstm, verbose=0)
    mean_prob = float(np.mean(y_pred))
    suspicious = mean_prob > CHEAT_THRESHOLD
    return suspicious, mean_prob

# ===== MAIN =====
def check_demo_folder(input_dir):
    player_stats = {}

    for player_folder in os.listdir(input_dir):
        player_path = os.path.join(input_dir, player_folder)
        if not os.path.isdir(player_path):
            continue

        steamid = player_folder
        player_stats[steamid] = {"segments": 0, "suspicious": 0, "probs": []}

        for f in os.listdir(player_path):
            if not f.endswith(".csv"):
                continue
            file_path = os.path.join(player_path, f)
            try:
                suspicious, prob = predict_file(file_path)
                player_stats[steamid]["segments"] += 1
                player_stats[steamid]["probs"].append(prob)
                player_stats[steamid]["suspicious"] += int(suspicious)

                status = "[!] Suspicious" if suspicious else "[✓] Legit"
                print(f"{f}: {status} ({prob*100:.2f}%)")
            except Exception as e:
                print(f"[X] Failed to process {file_path}: {e}")

    # ===== SUMMARY =====
    print("\n=== Player Summary ===")
    for steamid, stats in player_stats.items():
        clean_id = steamid.replace("user_", "")
        steam_link = f"https://www.cs2guard.com/player/{clean_id}"
        avg_prob = np.mean(stats["probs"]) * 100 if stats["probs"] else 0
        print(f"Player {get_steam_name(steamid)}: {steam_link} | {stats['segments']} segments, "
            f"{stats['suspicious']} suspicious, avg cheat probability {avg_prob:.2f}%")


if __name__ == "__main__":
    check_demo_folder(INPUT_DIR)
