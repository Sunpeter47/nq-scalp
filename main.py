import os
import json
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="NQ ML Gatekeeper - Final Production Version")

# A Render felületéről olvassa be a küszöböt. 
# Ha nincs ott beállítva semmi, akkor az optimális 0.38-as (38%) értéket használja.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.38))
PICKMYTRADE_URL = "https://api.pickmytrade.trade/v2/add-trade-data-latest?t=17149"

# Modell betöltése
MODEL_PATH = "nq_scalp_model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"[+] ML Modell betöltve. Aktuális küszöbérték: {CONFIDENCE_THRESHOLD}")
else:
    raise FileNotFoundError(f"Hiba: A {MODEL_PATH} nem található!")

class TVPayload(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    atr: float
    rsi: float
    vol_zscore: float
    hour: int
    minute: int

def forward_to_execution(side: str, close_price: float):
    current_time_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    payload = {
        "strategy_name": "",
        "symbol": "MNQM6",
        "date": current_time_str,
        "data": side,
        "quantity": 5,  # FRISSÍTVE: Kockázatkezelés beállítva 5 kontraktusra
        "risk_percentage": 0,
        "price": close_price,
        "gtd_in_second": 0,
        "stp_limit_stp_price": 0,
        "tp": 0,
        "percentage_tp": 0,
        "dollar_tp": 30,
        "sl": 0,
        "percentage_sl": 0,
        "dollar_sl": 15,
        "trail": 0,
        "trail_stop": 0,
        "trail_trigger": 0,
        "trail_freq": 0,
        "update_tp": False,
        "update_sl": False,
        "breakeven": 15,
        "breakeven_offset": 1,
        "token": "zpgsnm8el3DfHLBvOu4SHg",
        "pyramid": False,
        "same_direction_ignore": False,
        "reverse_order_close": True,
        "order_type": "MKT",
        "multiple_accounts": [
            {
                "token": "zpgsnm8el3DfHLBvOu4SHg",
                "account_id": "LFE05067983860008",
                "risk_percentage": 0,
                "quantity_multiplier": 1
            }
        ]
    }
    
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(PICKMYTRADE_URL, json=payload, headers=headers, timeout=5)
        print(f"[+] Szignál ({side.upper()}) elküldve a PickMyTrade-nek. Státusz: {response.status_code}")
    except Exception as e:
        print(f"[-] Végrehajtási hiba: {e}")

@app.get("/")
def read_root():
    return {"status": "online", "confidence_threshold": CONFIDENCE_THRESHOLD, "target": "PickMyTrade"}

@app.post("/tv-webhook")
def process_tradingview_signal(data: TVPayload, background_tasks: BackgroundTasks):
    features = pd.DataFrame([{
        'Open': data.open, 'High': data.high, 'Low': data.low, 'Close': data.close,
        'Volume': data.volume, 'ATR': data.atr, 'RSI': data.rsi, 'Vol_ZScore': data.vol_zscore,
        'Hour': data.hour, 'Minute': data.minute
    }])
    
    probabilities = model.predict_proba(features)[0]
    long_prob = probabilities[1]
    short_prob = probabilities[2]
    
    print(f"[i] Szignál elemzés -> Buy Prob: {long_prob:.2f} | Sell Prob: {short_prob:.2f} | Küszöb: {CONFIDENCE_THRESHOLD}")
    
    if long_prob > CONFIDENCE_THRESHOLD and long_prob > short_prob:
        print(f"[🔥] BUY ÁTENGEDVE ({long_prob*100:.1f}%) -> Küldés...")
        background_tasks.add_task(forward_to_execution, "buy", data.close)
        return {"decision": "BUY", "confidence": float(long_prob)}
        
    elif short_prob > CONFIDENCE_THRESHOLD and short_prob > long_prob:
        print(f"[🔥] SELL ÁTENGEDVE ({short_prob*100:.1f}%) -> Küldés...")
        background_tasks.add_task(forward_to_execution, "sell", data.close)
        return {"decision": "SELL", "confidence": float(short_prob)}
        
    else:
        return {"decision": "NO_TRADE", "buy_prob": float(long_prob), "sell_prob": float(short_prob)}
