import os
import json
import joblib
import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# FastAPI app inicializálása
app = FastAPI(title="NQ ML Gatekeeper Server")

# Kereskedési beállítások
CONFIDENCE_THRESHOLD = 0.45  # 45% feletti modell-bizonyosságnál lőjük ki a ravaszt
YOUR_WEBAPP_URL = "https://a-te-webappod-cime.com/webhook"  # Ide küldjük a jelet, ha jó

# A modell betöltése a memóriába indításkor
MODEL_PATH = "nq_scalp_model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("[+] Az ML Modell sikeresen betöltve a memóriába.")
else:
    raise FileNotFoundError(f"Hiba: A {MODEL_PATH} nem található a gyökérmappában!")

# TradingView-ból érkező JSON struktúra validálása
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

def forward_to_execution(side: str):
    """
    Aszinkron módon továbbküldi a jelet a te már meglévő Tradovate webappodnak.
    Így a FastAPI azonnal válaszol a TradingView-nak, nincs késleltetés.
    """
    payload = {
        "ticker": "NQ",
        "action": side,            # "BUY" vagy "SELL"
        "sl_points": 15.0,         # Az optimalizált Stop Loss
        "tp_points": 30.0,         # Az optimalizált Take Profit
        "magic": 2026              # Azonosító az algo kötéshez
    }
    try:
        response = requests.post(YOUR_WEBAPP_URL, json=payload, timeout=5)
        print(f"[+] Jel továbbítva a webappnak. Válasz: {response.status_code}")
    except Exception as e:
        print(f"[-] Hiba a jel továbbításakor: {e}")

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": True}

@app.post("/tv-webhook")
def process_tradingview_signal(data: TVPayload, background_tasks: BackgroundTasks):
    # 1. A beérkező adatokból pontosan ugyanolyan DataFrame-et építünk, mint a tanításnál
    features = pd.DataFrame([{
        'Open': data.open,
        'High': data.high,
        'Low': data.low,
        'Close': data.close,
        'Volume': data.volume,
        'ATR': data.atr,
        'RSI': data.rsi,
        'Vol_ZScore': data.vol_zscore,
        'Hour': data.hour,
        'Minute': data.minute
    }])
    
    # 2. Modell valószínűségek lekérése [Class 0: No Trade, Class 1: Long, Class 2: Short]
    probabilities = model.predict_proba(features)[0]
    long_prob = probabilities[1]
    short_prob = probabilities[2]
    
    print(f"[i] Szignál érkezett - Záróár: {data.close} | Long Prob: {long_prob:.2f} | Short Prob: {short_prob:.2f}")
    
    # 3. Döntési logika a beállított küszöbérték alapján
    if long_prob > CONFIDENCE_THRESHOLD and long_prob > short_prob:
        print(f"[🔥] LONG SZIGNÁL JÓVÁHAGYVA ({long_prob*100:.1f}%) -> Ravasz meghúzása...")
        background_tasks.add_task(forward_to_execution, "BUY")
        return {"decision": "BUY", "confidence": float(long_prob)}
        
    elif short_prob > CONFIDENCE_THRESHOLD and short_prob > long_prob:
        print(f"[🔥] SHORT SZIGNÁL JÓVÁHAGYVA ({short_prob*100:.1f}%) -> Ravasz meghúzása...")
        background_tasks.add_task(forward_to_execution, "SELL")
        return {"decision": "SELL", "confidence": float(short_prob)}
        
    else:
        print("[💤] Szignál elutasítva: Nem érte el a magabiztossági küszöböt.")
        return {"decision": "NO_TRADE", "long_prob": float(long_prob), "short_prob": float(short_prob)}
