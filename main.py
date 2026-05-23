import os
import json
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

# FastAPI alkalmazás inicializálása
app = FastAPI(title="NQ ML Gatekeeper - PickMyTrade Edition")

# Kereskedési beállítások és a te egyedi API végpontod
PICKMYTRADE_URL = "https://api.pickmytrade.trade/v2/add-trade-data-latest?t=17149"
CONFIDENCE_THRESHOLD = 0.45  # 45% feletti modell-bizonyosságnál lőjük ki a ravaszt

# Az ML modell betöltése a memóriába indításkor
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

def forward_to_execution(side: str, close_price: float):
    """
    Összeállítja a PickMyTrade által elvárt pontos JSON struktúrát,
    dinamikusan behelyettesíti az árat és az időt, majd kilövi a ravaszt.
    """
    # Aktuális időbélyeg lekérése a megfelelő formátumban
    current_time_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # A te pontos, egyedi JSON üzenetsablonod precíz replikációja
    payload = {
        "strategy_name": "",
        "symbol": "MNQM6",
        "date": current_time_str,
        "data": side,  # "buy" vagy "sell"
        "quantity": 10,
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
    
    # Küldési folyamat végrehajtása a PickMyTrade felé
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(PICKMYTRADE_URL, json=payload, headers=headers, timeout=5)
        print(f"[+] Szignál ({side.upper()}) kiküldve a PickMyTrade-nek. Ár: {close_price} | Válasz: {response.status_code}")
    except Exception as e:
        print(f"[-] Hiba történt a PickMyTrade API elérése közben: {e}")

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": True, "target_endpoint": "PickMyTrade"}

@app.post("/tv-webhook")
def process_tradingview_signal(data: TVPayload, background_tasks: BackgroundTasks):
    # 1. DataFrame építése az érkező gyertya adataiból a modell számára
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
    
    # 2. Modell valószínűségek lekérése [0: No Trade, 1: Long (buy), 2: Short (sell)]
    probabilities = model.predict_proba(features)[0]
    long_prob = probabilities[1]
    short_prob = probabilities[2]
    
    print(f"[i] Bejövő TV adat - Ár: {data.close} | Buy Prob: {long_prob:.2f} | Sell Prob: {short_prob:.2f}")
    
    # 3. Modell alapú szűrés és döntéshozatal
    if long_prob > CONFIDENCE_THRESHOLD and long_prob > short_prob:
        print(f"[🔥] BUY SZIGNÁL ÁTENGEDVE ({long_prob*100:.1f}%) -> Végrehajtás indítása...")
        # Háttérfolyamatként indítjuk a küldést, hogy a TradingView felé azonnali legyen a válaszidő
        background_tasks.add_task(forward_to_execution, "buy", data.close)
        return {"decision": "BUY", "confidence": float(long_prob)}
        
    elif short_prob > CONFIDENCE_THRESHOLD and short_prob > long_prob:
        print(f"[🔥] SELL SZIGNÁL ÁTENGEDVE ({short_prob*100:.1f}%) -> Végrehajtás indítása...")
        background_tasks.add_task(forward_to_execution, "sell", data.close)
        return {"decision": "SELL", "confidence": float(short_prob)}
        
    else:
        print("[💤] Szignál blokkolva: Nem érte el a matematikai minimum küszöböt.")
        return {"decision": "NO_TRADE", "buy_prob": float(long_prob), "sell_prob": float(short_prob)}
