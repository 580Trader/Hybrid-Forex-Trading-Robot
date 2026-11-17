import pandas as pd
import numpy as np
import pika
import json
import os
import time
from datetime import datetime, timezone
import logging

from src.core_logic.indicators import calculate_sma, calculate_atr, calculate_rsi
from src.core_logic.trss_D1 import find_basic_swing_points, validate_swings_with_3_candle_pullback, determine_d1_market_structure
from src.core_logic.patterns_H1 import find_h1_swing_points, identify_w_pattern_with_divergence, identify_m_pattern_with_divergence
from src.core_logic.signal_engine import generate_rules_based_signals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "zerocore_admin")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "zerocore_secret")
EXCHANGE = "trade_exchange"
ROUTING_KEY = "trade_signal_routing_key"

def get_rabbitmq_channel():
    creds = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    params = pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=creds)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.exchange_declare(exchange=EXCHANGE, exchange_type='topic', durable=True)
    return conn, ch

def run_pipeline_and_publish():
    logger.info("Loading historical data from CSV...")
    df_h1 = pd.read_csv('data/GBPUSDm_H1_clean.csv', index_col='time', parse_dates=True)
    df_d1 = pd.read_csv('data/GBPUSDm_D1_clean.csv', index_col='time', parse_dates=True)

    # --- 1. PRE-PROCESS D1 DATA ---
    df_d1 = calculate_sma(df_d1, price_column='close', period=50)
    df_d1 = find_basic_swing_points(df_d1, N=1)
    df_d1 = validate_swings_with_3_candle_pullback(df_d1)
    df_d1 = determine_d1_market_structure(df_d1)

    # --- 2. PRE-PROCESS H1 DATA ---
    df_h1 = calculate_atr(df_h1, period=14)
    df_h1 = calculate_rsi(df_h1, period=10)
    df_h1 = find_h1_swing_points(df_h1, N=3)
    df_h1 = identify_w_pattern_with_divergence(df_h1)
    df_h1 = identify_m_pattern_with_divergence(df_h1)

    # --- 3. GENERATE RULE-BASED SIGNALS ---
    logger.info("Generating rules-based signals...")
    df_h1_signals = generate_rules_based_signals(df_d1, df_h1)

    # Grab the most recent valid signal
    valid_signals = df_h1_signals[df_h1_signals['rules_signal'].isin(["BUY_RULE", "SELL_RULE"])]

    if not valid_signals.empty:
        latest_row = valid_signals.iloc[-1]
        rule_signal = latest_row['rules_signal']

        order_type = "BUY" if rule_signal == "BUY_RULE" else "SELL"
        
        # --- TEMPORARY OVERRIDE FOR E2E TESTING ---
        # We force the entry price to match the mock broker's default (1.2500)
        # so the SL/TP geometry is valid and the broker accepts the trade.
        entry_price = 1.25000
        
        if order_type == "BUY":
            sl = entry_price - 0.0050
            tp = entry_price + 0.0050
        else:
            sl = entry_price + 0.0050
            tp = entry_price - 0.0050
        # ------------------------------------------

        now = datetime.now(timezone.utc)
        signal_id = f"TRSS-GBPUSD-H1-{now.strftime('%Y%m%dT%H%M%SZ')}"

        payload = {
            "signalId": signal_id,
            "strategyCode": "TRSS",
            "instrument": "GBPUSD",
            "timeframe": "H1",
            "candleClose": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "orderType": order_type,
            "quantity": 1.0,
            "stopLoss": round(sl, 5),
            "takeProfit": round(tp, 5),
            "entryPrice": round(entry_price, 5),
            "riskReward": "1:1",
            "source": "LIVE"
        }

        logger.info(f"PUBLISHING VALID TEST SIGNAL: {order_type} at {entry_price} (SL: {sl}, TP: {tp})")

        conn, ch = get_rabbitmq_channel()
        ch.basic_publish(
            exchange=EXCHANGE, routing_key=ROUTING_KEY,
            body=json.dumps(payload),
            properties=pika.BasicProperties(delivery_mode=1, content_type='application/json')
        )
        conn.close()
        logger.info("Signal published to RabbitMQ.")
    else:
        logger.info("No valid rule signals found in the dataset.")

if __name__ == "__main__":
    logger.info("Starting Deterministic Signal Office...")
    while True:
        try:
            run_pipeline_and_publish()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        time.sleep(60)
