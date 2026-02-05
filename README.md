# Hybrid Forex Trading Robot 🤖📈

## Overview & Goal

Note: This repository represents the public, open-source strategy logic for ZeroCore, my proprietary distributed trading engine. It demonstrates the hybrid algorithmic approach (Rules + ML) without exposing the private execution infrastructure.
This project details the development of a hybrid Forex trading robot that combines a rules-based trading strategy with machine learning (AI) components. The core strategy is a customized **TRSS (Trend Reversal Structure Strategy)** designed to trade currency pairs like GBPUSD.

The primary goal is to create a more adaptive and robust automated trading system compared to purely rule-based or purely AI-driven approaches. By leveraging the consistency of rules and the pattern-recognition capabilities of AI, this project aims to solve challenges like static rule limitations in dynamic markets and the "black box" nature of some AI systems by creating a structured, yet intelligent, trading logic.

## Key Features & Current Status

The system is being built and refined component by component. The **core rules-based signal generation engine is now functionally complete** and undergoing iterative parameter tuning. The project is now moving into the **AI component development and backtesting phases.**

* **Data Ingestion & Preparation (`data_loader.py`):**
    * Loads historical D1 and H1 price data (OHLC, Volume) from CSV files.
    * Standardizes column names, handles datetime parsing, sets a DatetimeIndex, and localizes to UTC for consistency.
* **Technical Indicator Calculation (`indicators.py`):**
    * Calculates 50-period Simple Moving Average (SMA) for D1 trend filtering.
    * Calculates 14-period Average True Range (ATR) for H1 (used for Stop Loss calculation).
    * Calculates 10-period Relative Strength Index (RSI) for H1 (used for W/M pattern divergence).
* **D1 Market Structure Analysis (`trss_D1.py`):**
    * Identifies D1 Major Swing Highs (MSH) and Major Swing Lows (MSL) using a 3-candle pullback rule.
    * Determines D1 market structure ("Uptrend," "Downtrend," "Break of Structure" states) based on the sequence of these MSH/MSL points.
* **H1 Pattern Recognition (`patterns_H1.py`):**
    * Detects H1 W-Patterns (Double Bottoms) confirmed with Bullish RSI Divergence to generate buy signals.
    * Detects H1 M-Patterns (Double Tops) confirmed with Bearish RSI Divergence to generate sell signals.
    * *(Head & Shoulders patterns have been deferred to focus on W/M patterns for now).*
* **Rules-Based Signal Generation (`signal_engine.py`):**
    * Successfully combines the D1 market structure context (including the D1 50 SMA filter) with H1 entry patterns to generate preliminary "BUY\_RULE" and "SELL\_RULE" signals.
* **Risk Management Rules (`risk_rules.py`):**
    * Calculates Stop Loss (SL) based on H1 ATR from the signal candle's high/low plus a fixed pip offset.
    * Calculates a baseline Take Profit (TP) based on a 2:1 Risk:Reward ratio.
* **GUI Framework (by Brilliant Mpofu):**
    * A pre-existing user interface developed with Streamlit provides the foundation for visualizing market data, bot activity, and backtest results. Integration with the backend trading logic is a key future task.

## Preliminary Rules-Based Backtest Results

The following are initial backtest results for the **rules-based engine only** (before AI enhancement), run on the H1 training set data from **July 2021 to January 2024**. These results were generated from a specific set of H1 pattern parameters and are subject to change as parameter tuning continues. They provide a performance baseline to build upon.

* **Total Trades Simulated:** 100
* **Win Rate:** 37.37% (37 Wins / 62 Losses)
* **Total P&L:** +206.59 pips
* **Average Win:** +76.19 pips
* **Average Loss:** -43.37 pips
* **Realized Avg. Risk:Reward Ratio:** 1.76

*Note: These results do not include commissions or slippage.*

## Development Methodology: AI-Augmented Engineering

As the lead architect, I leveraged Large Language Models (LLMs) to accelerate the development lifecycle, focusing on high-level architectural logic while automating boilerplate generation.

* **Architecture First:** I defined the modular constraints (Signal Engine vs. Risk Engine) and data contracts.
* **Rapid Prototyping:** Used AI to generate initial Python function stubs for standard indicators (RSI, ATR) to speed up implementation.
* **Rigorous Validation:** All AI-generated logic was subjected to manual code review and backtested against visual chart analysis to ensure mathematical accuracy (e.g., verifying 3-candle pullback logic).

## Technologies & Tools

* **Core Language:** Python (Project standard: 3.10.x)
* **Data Handling & Analysis:** Pandas, NumPy
* **Machine Learning (AI Components):** Scikit-learn (for K-Means, `StandardScaler`), TensorFlow/Keras (for LSTM)
* **Plotting/Visualization:** Matplotlib
* **Development Environment:** Jupyter Notebooks, VS Code
* **Data Acquisition (Planned):** MetaTrader 5 (MT5) Python API
* **Model Persistence (Planned for AI):** `joblib`, `.h5` / Keras SavedModel format
* **GUI:** Streamlit
* **AI Coding Partner:** Gemini

## Current Status

**Active Development.** The core rules-based engine is functionally complete and generating preliminary trade signals with SL/TP levels. Current work involves the iterative parameter tuning of the rules-based logic (based on initial backtesting) and starting the development of the AI components (beginning with K-Means).

## Challenges & Learnings

* **Python Environment & Dependencies:** Overcame initial `pip install` failures due to SSL issues by using `--trusted-host` and managing virtual environments carefully.
* **Module Import Logic:** Debugged `sys.path` manipulation for importing custom modules from the `src` directory into Jupyter Notebooks.
* **Pandas DataFrame Operations:** Troubleshot `KeyError`s and subtle `pd.merge` issues by meticulously tracing DataFrame modifications to ensure column propagation through the data processing pipeline.
* **Iterative Pattern Recognition:** Developing Python functions for H1 W/M patterns that accurately reflect visual chart patterns is a highly iterative process requiring careful parameter tuning and extensive visual testing.
* **Translating Trading Concepts into Code:** Articulating nuanced trading rules (like multi-step BoS confirmation) into precise Python logic required detailed discussion and iterative refinement.

## Future Ideas

* Implement and test Head & Shoulders (H&S) / Inverse H&S H1 entry patterns.
* Build and integrate the K-Means market regime filter.
* Build and integrate the LSTM model for H1 entry confirmation and dynamic Take Profit.
* Implement `main_processor.py` to fully orchestrate the rules and AI components.
* Connect to the MT5 API for live data and potential trade execution.
* Conduct comprehensive backtesting of the full hybrid model using a dedicated library (e.g., `backtesting.py`).
* Further tune all strategy and AI parameters based on detailed backtesting results.
* **Complete the integration of the backend logic with the existing Streamlit GUI.**
* Explore XGBoost for additional confidence scoring if time permits.
* Implement more advanced trade management features like trailing stops or partial take profits.

## License

This project is licensed under the MIT License. You can see the full license in the `LICENSE` file in this repository.
