# Hybrid Forex Trading Robot 🤖📈

## Overview & Goal

This project aims to develop a hybrid Forex trading robot that combines a rules-based trading strategy (customized TRSS - Trend Reversal Structure Strategy) with machine learning (AI) components to enhance decision-making for trading currency pairs like GBPUSD. The core purpose is to create a more adaptive and potentially more robust automated trading system compared to purely rule-based or purely AI-driven approaches by leveraging the strengths of both. It attempts to solve the problem of static rule limitations in dynamic markets and the "black box" nature of some AI systems by creating a structured, yet intelligent, trading logic.

## Key Features (Current Implementation & In Progress)

Our development is actively progressing. Here's what we've built and are currently testing/refining:

* **Data Ingestion & Preparation (`data_loader.py`):**
    * Loads historical D1 and H1 price data (OHLC, Volume) from CSV files.
    * Standardizes column names, handles datetime parsing, sets a DatetimeIndex, and localizes to UTC.
    * Ensures data is numeric.
* **Technical Indicator Calculation (`indicators.py`):**
    * Calculates 50-period Simple Moving Average (SMA) for D1 trend filtering.
    * Calculates 14-period Average True Range (ATR) for H1 (for Stop Loss).
    * Calculates 10-period Relative Strength Index (RSI) for H1 (for W/M pattern divergence).
* **D1 Market Structure Analysis (`trss_D1.py`):**
    * Identifies potential D1 swing highs and lows (`find_basic_swing_points`).
    * Validates these into Major Swing Highs (MSH) and Major Swing Lows (MSL) using a 3-candle pullback rule (`validate_swings_with_3_candle_pullback`).
    * Determines D1 market structure ("Undetermined", "Uptrend", "Downtrend", "Bearish BoS Target", "Bullish BoS Target", and BoS confirmations) based on the sequence of MSH/MSL and specific Break of Structure (BoS) confirmation rules (`determine_d1_market_structure`). This component is undergoing active iterative testing and refinement.
* **H1 Pattern Recognition (`patterns_H1.py`):**
    * Identifies H1 swing points (`find_h1_swing_points`).
    * Detects W-Patterns (Double Bottoms) with Bullish RSI Divergence, generating `w_pattern_buy_signal`. Parameter tuning for this is in progress.
    * Detects M-Patterns (Double Tops) with Bearish RSI Divergence, generating `m_pattern_sell_signal`. Implementation and parameter tuning for this is in progress.
    * *(Head & Shoulders patterns have been deferred for now to focus on W/M patterns).*
* **Rules-Based Signal Generation (`signal_engine.py`):**
    * Logic defined to combine D1 market structure (including D1 50 SMA filter) with H1 pattern signals to generate preliminary "BUY_RULE" or "SELL_RULE" signals. Testing and debugging of signal generation based on D1/H1 alignment is in progress.
* **Risk Management Rules (`risk_rules.py`):**
    * Function drafted to calculate Stop Loss (SL) based on H1 ATR (from the low/high of H1 signal candle + 5 pips offset).
    * Function drafted to calculate a baseline Take Profit (TP) based on a 2:1 Risk:Reward ratio.
* **AI Integration Planning (`main_processor.py`, AI tasks for Mthabisi):**
    * The central `main_processor.py` is being designed to orchestrate the rules-based signals with AI model outputs.
    * K-Means (for Mthabisi): Detailed tasks defined for developing a D1 market regime identification model.
    * LSTM (for Mthabisi): Detailed tasks defined for developing an H1 entry confirmation and dynamic Take Profit (R:R) adjustment model.
* **Data Source Plan:**
    * The system design includes fetching data via the MT5 Python library from an Exness account.

## My Development Process & Collaboration with Gemini

As Project Manager and the lead on the rules-based logic and system integration, I've been driving the development by defining the architecture and working closely with Gemini as a coding partner and technical consultant.

* **Initial Vision & Architecture:** I established the foundational concept of a hybrid model, combining my trading knowledge of the TRSS strategy with AI enhancements, as outlined in my initial project proposal documents. This formed the blueprint for our work.
* **Iterative Development with Gemini:**
    * **Component-by-Component Build:** I've approached the project by breaking it down into manageable components: data loading, indicator calculation, D1 structure analysis, H1 pattern recognition, signal generation, and risk management.
    * **Providing Specifications to Gemini:** For each module (e.g., `data_loader.py`, `indicators.py`, `trss_D1.py`, `patterns_H1.py`), I provided Gemini with specific requirements, the core trading logic, and detailed rules. This included CSV data structure, indicator formulas, D1 swing point criteria, trend logic (my "1-2-3 movement" concept), BoS confirmation rules, H1 W/M pattern definitions (referencing external videos and RSI divergence), and Stop Loss/Take Profit calculation rules.
    * **Code Generation and Adaptation:** Gemini assisted by providing initial Python code structures and function templates. I then integrated this code into my project's `.py` modules and adapted it as needed.
    * **Intensive Testing & Debugging in Jupyter Notebooks:** This has been a crucial part of my process. For each function and module, I load data, call functions, rigorously inspect outputs (DataFrame methods, intermediate variables), and perform visual verification with Matplotlib (plotting prices, indicators, swing points, signals, market structure) against my manual chart analysis.
    * **Collaborative Troubleshooting:** Through this testing, I identify discrepancies or errors. I then either debug the Python code directly or describe the issues and my observations to Gemini for further collaborative troubleshooting and code revisions. This iterative loop has been central to our progress.
* **Defining AI Component Tasks:** For the K-Means and LSTM models, I defined their roles within the hybrid system (D1 regime filtering; H1 entry confirmation & dynamic R:R) and collaborated with Gemini to create detailed task lists for Mthabisi, the ML engineer.
* **GUI Integration Planning:** I am planning the GUI integration. With Gemini's assistance, we've drafted an "Interface Contract" for backend-GUI communication. I will either lead the Streamlit GUI implementation (potentially with Gemini's help) or adapt work started by Breezy.

## Technologies Used

* **Core Language:** Python (Project standard: 3.10.x)
* **Data Handling & Analysis:** Pandas, NumPy
* **Machine Learning (AI Components):** Scikit-learn (for K-Means, StandardScaler), TensorFlow/Keras (for LSTM)
* **Data Acquisition (Planned):** MetaTrader 5 (MT5) Python API
* **Plotting/Visualization:** Matplotlib (used extensively in Jupyter Notebooks for testing)
* **Development Environment:** Jupyter Notebooks, VS Code
* **Model Persistence (Planned for AI):** joblib, `.h5`/Keras SavedModel
* **GUI (Planned):** Streamlit
* **Version Control:** Git / GitHub
* **Project Management:** ClickUp
* **AI Coding Partner:** Gemini

## Current Status

Actively developing and iteratively testing the core rules-based logic. D1 market structure analysis and H1 W/M pattern identification functions are implemented and undergoing parameter tuning and visual verification. The initial `signal_engine.py` and `risk_rules.py` have been drafted. Tasks for AI component development (K-Means, LSTM) have been clearly defined for the ML engineer. Planning for GUI integration is underway.

## Challenges & Learnings

* **Python Environment & Dependencies:** Overcame initial `pip install` failures (e.g., for `matplotlib`, `ipykernel`) due to SSL issues by using `--trusted-host` and managing virtual environments carefully, especially after project folder renaming.
* **Module Import Logic:** Debugged `sys.path` manipulation for importing custom modules from the `src` directory into Jupyter Notebooks (located within `src`) and ensured all necessary `__init__.py` files were in place.
* **Pandas DataFrame Operations:** Troubleshot `KeyError`s by meticulously tracing DataFrame modifications and ensuring column propagation through the data processing pipeline.
* **Iterative Pattern Recognition Logic:** Developing Python functions for H1 W/M patterns that accurately reflect visual chart patterns is an iterative process requiring careful parameter tuning and extensive visual testing to balance sensitivity and specificity.
* **Translating Complex Trading Concepts into Code:** Articulating nuanced trading rules (like D1 market structure, specific BoS confirmation, and the "1-2-3 movement") into precise Python logic required detailed discussion and iterative refinement, often with Gemini's help to structure the code (e.g., the `determine_d1_market_structure` function).

## Future Ideas

* Implement and test Head & Shoulders (H&S) / Inverse H&S H1 entry patterns.
* Integrate the K-Means market regime filter from Mthabisi.
* Integrate the LSTM model from Mthabisi for H1 entry confirmation and dynamic Take Profit.
* Build out `main_processor.py` to fully orchestrate the rules and AI.
* Connect to the MT5 API for live data and potential trade execution.
* Conduct comprehensive backtesting with Khastro.
* Further tune all parameters based on backtesting results.
* Complete GUI integration (with Breezy/Kudzmaster or self-developed).
* Explore XGBoost for confidence scoring if time permits.
* Implement more advanced trade management features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
