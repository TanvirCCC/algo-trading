# MT4 EA Setup — LnterqoV3

## Step 1 — Copy the EA file
1. In MT4: File → Open Data Folder
2. Navigate to `MQL4/Experts/`
3. Copy `LnterqoV3.mq4` into that folder
4. Back in MT4: press F5 (or Navigator → Experts → right-click → Refresh)

## Step 2 — Find your bridge folder path
1. In MT4: File → Open Data Folder
2. Note the full path — it will be something like:
   `C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\XXXXXXXX\`
3. Open `live/config.py` and update `MT4_FILES_DIR` to match

## Step 3 — Enable Expert Advisors
1. MT4 → Tools → Options → Expert Advisors tab
2. Tick: "Allow automated trading"
3. Tick: "Allow DLL imports"

## Step 4 — Attach EA to Gold chart
1. Open a Gold M5 chart
2. Drag `LnterqoV3` from Navigator onto the chart
3. Settings:
   - `InpSymbol` = whatever Gold is called in your Market Watch
     (right-click Market Watch → check the exact name)
   - `InpLiveTrading` = **false** to start (paper mode)
   - `InpMagicNumber` = 20260001
4. Click OK — smiley face should appear top-right of chart

## Step 5 — Verify data export
After ~60 seconds, check your bridge folder for:
- `gold_5m_live.csv`  ← 5m bars
- `gold_d1_live.csv`  ← daily bars
- `lnterqo_status.csv` ← EA heartbeat

## Step 6 — Start the signal engine
```bash
cd /path/to/algo-trading
python live/signal_engine.py
```

## Step 7 — Open the dashboard
```bash
streamlit run dashboard/app.py
```
Opens at http://localhost:8501

## Step 8 — Go live
Once you've verified signals look correct in paper mode:
1. In MT4 EA settings: set `InpLiveTrading = true`
2. Monitor the dashboard

## Gold symbol on CMC Markets
Common names to check in Market Watch:
- `Gold`
- `XAUUSD`
- `XAUUSDm`
Check yours: right-click any instrument in Market Watch → Properties
