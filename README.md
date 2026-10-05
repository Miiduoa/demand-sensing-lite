# Demand Sensing Lite｜多 SKU 需求感知＋補貨模擬

[![verify](https://github.com/Miiduoa/demand-sensing-lite/actions/workflows/verify.yml/badge.svg)](https://github.com/Miiduoa/demand-sensing-lite/actions/workflows/verify.yml)

> 作者：顧晉瑋（靜宜大學 資訊管理學系）｜備審作品集  
> 授權：MIT｜資料：**合成／可重現種子**，非真實銷售資料

---

## 備審向說明：問題 → 方法 → 系統 → 結果

### 問題（Why）

零售／電商常面對「多品項、日波動」的需求：預測誤差會直接造成**缺貨**或**庫存過高**。  
本專題以精簡可跑通的管線，示範如何把「需求感知」接到「補貨決策」，並用量化指標比較政策。

### 方法（How）

1. **合成日需求產生器**（`seed` 可重現）：多 SKU、週／月季節、輕微趨勢、偶發促銷尖峰。  
2. **時間切分預測**：特徵含星期、月份、促銷旗標、lag／滾動統計；模型用 `HistGradientBoostingRegressor`（sklearn）。  
3. **補貨模擬**：預測驅動的 **(s,S)／Days-of-Cover** 對照「固定門檻天真基線」；基線需求與初始庫存只由訓練區間估計，避免測試資料提前進入政策設定。

### 系統（What you can run）

| 入口 | 指令 |
|------|------|
| CLI 一鍵示範 | `python -m demand_sensing_lite` |
| Streamlit 儀表板 | `streamlit run app.py` |

### 結果（What to look at）

- Hold-out **MAE / MAPE**（時間切分，避免隨機 shuffle 洩漏）。  
- 各 SKU 的 **服務水準、平均庫存、下單次數**：(s,S) vs naive。  
- 典型觀察（種子=42、預設參數）：預測驅動政策在相近或更低庫存下，服務水準往往優於天真基線（實際數字以本機執行輸出為準）。

---

## 快速開始

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# CLI
python -m demand_sensing_lite

# 可選 UI
streamlit run app.py
```

### 主要參數（CLI）

```bash
python -m demand_sensing_lite --days 365 --test-days 56 --seed 42 --lead-time 3 --doc 7
```

---

## 專案結構

```
demand-sensing-lite/
├── app.py                      # Streamlit
├── demand_sensing_lite/
│   ├── __main__.py             # CLI
│   ├── generate.py             # 合成需求
│   ├── forecast.py             # 特徵 + 時間切分預測
│   ├── inventory.py            # (s,S) / naive 模擬
│   └── pipeline.py             # 端到端
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 學習重點（對準資管）

- 時間序列特徵工程與**嚴格時間切分**評估  
- 預測誤差如何影響營運 KPI（服務水準／庫存）  
- 決策支援思維：模型輸出 → 可解釋的補貨規則 → 模擬比較

## 誠實聲明

- 本 repo **不宣稱競賽得獎或業界部署成績**。  
- 數值來自合成資料與簡化假設（固定前置時間、無產能上限等），僅供學習與備審展示。
