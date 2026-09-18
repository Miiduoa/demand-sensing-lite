"""Streamlit demo for Demand Sensing Lite."""

from __future__ import annotations

import streamlit as st

from demand_sensing_lite.pipeline import run_demo

st.set_page_config(page_title="Demand Sensing Lite", layout="wide")
st.title("Demand Sensing Lite｜多 SKU 需求感知＋補貨模擬")
st.caption("合成資料｜sklearn 時間切分預測｜(s,S) vs 天真基線 — 備審作品示範")

with st.sidebar:
    st.header("參數")
    n_days = st.slider("模擬天數", 180, 730, 365, 30)
    test_days = st.slider("Hold-out 天數", 28, 90, 56, 7)
    seed = st.number_input("隨機種子", 0, 9999, 42)
    lead_time = st.slider("前置時間 (天)", 1, 10, 3)
    doc = st.slider("Days of Cover", 3.0, 14.0, 7.0, 0.5)
    run = st.button("執行模擬", type="primary")

if run or "result" not in st.session_state:
    with st.spinner("產生資料、訓練、模擬中…"):
        st.session_state["result"] = run_demo(
            n_days=n_days,
            test_days=test_days,
            seed=int(seed),
            lead_time=lead_time,
            days_of_cover=doc,
        )

r = st.session_state["result"]
m = r["forecast_metrics"]
inv = r["inventory"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hold-out MAE", f"{m['mae']:.2f}")
c2.metric("Hold-out MAPE", f"{m['mape']:.1f}%")
c3.metric("(s,S) 平均服務水準", f"{inv['ss_service_level'].mean():.1%}")
c4.metric("Naive 平均服務水準", f"{inv['naive_service_level'].mean():.1%}")

st.subheader("庫存政策比較")
st.dataframe(inv, use_container_width=True)

st.subheader("預測樣本（Hold-out）")
st.dataframe(r["predictions"].head(40), use_container_width=True)

st.info(
    "說明：資料為合成日需求（可重現種子）。預測採時間切分 HistGradientBoosting；"
    "補貨以預測驅動的 (s,S)／Days-of-Cover 對照固定門檻天真基線。"
)
