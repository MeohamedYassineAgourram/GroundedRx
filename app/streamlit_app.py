"""Demo UI (Phase 7). STUB -- built after Phases 2/3/5/6 (UI is first to cut).

Planned:
  - Left: patient list from eval_cases.json; click to load.
  - Center: grounded aftercare plan -- danger-signs (red), meds, lifestyle -- each line with
    its citation id; badge "generated offline on a 4B model."
  - Bottom: headline numbers + a button revealing the ablation table and frontier PNG.
  - Demo move: airplane mode -> generate live -> point at citations -> flip to evidence.

Run (once built): streamlit run app/streamlit_app.py
"""
from __future__ import annotations

try:
    import streamlit as st

    st.set_page_config(page_title="GroundedRx", page_icon="💊")
    st.title("GroundedRx")
    st.info("Demo UI is a Phase 7 stub. Build after the evidence harness (Phases 2/3/5/6) is done.")
except ImportError:
    print("streamlit not installed yet; UI is a Phase 7 stub.")
