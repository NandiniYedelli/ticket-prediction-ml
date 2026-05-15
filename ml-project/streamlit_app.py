"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         WaitSure — Streamlit UI  (streamlit_app.py)                          ║
║         Run: streamlit run streamlit_app.py                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

This file is a pure UI wrapper around predict_app.py.
Backend functions (merge_datasets, load_model, predict_ticket) are imported
as-is — nothing in predict_app.py is modified.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

# ── Import backend (unchanged) ────────────────────────────────────────────────
from predict_app import merge_datasets, load_model, predict_ticket, CAT_COLS

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WaitSure — IRCTC Waitlist Predictor",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — IRCTC-inspired modern UI with animations
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

/* ── Root variables ── */
:root {
    --navy:        #1a2744;
    --navy-mid:    #243356;
    --blue:        #1e4fa0;
    --blue-bright: #2563eb;
    --orange:      #f97316;
    --orange-dark: #ea6b0a;
    --sky:         #e8f0fe;
    --white:       #FFFFFF;
    --offwhite:    #f7f9fc;
    --border:      #dce6f0;
    --text:        #1a2744;
    --text-mid:    #4a5568;
    --muted:       #8ba3c1;
    --radius:      14px;

    /* result colours */
    --green:       #16a34a;
    --yellow:      #d97706;
    --red:         #dc2626;
}

/* ── Base ── */
html, body, [class*="css"], .stApp {
    background-color: var(--offwhite) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 0 4rem; max-width: 100%; margin: 0; }

/* ════════════════════════════════════════
   TOP NAV BAR (IRCTC-style dark navy)
   ════════════════════════════════════════ */
.topnav {
    background: linear-gradient(90deg, var(--navy) 0%, var(--navy-mid) 100%);
    padding: 0.6rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 3px solid var(--orange);
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 4px 24px rgba(26,39,68,0.4);
    animation: navSlide 0.5s ease both;
}
@keyframes navSlide {
    from { transform: translateY(-60px); opacity: 0; }
    to   { transform: translateY(0);     opacity: 1; }
}

.topnav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.topnav-logo-icon {
    width: 44px; height: 44px;
    background: var(--orange);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 0 0 3px rgba(249,115,22,0.3);
    animation: logoPop 0.6s 0.2s ease both;
}
@keyframes logoPop {
    from { transform: scale(0.5); opacity: 0; }
    to   { transform: scale(1);   opacity: 1; }
}
.topnav-brand {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 1px;
    line-height: 1;
}
.topnav-brand span {
    color: var(--orange);
}
.topnav-tagline {
    font-size: 0.62rem;
    color: rgba(255,255,255,0.55);
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}

.topnav-right {
    display: flex;
    align-items: center;
    gap: 16px;
}
.nav-badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.75);
    font-size: 0.68rem;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 1.2px;
    font-family: 'Space Mono', monospace;
    backdrop-filter: blur(8px);
}

/* ════════════════════════════════════════
   TAB BAR (like IRCTC Trains/Buses/Flights)
   ════════════════════════════════════════ */
.tab-bar {
    background: var(--white);
    padding: 0 2.5rem;
    display: flex;
    align-items: center;
    gap: 0;
    border-bottom: 2px solid var(--border);
    box-shadow: 0 2px 12px rgba(26,39,68,0.07);
    animation: fadeDown 0.5s 0.1s ease both;
    overflow-x: auto;
}
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.tab-item {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 0.85rem 1.4rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-mid);
    border-bottom: 3px solid transparent;
    margin-bottom: -2px;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.2s ease;
    letter-spacing: 0.3px;
}
.tab-item:hover {
    color: var(--blue-bright);
    background: var(--sky);
}
.tab-item.active {
    color: var(--blue-bright);
    border-bottom: 3px solid var(--blue-bright);
    background: var(--sky);
    border-radius: 4px 4px 0 0;
}
.tab-icon {
    font-size: 1.05rem;
}

/* ════════════════════════════════════════
   HERO BANNER (train image area)
   ════════════════════════════════════════ */
.hero-banner {
    background: linear-gradient(135deg, var(--navy) 0%, #1e3a6e 45%, #2d5a9e 100%);
    padding: 2.2rem 2.5rem 3.8rem;
    position: relative;
    overflow: hidden;
    min-height: 180px;
}

/* Animated mesh orbs */
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -60px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(37,99,235,0.35) 0%, transparent 65%);
    animation: orbFloat 7s ease-in-out infinite alternate;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -120px; left: -80px;
    width: 500px; height: 300px;
    background: radial-gradient(ellipse, rgba(249,115,22,0.15) 0%, transparent 60%);
    animation: orbFloat2 9s ease-in-out infinite alternate;
}
@keyframes orbFloat  { 0% { transform: translate(0,0) scale(1); } 100% { transform: translate(-30px,20px) scale(1.2); } }
@keyframes orbFloat2 { 0% { transform: translate(0,0) scale(1); } 100% { transform: translate(25px,-15px) scale(1.15); } }

/* Animated railway tracks */
.railway-track {
    position: absolute;
    bottom: 24px; left: 0; right: 0;
    display: flex; gap: 22px;
    padding: 0 2.5rem;
    overflow: hidden;
}
.railway-track span {
    width: 36px; height: 4px;
    background: rgba(255,255,255,0.18);
    border-radius: 2px;
    flex-shrink: 0;
    animation: trackPulse 2.2s ease-in-out infinite;
}
.railway-track span:nth-child(3n)   { animation-delay: 0s; }
.railway-track span:nth-child(3n+1) { animation-delay: -0.74s; }
.railway-track span:nth-child(3n+2) { animation-delay: -1.47s; }
@keyframes trackPulse {
    0%,100% { opacity: 0.12; transform: scaleX(1); }
    50%      { opacity: 0.55; transform: scaleX(1.4); }
}

/* Animated train silhouette */
.hero-train {
    position: absolute;
    right: -200px; bottom: 28px;
    font-size: 5rem;
    opacity: 0.12;
    animation: trainRide 18s linear infinite;
    filter: blur(0.5px);
}
@keyframes trainRide {
    0%   { right: -200px; opacity: 0; }
    5%   { opacity: 0.12; }
    90%  { opacity: 0.12; }
    100% { right: 110%; opacity: 0; }
}

.hero-heading {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #fff;
    line-height: 1.05;
    margin: 0;
    letter-spacing: 0.5px;
    animation: fadeSlideUp 0.7s ease both;
    position: relative;
    z-index: 2;
}
.hero-heading span { color: var(--orange); }
.hero-sub {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.65);
    margin-top: 0.4rem;
    letter-spacing: 0.5px;
    animation: fadeSlideUp 0.7s 0.12s ease both;
    position: relative; z-index: 2;
}
.hero-pills {
    display: flex; gap: 8px; flex-wrap: wrap;
    margin-top: 1rem;
    animation: fadeSlideUp 0.7s 0.24s ease both;
    position: relative; z-index: 2;
}
.hero-pill {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.22);
    backdrop-filter: blur(6px);
    color: rgba(255,255,255,0.9);
    font-size: 0.65rem;
    padding: 3px 11px;
    border-radius: 20px;
    letter-spacing: 1.8px;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ════════════════════════════════════════
   MAIN CONTENT AREA
   ════════════════════════════════════════ */
.content-wrap {
    padding: 2rem 2.5rem;
    max-width: 1120px;
    margin: 0 auto;
}

/* ── Info chips row ── */
.chip-row {
    display: flex; gap: 10px; flex-wrap: wrap;
    margin-bottom: 1.5rem;
    animation: fadeSlideUp 0.5s 0.15s ease both;
}
.info-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(30,79,160,0.08);
    border: 1px solid rgba(30,79,160,0.2);
    color: var(--blue);
    font-size: 0.78rem;
    padding: 5px 14px;
    border-radius: 20px;
    font-weight: 500;
    transition: all 0.2s;
    cursor: default;
}
.info-chip:hover {
    background: rgba(30,79,160,0.15);
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(30,79,160,0.15);
}

/* ── Cards ── */
.card {
    background: var(--white);
    border: 1.5px solid #c8daf5;
    border-radius: var(--radius);
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 16px rgba(26,39,68,0.06), inset 0 1px 0 rgba(255,255,255,0.8);
    transition: box-shadow 0.28s ease, transform 0.28s ease, border-color 0.28s ease;
    animation: fadeSlideUp 0.5s ease both;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 5px; height: 100%;
    background: linear-gradient(180deg, var(--blue-bright) 0%, var(--orange) 100%);
    border-radius: 4px 0 0 4px;
    opacity: 0;
    transition: opacity 0.3s;
}
.card:hover {
    box-shadow: 0 10px 40px rgba(37,99,235,0.14), 0 2px 8px rgba(249,115,22,0.08);
    transform: translateY(-3px);
    border-color: #93b8f0;
}
.card:hover::before { opacity: 1; }

.card-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--blue);
    font-weight: 700;
    margin-bottom: 1.1rem;
    padding-bottom: 0.7rem;
    border-bottom: 2px solid var(--sky);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ══════════════════════════════════════════════════════
   NUCLEAR WIDGET OVERRIDE — force light theme on all inputs
   ══════════════════════════════════════════════════════ */

/* Every widget wrapper */
div[data-testid="stSelectbox"],
div[data-testid="stNumberInput"],
div[data-testid="stDateInput"],
div[data-testid="stSlider"] {
    color: #1a2744 !important;
}

/* The inner control box */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div > div,
div[data-testid="stNumberInput"] > div,
div[data-testid="stDateInput"] > div > div,
div[data-testid="stDateInput"] > div {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1.5px solid #b8d0f0 !important;
    border-radius: 10px !important;
    color: #1a2744 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

/* Hover & focus */
div[data-testid="stSelectbox"] > div:hover,
div[data-testid="stNumberInput"] > div:hover,
div[data-testid="stDateInput"] > div:hover {
    border-color: #2563eb !important;
    box-shadow: 0 2px 12px rgba(37,99,235,0.10) !important;
}
div[data-testid="stSelectbox"] > div:focus-within,
div[data-testid="stNumberInput"] > div:focus-within,
div[data-testid="stDateInput"] > div:focus-within {
    border-color: #f97316 !important;
    box-shadow: 0 0 0 3px rgba(249,115,22,0.15) !important;
}

/* ALL descendant text & bg inside widgets */
div[data-testid="stSelectbox"] *,
div[data-testid="stNumberInput"] *,
div[data-testid="stDateInput"] * {
    color: #1a2744 !important;
    background-color: transparent !important;
}

/* Re-apply white bg only on the actual input/control box */
div[data-testid="stSelectbox"] [data-baseweb="select"],
div[data-testid="stNumberInput"] [data-baseweb="input"],
div[data-testid="stDateInput"] [data-baseweb="input"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* The actual text/input elements */
div[data-testid="stSelectbox"] input,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stSelectbox"] span,
div[data-testid="stSelectbox"] p,
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a2744 !important;
    font-weight: 500 !important;
    -webkit-text-fill-color: #1a2744 !important;
}

/* Number input +/- buttons */
div[data-testid="stNumberInput"] button {
    background: #e8f0fe !important;
    background-color: #e8f0fe !important;
    color: #1a2744 !important;
    border: 1px solid #b8d0f0 !important;
    border-radius: 6px !important;
}
div[data-testid="stNumberInput"] button:hover {
    background: #c8daf5 !important;
    background-color: #c8daf5 !important;
}
div[data-testid="stNumberInput"] button svg,
div[data-testid="stNumberInput"] button p {
    color: #1a2744 !important;
    fill: #1a2744 !important;
}

/* Labels */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stSlider"] label {
    color: #4a5568 !important;
    background: transparent !important;
    background-color: transparent !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* Dropdown arrow icon */
div[data-testid="stSelectbox"] svg {
    fill: #1a2744 !important;
    color: #1a2744 !important;
}

/* ════════════════════════════════════════
   CALENDAR / DATE PICKER — NUCLEAR OVERRIDE
   ════════════════════════════════════════ */

/* Popover wrapper — white bg, blue border */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] > div > div {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 2px solid #c8daf5 !important;
    border-radius: 14px !important;
    box-shadow: 0 12px 48px rgba(26,39,68,0.18) !important;
    overflow: visible !important;
    color: #1a2744 !important;
}

/* Calendar root — force every single child to white */
[data-baseweb="calendar"],
[data-baseweb="calendar"] *:not(button) {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a2744 !important;
}

/* Day-of-week headers */
[data-baseweb="calendar"] div[role="columnheader"],
[data-baseweb="calendar"] div[role="columnheader"] * {
    color: #8ba3c1 !important;
    -webkit-text-fill-color: #8ba3c1 !important;
    font-weight: 700 !important;
    font-size: 0.72rem !important;
}

/* All date buttons — default state */
[data-baseweb="calendar"] button,
[data-baseweb="calendar"] [role="gridcell"] button {
    background: transparent !important;
    background-color: transparent !important;
    color: #1a2744 !important;
    -webkit-text-fill-color: #1a2744 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: background 0.15s, transform 0.15s !important;
}

/* Date cell hover */
[data-baseweb="calendar"] [role="gridcell"] button:hover {
    background: #dbeafe !important;
    background-color: #dbeafe !important;
    color: #2563eb !important;
    -webkit-text-fill-color: #2563eb !important;
    transform: scale(1.1) !important;
}

/* Selected date — orange highlight */
[data-baseweb="calendar"] [aria-selected="true"] button,
[data-baseweb="calendar"] button[aria-selected="true"] {
    background: #f97316 !important;
    background-color: #f97316 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    box-shadow: 0 3px 12px rgba(249,115,22,0.45) !important;
    border-radius: 8px !important;
}

/* Today — blue outline */
[data-baseweb="calendar"] button[data-today="true"] {
    border: 2px solid #2563eb !important;
    font-weight: 700 !important;
}

/* Prev / Next arrows */
[data-baseweb="calendar"] button[aria-label="Go to previous month"],
[data-baseweb="calendar"] button[aria-label="Go to next month"] {
    background: #e8f0fe !important;
    background-color: #e8f0fe !important;
    color: #1a2744 !important;
    -webkit-text-fill-color: #1a2744 !important;
    border-radius: 8px !important;
}
[data-baseweb="calendar"] button[aria-label="Go to previous month"]:hover,
[data-baseweb="calendar"] button[aria-label="Go to next month"]:hover {
    background: #bfdbfe !important;
    background-color: #bfdbfe !important;
}

/* Month/Year selects inside calendar */
[data-baseweb="calendar"] [data-baseweb="select"] > div,
[data-baseweb="calendar"] [data-baseweb="select"] * {
    background: #f0f6ff !important;
    background-color: #f0f6ff !important;
    color: #1a2744 !important;
    -webkit-text-fill-color: #1a2744 !important;
    border-color: #c8daf5 !important;
}

/* Dropdown option lists (month / year) */
[data-baseweb="menu"],
[data-baseweb="menu"] ul,
[data-baseweb="menu"] li,
[data-baseweb="menu"] * {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a2744 !important;
    -webkit-text-fill-color: #1a2744 !important;
}
[data-baseweb="menu"] {
    border: 1.5px solid #c8daf5 !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 32px rgba(26,39,68,0.15) !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [aria-selected="true"] {
    background: #e8f0fe !important;
    background-color: #e8f0fe !important;
    color: #2563eb !important;
    -webkit-text-fill-color: #2563eb !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: var(--orange) !important;
    border-color: var(--orange) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:first-child {
    background: var(--sky) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:nth-child(2) {
    background: var(--orange) !important;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--orange) 0%, var(--orange-dark) 100%) !important;
    color: #fff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2.8rem !important;
    cursor: pointer !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(249,115,22,0.4) !important;
    transition: all 0.25s cubic-bezier(.4,0,.2,1) !important;
    position: relative !important;
    overflow: hidden !important;
    text-transform: uppercase !important;
}
.stButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
    transition: left 0.4s ease;
}
.stButton > button:hover {
    box-shadow: 0 8px 32px rgba(249,115,22,0.55) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:hover::before { left: 160%; }
.stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
}

/* ════════════════════════════════════════
   RESULT PANEL
   ════════════════════════════════════════ */
.result-panel {
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
    animation: panelReveal 0.65s cubic-bezier(.4,0,.2,1) both;
    border: 2px solid transparent;
}
@keyframes panelReveal {
    from { opacity: 0; transform: translateY(24px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0)    scale(1); }
}
.result-green  {
    background: linear-gradient(135deg, rgba(22,163,74,0.05) 0%, rgba(232,240,254,0.5) 100%);
    border-color: rgba(22,163,74,0.35);
}
.result-yellow {
    background: linear-gradient(135deg, rgba(217,119,6,0.05) 0%, rgba(255,251,235,0.5) 100%);
    border-color: rgba(217,119,6,0.35);
}
.result-red {
    background: linear-gradient(135deg, rgba(220,38,38,0.05) 0%, rgba(254,242,242,0.5) 100%);
    border-color: rgba(220,38,38,0.35);
}

/* shimmer sweep */
.result-panel::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent);
    animation: shimmerSweep 1.3s 0.3s ease forwards;
    pointer-events: none;
}
@keyframes shimmerSweep { to { left: 160%; } }

.result-prob {
    font-family: 'Rajdhani', sans-serif;
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.2rem;
    animation: countUp 0.9s cubic-bezier(.4,0,.2,1) both;
    letter-spacing: -1px;
}
@keyframes countUp {
    from { opacity: 0; transform: scale(0.6) translateY(10px); }
    to   { opacity: 1; transform: scale(1)   translateY(0); }
}
.verdict-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 8px;
}
.advice-box {
    background: rgba(232,240,254,0.6);
    border-left: 4px solid var(--blue-bright);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.1rem;
    font-size: 0.88rem;
    color: var(--text-mid);
    margin-top: 1rem;
    backdrop-filter: blur(4px);
}

/* ── Progress bar ── */
.progress-track {
    background: var(--sky);
    border-radius: 99px;
    height: 10px;
    width: 100%;
    margin-top: 0.6rem;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 99px;
    animation: progressGrow 1.1s 0.4s cubic-bezier(.4,0,.2,1) both;
}
@keyframes progressGrow { from { width: 0% !important; } }

/* ── Ticket summary grid ── */
.ticket-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.3rem;
}
.ticket-item {
    background: rgba(232,240,254,0.5);
    border: 1.5px solid #c8daf5;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    transition: all 0.22s ease;
    cursor: default;
}
.ticket-item:hover {
    background: rgba(37,99,235,0.10);
    border-color: #93b8f0;
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(37,99,235,0.12);
}
.ticket-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
}
.ticket-value {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    margin-top: 2px;
}

/* ── Status dot ── */
.status-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse 2s ease-in-out infinite;
}
.dot-green  { background: var(--green); }
.dot-yellow { background: var(--yellow); }
.dot-red    { background: var(--red); }
@keyframes pulse {
    0%,100% { transform: scale(1);    box-shadow: 0 0 0 0 currentColor; }
    50%     { transform: scale(1.15); box-shadow: 0 0 0 5px transparent; }
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 2px 12px rgba(26,39,68,0.06) !important;
    transition: box-shadow 0.2s, transform 0.2s !important;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 6px 24px rgba(26,39,68,0.12) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetric"] label {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: var(--navy) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}

/* ── Days badge ── */
.days-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    color: var(--blue-bright);
    background: rgba(37,99,235,0.08);
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: 8px;
    padding: 4px 10px;
    margin-top: 4px;
    font-weight: 500;
}
.days-badge strong {
    color: var(--navy);
    font-family: 'Space Mono', monospace;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--orange) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Selectbox inner text color ── */
div[data-testid="stSelectbox"] [data-baseweb="select"] div,
div[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #1a2744 !important;
    -webkit-text-fill-color: #1a2744 !important;
    background: transparent !important;
}

/* ── Section fade-in stagger ── */
.card:nth-child(1) { animation-delay: 0.05s; }
.card:nth-child(2) { animation-delay: 0.12s; }
.card:nth-child(3) { animation-delay: 0.19s; }

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 3rem;
    padding: 1.5rem;
    color: var(--muted);
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1.2px;
    border-top: 1px solid var(--border);
}

/* ── Floating background orb ── */
.bg-orb {
    position: fixed;
    bottom: -100px; right: -100px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(37,99,235,0.05) 0%, transparent 70%);
    pointer-events: none;
    z-index: -1;
    animation: orbFloat 12s ease-in-out infinite alternate;
}
</style>

<!-- Floating background orb -->
<div class="bg-orb"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED BACKEND LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_data():
    return merge_datasets()

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


# ─────────────────────────────────────────────────────────────────────────────
# TOP NAV BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topnav">
  <div class="topnav-logo">
    <div class="topnav-logo-icon">🚂</div>
    <div>
      <div class="topnav-brand">Wait<span>Sure</span></div>
      <div class="topnav-tagline">IRCTC Waitlist Predictor</div>
    </div>
  </div>
  <div class="topnav-right">
    <span class="nav-badge">ML-POWERED</span>
    <span class="nav-badge">v1.0</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB BAR  (only the tabs relevant to this app)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tab-bar">
  <div class="tab-item active"><span class="tab-icon">🎫</span> Predict Waitlist</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-heading">Indian Railways<br><span>Waitlist Confirmation</span></div>
  <div class="hero-sub">Safety &nbsp;|&nbsp; Security &nbsp;|&nbsp; Punctuality &nbsp;·&nbsp; Powered by Machine Learning</div>
  <div class="hero-pills">
    <span class="hero-pill">ML-Powered</span>
    <span class="hero-pill">Real-Time</span>
    <span class="hero-pill">IRCTC Data</span>
  </div>
  <div class="hero-train">🚄</div>
  <div class="railway-track">
    <span></span><span></span><span></span><span></span><span></span>
    <span></span><span></span><span></span><span></span><span></span>
    <span></span><span></span><span></span><span></span><span></span>
    <span></span><span></span><span></span><span></span><span></span>
    <span></span><span></span><span></span><span></span><span></span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL (with status)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading datasets and model…"):
    try:
        df_merged = get_data()
        model, scaler, encoders, feat_names = get_model()
        load_ok = True
    except Exception as e:
        load_ok = False
        st.error(f"⚠️  Failed to load backend: {e}")

if not load_ok:
    st.stop()

# Derive dropdown options from merged data
train_nos   = sorted([int(x) for x in df_merged["train_no"].dropna().unique()])
train_types = sorted(df_merged["train_type"].dropna().unique().tolist())
stations    = sorted(set(
    df_merged["source_station"].dropna().unique().tolist() +
    df_merged["destination"].dropna().unique().tolist()
))
classes  = sorted(df_merged["travel_class"].dropna().unique().tolist())
quotas   = sorted(df_merged["quota"].dropna().unique().tolist())
seasons  = sorted(df_merged["season"].dropna().unique().tolist())

# ─────────────────────────────────────────────────────────────────────────────
# DATASET SUMMARY CHIPS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

col_chip1, col_chip2, col_chip3 = st.columns([1,1,2])
with col_chip1:
    st.markdown(f'<div class="info-chip">📊 {df_merged.shape[0]:,} records merged</div>', unsafe_allow_html=True)
with col_chip2:
    conf_rate = int(df_merged["confirmed"].mean()*100)
    st.markdown(f'<div class="info-chip">✅ {conf_rate}% historical confirmation rate</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">🎫 &nbsp;Ticket Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    train_no   = st.selectbox("Train Number",       train_nos,   index=min(4, len(train_nos)-1))
    train_type = st.selectbox("Train Type",          train_types, index=0)
    source     = st.selectbox("Source Station",      stations,    index=min(4, len(stations)-1))
    dest       = st.selectbox("Destination Station", stations,    index=min(5, len(stations)-1))

with col2:
    travel_class = st.selectbox("Travel Class", classes, index=0)
    quota        = st.selectbox("Quota",        quotas,  index=0)
    season       = st.selectbox("Season",       seasons, index=0)
    st.write("")  # spacing

st.markdown('</div>', unsafe_allow_html=True)

# Dates + numeric inputs
st.markdown('<div class="card"><div class="card-title">📅 &nbsp;Journey & Waitlist Info</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    journey_date = st.date_input("Journey Date",  value=pd.Timestamp("2025-06-15"))
    booking_date = st.date_input("Booking Date",  value=pd.Timestamp("2025-01-01"))
    # Auto-compute days before travel
    dbt = max(0, (pd.Timestamp(journey_date) - pd.Timestamp(booking_date)).days)
    st.markdown(
        f'<div class="days-badge">⏱ Days before travel: <strong>{dbt}</strong></div>',
        unsafe_allow_html=True
    )

with col4:
    waitlist_num = st.number_input("Waitlist Number (e.g. 14)",      min_value=1,  max_value=500, value=14,  step=1)
    total_seats  = st.number_input("Total Seats in Coach (e.g. 72)", min_value=10, max_value=500, value=72,  step=1)

col5, col6 = st.columns(2)
with col5:
    hist_rate    = st.slider("Historical Confirm Rate", min_value=0.0, max_value=1.0, value=0.72, step=0.01,
                              format="%.2f")
with col6:
    cancel_trend = st.slider("Cancellation Trend (avg cancellations)", min_value=0, max_value=200, value=10, step=1)

st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button("⚡  Search Train / Predict Confirmation")


# ─────────────────────────────────────────────────────────────────────────────
# RUN PREDICTION & DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:
    ticket = {
        "train_no":                int(train_no),
        "train_type":              train_type,
        "source_station":          source,
        "destination":             dest,
        "journey_date":            str(journey_date),
        "booking_date":            str(booking_date),
        "days_before_travel":      dbt,
        "travel_class":            travel_class,
        "quota":                   quota,
        "season":                  season,
        "waitlist_number":         int(waitlist_num),
        "total_seats":             int(total_seats),
        "historical_confirm_rate": float(hist_rate),
        "cancellation_trend":      float(cancel_trend),
    }

    with st.spinner("Running WaitSure model…"):
        result = predict_ticket(ticket, model, scaler, encoders, feat_names)

    prob     = result["probability"]
    prob_pct = f"{prob:.1%}"

    # Determine colour class
    if prob >= 0.75:
        panel_cls   = "result-green"
        dot_cls     = "dot-green"
        bar_color   = "#16a34a"
        emoji       = "✅"
        verdict_lbl = "LIKELY CONFIRMED"
    elif prob >= 0.50:
        panel_cls   = "result-yellow"
        dot_cls     = "dot-yellow"
        bar_color   = "#d97706"
        emoji       = "⚠️"
        verdict_lbl = "POSSIBLE CONFIRMATION"
    else:
        panel_cls   = "result-red"
        dot_cls     = "dot-red"
        bar_color   = "#dc2626"
        emoji       = "❌"
        verdict_lbl = "UNLIKELY TO CONFIRM"

    bar_width    = int(prob * 100)
    advice_clean = result["advice"]

    # Ticket summary grid
    ticket_html = f"""
    <div class="ticket-grid">
      <div class="ticket-item"><div class="ticket-label">Train</div><div class="ticket-value">{ticket['train_no']} · {ticket['train_type']}</div></div>
      <div class="ticket-item"><div class="ticket-label">Route</div><div class="ticket-value">{ticket['source_station']} → {ticket['destination']}</div></div>
      <div class="ticket-item"><div class="ticket-label">Class / Quota</div><div class="ticket-value">{ticket['travel_class']} / {ticket['quota']}</div></div>
      <div class="ticket-item"><div class="ticket-label">Journey Date</div><div class="ticket-value">{ticket['journey_date']}</div></div>
      <div class="ticket-item"><div class="ticket-label">WL Position</div><div class="ticket-value">WL/{ticket['waitlist_number']}</div></div>
      <div class="ticket-item"><div class="ticket-label">Days Ahead</div><div class="ticket-value">{ticket['days_before_travel']} days</div></div>
    </div>
    """

    result_html = f"""
    <div class="result-panel {panel_cls}">
      {ticket_html}
      <div class="result-prob" style="color:{bar_color}">{prob_pct}</div>
      <div class="verdict-text">
        <span class="status-dot {dot_cls}"></span>
        {emoji} &nbsp;{verdict_lbl}
      </div>
      <div class="progress-track">
        <div class="progress-fill" style="width:{bar_width}%; background:{bar_color};"></div>
      </div>
      <div class="advice-box">💡 {advice_clean}</div>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)

    # Raw probability metric as a native Streamlit metric (optional extra)
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Confirmation Probability", prob_pct)
    m2.metric("WL Position",              f"WL/{waitlist_num}")
    m3.metric("Days Before Travel",       f"{dbt} days")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  WaitSure v1.0 &nbsp;·&nbsp; ML-Powered IRCTC Predictor &nbsp;·&nbsp; For informational use only
</div>
""", unsafe_allow_html=True)
