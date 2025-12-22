import streamlit as st
import pandas as pd
import fundamentus
import plotly.graph_objects as go
import plotly.express as px
import feedparser
import yfinance as yf
import datetime
from datetime import timedelta
from time import mktime

# --- 1. Configuração da Página (SEO) ---
st.set_page_config(
    page_title="Melhores Ações Ibovespa 2025 | Ranking Fundamentalista e Dividendos",
    layout="wide",
    page_icon="🇧🇷"
)

# --- CSS Global ---
st.markdown("""
    <style>
    /* Cabeçalhos à direita */
    [data-testid="stDataFrame"] table tr th { text-align: right !important; }
    /* Células à esquerda */
    [data-testid="stDataFrame"] table tr td { text-align: left !important; }
    </style>
""", unsafe_allow_html=True)

# --- Estado ---
if 'expander_open' not in st.session_state: st.session_state.expander_open = True
def close_expander(): st.session_state.expander_open = False

# --- Auxiliares ---
def clean_fundamentus_col(x):
    if pd.isna(x) or x == '': return 0.0
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        x = x.strip()
        if x.endswith('%'):
            x = x.replace('%', '').replace('.', '').replace(',', '.')
            try: return float(x) / 100
            except: return 0.0
        x = x.replace('.', '').replace(',', '.')
        try: return float(x)
        except: return 0.0
    return 0.0

def format_short_number(val):
    if pd.isna(val) or val == 0: return ""
    abs_val = abs(val)
    if abs_val >= 1e9: return f"{val/1e9:.1f}B"
    elif abs_val >= 1e6: return f"{val/1e6:.0f}M"
    return f"{val:.0f}"

def get_current_data():
    now = datetime.datetime.now()
    return now.strftime("%B"), now.year

# --- Dados Principais ---
@st.cache_data(ttl=3600*6)
def get_ranking_data():
    try:
        df = fundamentus.get_resultado()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'papel'}, inplace=True)
        cols = ['pl', 'roe', 'dy', 'evebit', 'cotacao', 'liq2m', 'mrgliq', 'divbpatr', 'c5y']
        for col in cols:
            if col in df
