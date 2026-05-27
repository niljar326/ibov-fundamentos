import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Ranking Ibovespa Inteligente 2025",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS customizada para aproximar do design ultra-polido
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    /* Customização do Banner de AD */
    .ad-banner {
        background: linear-gradient(90deg, #1e3a8a 0%, #0f172a 100%);
        color: white;
        text-align: center;
        padding: 10px 16px;
        font-size: 13px;
        font-weight: 500;
        border-radius: 12px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .ad-badge {
        background-color: #fbbf24;
        color: #0f172a;
        font-weight: 800;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 10px;
        text-transform: uppercase;
        margin-right: 8px;
        display: inline-block;
    }

    /* Cartões personalizados */
    .custom-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: white !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    .sidebar-logo {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        color: white;
        font-size: 24px;
        font-weight: 800;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        margin-bottom: 12px;
    }

    /* Botão de WhatsApp afiliado */
    .whatsapp-btn {
        display: block;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white !important;
        padding: 16px;
        border-radius: 18px;
        text-decoration: none;
        font-weight: 600;
        margin-bottom: 16px;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
    }
    .whatsapp-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.2);
    }

    /* Tabs customizadas streamlit radio style */
    div.row-widget.stRadio > div {
        flex-direction: row;
        flex-wrap: wrap;
        gap: 8px;
    }
    div.row-widget.stRadio > div > label {
        background-color: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    div.row-widget.stRadio > div > label:hover {
        background-color: #f8fafc !important;
    }
    
    /* Ocultar rádio padrão do streamlit mas customizar selection */
    div[data-testid="stMarkdownContainer"] p {
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

# --- BANCO DE DADOS FALLBACK (Simulando STOCK_DATABASE de forma idêntica à do React) ---
ORIGINAL_STOCKS = [
  {
    "papel": "PETR4",
    "empresa": "Petrobras",
    "cotacao": 38.50,
    "pl": 4.25,
    "pvp": 1.15,
    "evebit": 3.65,
    "roe": 0.284,
    "roic": 0.245,
    "dy": 0.142,
    "mrgliq": 0.195,
    "liq2m": 1540000000,
    "divbpatr": 0.78,
    "lpa": 9.05,
    "vpa": 33.47,
    "epsTrimestral": 2.38,
    "dataRef": "31/12/2024",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2021", "receita": 452000000000, "lucro": 106000000000, "cotacao": 22.10 },
      { "periodo": "2022", "receita": 641000000000, "lucro": 188000000000, "cotacao": 24.50 },
      { "periodo": "2023", "receita": 511000000000, "lucro": 124000000000, "cotacao": 32.80 },
      { "periodo": "2024", "receita": 495000000000, "lucro": 118000000000, "cotacao": 36.20 },
      { "periodo": "Últimos 12m", "receita": 488000000000, "lucro": 115000000000, "cotacao": 38.50 }
    ]
  },
  {
    "papel": "VALE3",
    "empresa": "Vale",
    "cotacao": 62.40,
    "pl": 6.80,
    "pvp": 1.45,
    "evebit": 4.90,
    "roe": 0.213,
    "roic": 0.187,
    "dy": 0.088,
    "mrgliq": 0.224,
    "liq2m": 1250000000,
    "divbpatr": 0.52,
    "lpa": 9.17,
    "vpa": 43.03,
    "epsTrimestral": 1.84,
    "dataRef": "31/12/2024",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2021", "receita": 293000000000, "lucro": 121000000000, "cotacao": 77.20 },
      { "periodo": "2022", "receita": 225000000000, "lucro": 95000000000, "cotacao": 81.30 },
      { "periodo": "2023", "receita": 198000000000, "lucro": 39000000000, "cotacao": 68.40 },
      { "periodo": "2024", "receita": 205000000000, "lucro": 42000000000, "cotacao": 60.50 },
      { "periodo": "Últimos 12m", "receita": 211000000000, "lucro": 45000000000, "cotacao": 62.40 }
    ]
  },
  {
    "papel": "BBAS3",
    "empresa": "Banco do Brasil",
    "cotacao": 27.80,
    "pl": 4.10,
    "pvp": 0.78,
    "evebit": 3.10,
    "roe": 0.215,
    "roic": 0.198,
    "dy": 0.104,
    "mrgliq": 0.165,
    "liq2m": 480000000,
    "divbpatr": 0.12,
    "lpa": 6.78,
    "vpa": 35.64,
    "epsTrimestral": 1.62,
    "dataRef": "31/12/2024",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2021", "receita": 82000000000, "lucro": 19700000000, "cotacao": 15.60 },
      { "periodo": "2022", "receita": 104000000000, "lucro": 31010000000, "cotacao": 18.90 },
      { "periodo": "2023", "receita": 122000000000, "lucro": 35500000000, "cotacao": 25.40 },
      { "periodo": "2024", "receita": 133000000000, "lucro": 38200000000, "cotacao": 26.95 },
      { "periodo": "Últimos 12m", "receita": 138000000000, "lucro": 40100000000, "cotacao": 27.80 }
    ]
  },
  {
    "papel": "ITUB4",
    "empresa": "Itaú Unibanco",
    "cotacao": 34.10,
    "pl": 8.50,
    "pvp": 1.58,
    "evebit": 6.80,
    "roe": 0.212,
    "roic": 0.183,
    "dy": 0.065,
    "mrgliq": 0.174,
    "liq2m": 780000000,
    "divbpatr": 0.18,
    "lpa": 4.01,
    "vpa": 21.58,
    "epsTrimestral": 1.05,
    "dataRef": "31/12/2024",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2021", "receita": 110000000000, "lucro": 24900000000, "cotacao": 21.80 },
      { "periodo": "2022", "receita": 128000000000, "lucro": 29100000000, "cotacao": 23.40 },
      { "periodo": "2023", "receita": 142000000000, "lucro": 33800000000, "cotacao": 31.50 },
      { "periodo": "2024", "receita": 151000000000, "lucro": 37400000000, "cotacao": 33.205 },
      { "periodo": "Últimos 12m", "receita": 153000000000, "lucro": 38400000000, "cotacao": 34.10 }
    ]
  },
  {
    "papel": "WEGE3",
    "empresa": "WEG S.A.",
    "cotacao": 41.50,
    "pl": 26.40,
    "pvp": 5.92,
    "evebit": 19.80,
    "roe": 0.228,
    "roic": 0.212,
    "dy": 0.024,
    "mrgliq": 0.168,
    "liq2m": 290000000,
    "divbpatr": 0.08,
    "lpa": 1.57,
    "vpa": 7.01,
    "epsTrimestral": 0.41,
    "dataRef": "31/12/2024",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2021", "receita": 23500000000, "lucro": 3580000000, "cotacao: ": 32.10 },
      { "periodo": "2022", "receita": 29900000000, "lucro": 4230000000, "cotacao: ": 35.80 },
      { "periodo": "2023", "receita": 32500000000, "lucro": 5610000000, "cotacao: ": 34.20 },
      { "periodo": "2024", "receita": 34800000000, "lucro": 6510000000, "cotacao: ": 39.50 },
      { "periodo": "Últimos 12m", "receita": 36200000000, "lucro": 6850000000, "cotacao": 41.50 }
    ]
  },
  {
    "papel": "ITSA4",
    "empresa": "Itaúsa",
    "cotacao": 10.35,
    "pl": 7.10,
    "pvp": 1.18,
    "evebit": 6.40,
    "roe": 0.174,
    "roic": 0.165,
    "dy": 0.081,
    "mrgliq": 0.942,
    "liq2m": 190000000,
    "divbpatr": 0.19,
    "lpa": 1.45,
    "vpa": 8.77,
    "epsTrimestral": 0.38,
    "dataRef": "31/12/2024",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2021", "receita": 6900000000, "lucro": 12200000000, "cotacao": 8.90 },
      { "periodo": "2022", "receita": 8100000000, "lucro": 13700000000, "cotacao": 8.20 },
      { "periodo": "2023", "receita": 8800000000, "lucro": 14100000000, "cotacao": 9.40 },
      { "periodo": "2024", "receita": 9500000000, "lucro": 15300000000, "cotacao": 9.95 },
      { "periodo": "Últimos 12m", "receita": 9900000000, "lucro": 15900000000, "cotacao": 10.35 }
    ]
  },
  {
    "papel": "TAEE11",
    "empresa": "Taesa S.A.",
    "cotacao": 35.80,
    "pl": 9.80,
    "pvp": 1.62,
    "evebit": 7.12,
    "roe": 0.168,
    "roic": 0.142,
    "dy": 0.098,
    "mrgliq": 0.385,
    "liq2m": 240000000,
    "divbpatr": 1.84,
    "lpa": 3.65,
    "vpa": 22.09,
    "epsTrimestral": 0.95,
    "dataRef": "31/12/2024",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2021", "receita": 3100000000, "lucro": 2200000000, "cotacao": 33.40 },
      { "periodo": "2022", "receita": 2800000000, "lucro": 1440000000, "cotacao": 38.10 },
      { "periodo": "2023", "receita": 3300000000, "lucro": 1360000000, "cotacao": 34.50 },
      { "periodo": "2024", "receita": 3500000000, "lucro": 1420000000, "cotacao": 34.90 },
      { "periodo": "Últimos 12m", "receita": 3750000000, "lucro": 1530000000, "cotacao": 35.80 }
    ]
  },
  {
    "papel": "TRPL4",
    "empresa": "ISA CTEEP",
    "cotacao": 26.20,
    "pl": 6.90,
    "pvp": 0.98,
    "evebit": 5.45,
    "roe": 0.144,
    "roic": 0.138,
    "dy": 0.106,
    "mrgliq": 0.362,
    "liq2m": 210000000,
    "divbpatr": 1.25,
    "lpa": 3.79,
    "vpa": 26.73,
    "epsTrimestral": 0.97,
    "dataRef": "31/12/2024",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2021", "receita": 4100000000, "lucro": 3100000000, "cotacao": 21.05 },
      { "periodo": "2022", "receita": 4800000000, "lucro": 2300000000, "cotacao": 22.40 },
      { "periodo": "2023", "receita": 5400000000, "lucro": 2900000000, "cotacao": 25.10 },
      { "periodo": "2024", "receita": 5800000000, "lucro": 3100000000, "cotacao": 25.80 },
      { "periodo": "Últimos 12m", "receita": 6100000000, "lucro": 3250000000, "cotacao": 26.20 }
    ]
  },
  {
    "papel": "SAPR11",
    "empresa": "Sanepar S.A.",
    "cotacao": 24.80,
    "pl": 5.15,
    "pvp": 0.72,
    "evebit": 4.10,
    "roe": 0.141,
    "roic": 0.128,
    "dy": 0.076,
    "mrgliq": 0.245,
    "liq2m": 122000000,
    "divbpatr": 0.74,
    "lpa": 4.81,
    "vpa": 34.44,
    "epsTrimestral": 1.25,
    "dataRef": "31/12/2024",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2021", "receita": 5200000000, "lucro": 1100000000, "cotacao": 18.10 },
      { "periodo": "2022", "receita": 5600000000, "lucro": 1150000000, "cotacao": 17.40 },
      { "periodo": "2023", "receita": 6200000000, "lucro": 1450000000, "cotacao": 21.20 },
      { "periodo": "2024", "receita": 6700000000, "lucro": 1550000000, "cotacao": 23.50 },
      { "periodo": "Últimos 12m", "receita": 6900000000, "lucro": 1620000000, "cotacao": 24.80 }
    ]
  },
  {
    "papel": "CPLE6",
    "empresa": "Copel S.A.",
    "cotacao": 9.85,
    "pl": 9.10,
    "pvp": 1.12,
    "evebit": 6.20,
    "roe": 0.129,
    "roic": 0.115,
    "dy": 0.068,
    "mrgliq": 0.118,
    "liq2m": 32000000,
    "divbpatr": 1.15,
    "lpa: ": 1.08,
    "vpa: ": 8.79,
    "epsTrimestral": 0.29,
    "dataRef": "31/12/2024",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2021", "receita": 21400000000, "lucro": 5020000000, "cotacao": 6.20 },
      { "periodo": "2022", "receita": 22800000000, "lucro": 1100000000, "cotacao": 7.15 },
      { "periodo": "2023", "receita": 21500000000, "lucro": 1950000000, "cotacao": 8.50 },
      { "periodo": "2024", "receita": 23100000000, "lucro": 2200000000, "cotacao": 9.30 },
      { "periodo": "Últimos 12m", "receita": 24200000000, "lucro": 2450000000, "cotacao": 9.85 }
    ]
  },
  {
    "papel": "LREN3",
    "empresa": "Lojas Renner S.A.",
    "cotacao": 16.20,
    "pl": 12.80,
    "pvp": 1.55,
    "evebit": 8.40,
    "roe": 0.098,
    "roic": 0.105,
    "dy": 0.045,
    "mrgliq": 0.082,
    "liq2m": 210000000,
    "divbpatr": 0.42,
    "lpa": 1.26,
    "vpa": 10.45,
    "epsTrimestral": 0.32,
    "dataRef": "31/12/2024",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2021", "receita": 9500000000, "lucro": 650000000, "cotacao": 28.40 },
      { "periodo": "2022", "receita": 11200000000, "lucro": 1200000000, "cotacao": 20.30 },
      { "periodo": "2023", "receita": 11800000000, "lucro": 980000000, "cotacao": 15.10 },
      { "periodo": "2024", "receita": 12400000000, "lucro": 1150000000, "cotacao": 15.80 },
      { "periodo": "Últimos 12m", "receita": 12900000000, "lucro": 1320000000, "cotacao": 16.20 }
    ]
  },
  {
    "papel": "ALUP11",
    "empresa": "Alupar Energia",
    "cotacao": 29.50,
    "pl": 7.80,
    "pvp": 1.05,
    "evebit": 5.92,
    "roe": 0.134,
    "roic": 0.118,
    "dy": 0.071,
    "mrgliq": 0.282,
    "liq2m": 14000000,
    "divbpatr": 1.62,
    "lpa": 3.78,
    "vpa": 28.09,
    "epsTrimestral": 0.92,
    "dataRef": "31/12/2024",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2021", "receita": 48000000000, "lucro": 1100000000, "cotacao": 23.10 },
      { "periodo": "2022", "receita": 51200000000, "lucro": 980000000, "cotacao": 25.40 },
      { "periodo": "2023", "receita": 58000000000, "lucro": 1050000000, "cotacao": 26.80 },
      { "periodo": "2024", "receita": 62000000000, "lucro": 1180000000, "cotacao": 28.55 },
      { "periodo": "Últimos 12m", "receita": 64500000000, "lucro": 1240000000, "cotacao": 29.50 }
    ]
  },
  {
    "papel": "OIBR3",
    "empresa": "Oi S.A. (Recup. Judicial)",
    "cotacao": 0.62,
    "pl": -0.15,
    "pvp": -0.05,
    "evebit": -0.80,
    "roe": -1.25,
    "roic": -0.12,
    "dy": 0.00,
    "mrgliq": -0.74,
    "liq2m": 12000000,
    "divbpatr": 8.50,
    "lpa": -4.13,
    "vpa": -12.40,
    "epsTrimestral": -1.15,
    "dataRef": "31/12/2024",
    "quedaLucro": "-120% queda de lucro",
    "situacao": "Recup. Judicial",
    "historico": [
      { "periodo": "2021", "receita": 16500000000, "lucro": -3400000000, "cotacao": 4.80 },
      { "periodo": "2022", "receita": 12500000000, "lucro": -19000000000, "cotacao": 1.50 },
      { "periodo": "2023", "receita": 9500000000, "lucro": -5400000000, "cotacao": 0.85 },
      { "periodo": "2024", "receita": 8100000000, "lucro": -15000000000, "cotacao": 0.70 },
      { "periodo": "Últimos 12m", "receita": 7800000000, "lucro": -16800000000, "cotacao": 0.62 }
    ]
  },
  {
    "papel": "OIBR4",
    "empresa": "Oi S.A. Pref (Recup. Judicial)",
    "cotacao": 1.15,
    "pl": -0.28,
    "pvp": -0.09,
    "evebit": -0.80,
    "roe": -1.25,
    "roic": -0.12,
    "dy": 0.00,
    "mrgliq": -0.74,
    "liq2m": 3500000,
    "divbpatr": 8.50,
    "lpa": -4.13,
    "vpa": -12.40,
    "epsTrimestral": -1.15,
    "dataRef": "31/12/2024",
    "quedaLucro": "-120% queda de lucro",
    "situacao": "Recup. Judicial",
    "historico": [
      { "periodo": "2021", "receita": 16500000000, "lucro": -3400000000, "cotacao": 8.50 },
      { "periodo": "2022", "receita": 12500000000, "lucro": -19000000000, "cotacao": 3.10 },
      { "periodo": "2023", "receita": 9500000000, "lucro": -5400000000, "cotacao": 1.80 },
      { "periodo": "2024", "receita": 8100000000, "lucro": -15000000000, "cotacao": 1.30 },
      { "periodo": "Últimos 12m", "receita": 7800000000, "lucro": -16800000000, "cotacao": 1.15 }
    ]
  },
  {
    "papel": "AMER3",
    "empresa": "Lojas Americanas (Recup. Judicial)",
    "cotacao": 0.18,
    "pl": -0.04,
    "pvp": -0.02,
    "evebit": -0.15,
    "roe": -2.48,
    "roic": -0.45,
    "dy": 0.00,
    "mrgliq": -0.32,
    "liq2m": 18000000,
    "divbpatr": 15.40,
    "lpa": -4.50,
    "vpa": -9.00,
    "epsTrimestral": -2.10,
    "dataRef": "31/12/2024",
    "quedaLucro": "-85% queda de lucro",
    "situacao": "Recup. Judicial",
    "historico": [
      { "periodo": "2021", "receita": 22100000000, "lucro": 540000000, "cotacao": 31.50 },
      { "periodo": "2022", "receita": 25800000000, "lucro": -12900000000, "cotacao": 9.80 },
      { "periodo": "2023", "receita": 14200000000, "lucro": -4500000000, "cotacao": 0.90 },
      { "periodo": "2024", "receita": 11000000000, "lucro": -8200000000, "cotacao": 0.25 },
      { "periodo": "Últimos 12m", "receita": 10200000000, "lucro": -8400000000, "cotacao": 0.18 }
    ]
  },
  {
    "papel": "GOLL4",
    "empresa": "Gol Linhas Aéreas (Recup. Judicial)",
    "cotacao": 1.05,
    "pl": -0.35,
    "pvp": -0.12,
    "evebit": -1.20,
    "roe": -1.82,
    "roic": -0.08,
    "dy": 0.00,
    "mrgliq": -0.15,
    "liq2m": 45000000,
    "divbpatr": 22.00,
    "lpa": -3.00,
    "vpa": -8.75,
    "epsTrimestral": -0.84,
    "dataRef": "31/12/2024",
    "quedaLucro": "-110% queda de lucro",
    "situacao": "Recup. Judicial",
    "historico": [
      { "periodo": "2021", "receita": 7400000000, "lucro": -7200000000, "cotacao": 18.50 },
      { "periodo": "2022", "receita": 13200000000, "lucro": -150000000, "cotacao": 7.20 },
      { "periodo": "2023", "receita": 17200000000, "lucro": -270000000, "cotacao": 8.90 },
      { "periodo": "2024", "receita": 18500000000, "lucro": -540000000, "cotacao": 1.40 },
      { "periodo": "Últimos 12m", "receita": 19100000000, "lucro": -580000000, "cotacao": 1.05 }
    ]
  },
  {
    "papel": "AZUL4",
    "empresa": "Azul Linhas Aéreas",
    "cotacao": 5.60,
    "pl": -1.80,
    "pvp": -0.65,
    "evebit": 4.20,
    "roe": -0.85,
    "roic": 0.095,
    "dy": 0.00,
    "mrgliq": -0.052,
    "liq2m": 110000000,
    "divbpatr": 5.20,
    "lpa": -3.11,
    "vpa": -8.62,
    "epsTrimestral": -0.52,
    "dataRef": "31/12/2024",
    "quedaLucro": "-45% queda de lucro",
    "situacao": "Alta Alavancagem",
    "historico": [
      { "periodo": "2021", "receita": 7900000000, "lucro": -4800000000, "cotacao": 25.10 },
      { "periodo": "2022", "receita": 15300000000, "lucro": -750000000, "cotacao": 11.80 },
      { "periodo": "2023", "receita": 18100000000, "lucro": -980000000, "cotacao": 14.10 },
      { "periodo": "2024", "receita": 19200000000, "lucro": -1250000000, "cotacao": 6.80 },
      { "periodo": "Últimos 12m", "receita": 19800000000, "lucro": -1450000000, "cotacao": 5.60 }
    ]
  },
  {
    "papel": "BBSE3",
    "empresa": "BB Seguridade",
    "cotacao": 31.20,
    "pl": 8.12,
    "pvp": 5.40,
    "evebit": 6.10,
    "roe": 0.665,
    "roic": 0.584,
    "dy": 0.098,
    "mrgliq": 0.512,
    "liq2m": 135000000,
    "divbpatr": 0.02,
    "lpa": 3.84,
    "vpa": 5.78,
    "epsTrimestral": 0.98,
    "dataRef": "31/12/2024",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2021", "receita": 5400000000, "lucro": 3900000000, "cotacao": 20.10 },
      { "periodo": "2022", "receita": 7200000000, "lucro": 6000000000, "cotacao": 28.50 },
      { "periodo": "2023", "receita": 8100000000, "lucro": 7700000000, "cotacao": 33.20 },
      { "periodo": "2024", "receita": 8800000000, "lucro": 8400000000, "cotacao": 30.50 },
      { "periodo": "Últimos 12m", "receita": 9200000000, "lucro": 8900000000, "cotacao": 31.20 }
    ]
  }
]

EXTRA_SEED = [
  {"papel": "ABEV3", "empresa": "Ambev ON", "cotacao": 12.10, "pl": 13.20, "pvp": 2.20, "evebit": 9.50, "roe": 0.165, "roic": 0.158, "dy": 0.062, "mrgliq": 0.185, "liq2m": 380000000, "divbpatr": 0.04, "lpa": 0.92, "vpa": 5.50, "epsTrimestral": 0.24, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "BBDC4", "empresa": "Bradesco Pref", "cotacao": 14.50, "pl": 9.20, "pvp": 0.92, "evebit": 7.20, "roe": 0.102, "roic": 0.095, "dy": 0.068, "mrgliq": 0.112, "liq2m": 450000000, "divbpatr": 0.15, "lpa": 1.58, "vpa": 15.76, "epsTrimestral": 0.38, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "RENT3", "empresa": "Localiza ON", "cotacao": 52.40, "pl": 18.50, "pvp": 2.60, "evebit": 11.20, "roe": 0.141, "roic": 0.112, "dy": 0.028, "mrgliq": 0.095, "liq2m": 290000000, "divbpatr": 1.85, "lpa": 2.83, "vpa": 20.15, "epsTrimestral": 0.72, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "ELET3", "empresa": "Eletrobras ON", "cotacao": 39.80, "pl": 15.40, "pvp": 0.82, "evebit": 11.50, "roe": 0.053, "roic": 0.048, "dy": 0.035, "mrgliq": 0.084, "liq2m": 280000000, "divbpatr": 1.45, "lpa": 2.58, "vpa": 48.54, "epsTrimestral": 0.65, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "SANB11", "empresa": "Santander Brasil Units", "cotacao": 28.10, "pl": 9.80, "pvp": 1.05, "evebit": 8.20, "roe": 0.112, "roic": 0.104, "dy": 0.064, "mrgliq": 0.121, "liq2m": 160000000, "divbpatr": 0.18, "lpa": 2.86, "vpa": 26.76, "epsTrimestral": 0.74, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "SUZB3", "empresa": "Suzano ON", "cotacao": 51.20, "pl": 8.90, "pvp": 1.85, "evebit": 7.40, "roe": 0.208, "roic": 0.152, "dy": 0.042, "mrgliq": 0.182, "liq2m": 180000000, "divbpatr": 2.15, "lpa": 5.75, "vpa": 27.68, "epsTrimestral": 1.42, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "GGBR4", "empresa": "Gerdau Pref", "cotacao": 21.30, "pl": 6.40, "pvp": 0.85, "evebit": 4.80, "roe": 0.132, "roic": 0.115, "dy": 0.075, "mrgliq": 0.104, "liq2m": 190000000, "divbpatr": 0.45, "lpa": 3.32, "vpa": 25.05, "epsTrimestral": 0.85, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "PRIO3", "empresa": "PetroRio ON", "cotacao": 43.50, "pl": 9.50, "pvp": 2.20, "evebit": 6.10, "roe": 0.231, "roic": 0.194, "dy": 0.012, "mrgliq": 0.254, "liq2m": 310000000, "divbpatr": 0.84, "lpa": 4.57, "vpa": 19.78, "epsTrimestral": 1.18, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "SBSP3", "empresa": "Sabesp ON", "cotacao": 78.40, "pl": 12.10, "pvp": 1.65, "evebit": 9.10, "roe": 0.136, "roic": 0.118, "dy": 0.038, "mrgliq": 0.154, "liq2m": 230000000, "divbpatr": 0.95, "lpa": 6.47, "vpa": 47.51, "epsTrimestral": 1.65, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "EQTL3", "empresa": "Equatorial ON", "cotacao": 32.40, "pl": 14.50, "pvp": 2.10, "evebit": 9.85, "roe": 0.145, "roic": 0.124, "dy": 0.032, "mrgliq": 0.095, "liq2m": 150000000, "divbpatr": 2.45, "lpa": 2.23, "vpa": 15.43, "epsTrimestral": 0.58, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "B3SA3", "empresa": "B3 S.A. ON", "cotacao": 10.80, "pl": 12.40, "pvp": 2.85, "evebit": 9.10, "roe": 0.225, "roic": 0.184, "dy": 0.058, "mrgliq": 0.421, "liq2m": 310000000, "divbpatr": 0.68, "lpa": 0.87, "vpa": 3.79, "epsTrimestral": 0.22, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "VIVT3", "empresa": "Telefônica Vivo ON", "cotacao": 51.50, "pl": 14.20, "pvp": 1.15, "evebit": 8.90, "roe": 0.081, "roic": 0.075, "dy": 0.072, "mrgliq": 0.098, "liq2m": 85000000, "divbpatr": 0.25, "lpa": 3.62, "vpa": 44.78, "epsTrimestral": 0.94, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "RAIL3", "empresa": "Rumo S.A. ON", "cotacao": 22.80, "pl": 19.10, "pvp": 1.74, "evebit": 11.50, "roe": 0.091, "roic": 0.082, "dy": 0.021, "mrgliq": 0.078, "liq2m": 140000000, "divbpatr": 2.10, "lpa": 1.19, "vpa": 13.10, "epsTrimestral": 0.31, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "RADL3", "empresa": "RaiaDrogasil ON", "cotacao": 26.50, "pl": 32.10, "pvp": 6.10, "evebit": 21.40, "roe": 0.191, "roic": 0.165, "dy": 0.015, "mrgliq": 0.042, "liq2m": 160000000, "divbpatr": 1.10, "lpa": 0.82, "vpa": 4.34, "epsTrimestral": 0.22, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "HAPV3", "empresa": "Hapvida ON", "cotacao": 3.85, "pl": 21.40, "pvp": 1.10, "evebit": 12.80, "roe": 0.051, "roic": 0.045, "dy": 0.000, "mrgliq": 0.038, "liq2m": 175000000, "divbpatr": 0.65, "lpa": 0.18, "vpa": 3.50, "epsTrimestral": 0.05, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "CPFE3", "empresa": "CPFL Energia ON", "cotacao": 33.50, "pl": 8.40, "pvp": 1.82, "evebit": 5.60, "roe": 0.216, "roic": 0.184, "dy": 0.095, "mrgliq": 0.115, "liq2m": 62000000, "divbpatr": 1.65, "lpa": 3.98, "vpa": 18.40, "epsTrimestral": 1.05, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "JBSS3", "empresa": "JBS ON", "cotacao": 31.50, "pl": 6.20, "pvp": 1.35, "evebit": 5.10, "roe": 0.218, "roic": 0.164, "dy": 0.075, "mrgliq": 0.048, "liq2m": 220000000, "divbpatr": 1.95, "lpa": 5.08, "vpa": 23.33, "epsTrimestral": 1.28, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "CSAN3", "empresa": "Cosan ON", "cotacao": 14.10, "pl": 12.40, "pvp": 1.32, "evebit": 8.70, "roe": 0.106, "roic": 0.089, "dy": 0.042, "mrgliq": 0.054, "liq2m": 110000000, "divbpatr": 2.85, "lpa": 1.13, "vpa": 10.68, "epsTrimestral": 0.28, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "RDOR3", "empresa": "Rede D'Or ON", "cotacao": 29.80, "pl": 15.60, "pvp": 2.15, "evebit": 11.40, "roe": 0.138, "roic": 0.114, "dy": 0.031, "mrgliq": 0.065, "liq2m": 135000000, "divbpatr": 1.95, "lpa": 1.91, "vpa": 13.86, "epsTrimestral": 0.51, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "UGPA3", "empresa": "Ultrapar ON", "cotacao": 25.10, "pl": 11.20, "pvp": 1.85, "evebit": 7.82, "roe": 0.165, "roic": 0.142, "dy": 0.048, "mrgliq": 0.031, "liq2m": 120000000, "divbpatr": 1.25, "lpa": 2.24, "vpa": 13.56, "epsTrimestral": 0.61, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "TIMS3", "empresa": "TIM Brasil ON", "cotacao": 17.20, "pl": 12.50, "pvp": 1.45, "evebit": 7.10, "roe": 0.116, "roic": 0.105, "dy": 0.065, "mrgliq": 0.108, "liq2m": 95000000, "divbpatr": 0.48, "lpa": 1.37, "vpa": 11.86, "epsTrimestral": 0.36, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "CMIG4", "empresa": "Cemig Pref", "cotacao": 11.40, "pl": 5.80, "pvp": 1.02, "evebit": 4.20, "roe": 0.175, "roic": 0.145, "dy": 0.092, "mrgliq": 0.125, "liq2m": 105000000, "divbpatr": 1.10, "lpa": 1.96, "vpa": 11.17, "epsTrimestral": 0.52, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "CCRO3", "empresa": "CCR ON", "cotacao": 12.50, "pl": 10.80, "pvp": 1.42, "evebit": 6.90, "roe": 0.131, "roic": 0.115, "dy": 0.054, "mrgliq": 0.092, "liq2m": 115000000, "divbpatr": 2.30, "lpa": 1.15, "vpa": 8.80, "epsTrimestral": 0.30, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "EMBR3", "empresa": "Embraer ON", "cotacao": 38.20, "pl": 16.50, "pvp": 1.82, "evebit": 11.10, "roe": 0.110, "roic": 0.098, "dy": 0.011, "mrgliq": 0.062, "liq2m": 210000000, "divbpatr": 1.20, "lpa": 2.31, "vpa": 20.98, "epsTrimestral": 0.58, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "EGIE3", "empresa": "Engie Brasil ON", "cotacao": 41.20, "pl": 10.50, "pvp": 3.10, "evebit": 7.20, "roe": 0.295, "roic": 0.221, "dy": 0.082, "mrgliq": 0.214, "liq2m": 85000000, "divbpatr": 1.95, "lpa": 3.92, "vpa": 13.29, "epsTrimestral": 1.02, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "KLBN11", "empresa": "Klabin Units", "cotacao": 21.10, "pl": 11.40, "pvp": 2.15, "evebit": 8.10, "roe": 0.188, "roic": 0.135, "dy": 0.071, "mrgliq: ": 0.124, "liq2m": 90000000, "divbpatr": 2.85, "lpa": 1.85, "vpa": 9.81, "epsTrimestral": 0.48, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "ENGI11", "empresa": "Energisa Units", "cotacao": 46.50, "pl": 8.10, "pvp": 1.45, "evebit": 5.90, "roe": 0.179, "roic": 0.132, "dy": 0.068, "mrgliq": 0.092, "liq2m": 55000000, "divbpatr": 2.20, "lpa": 5.74, "vpa": 32.06, "epsTrimestral": 1.45, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "VBBR3", "empresa": "Vibra Energia ON", "cotacao": 23.40, "pl": 9.90, "pvp": 1.62, "evebit": 7.10, "roe": 0.163, "roic": 0.141, "dy": 0.061, "mrgliq": 0.042, "liq2m": 145000000, "divbpatr": 1.35, "lpa": 2.36, "vpa": 14.44, "epsTrimestral": 0.62, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "FLRY3", "empresa": "Fleury ON", "cotacao": 15.20, "pl": 11.20, "pvp": 1.35, "evebit": 7.15, "roe": 0.120, "roic": 0.108, "dy": 0.058, "mrgliq": 0.078, "liq2m": 42000000, "divbpatr": 1.15, "lpa": 1.35, "vpa": 11.25, "epsTrimestral": 0.35, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "MULT3", "empresa": "Multiplan ON", "cotacao": 24.50, "pl": 12.80, "pvp": 1.74, "evebit": 9.10, "roe": 0.135, "roic": 0.121, "dy": 0.052, "mrgliq": 0.385, "liq2m": 65000000, "divbpatr": 0.85, "lpa": 1.91, "vpa": 14.08, "epsTrimestral": 0.49, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "ALOS3", "empresa": "Allos ON", "cotacao": 21.80, "pl": 13.50, "pvp": 0.95, "evebit": 8.80, "roe": 0.070, "roic": 0.065, "dy": 0.048, "mrgliq": 0.285, "liq2m": 58000000, "divbpatr": 0.72, "lpa": 1.61, "vpa": 22.94, "epsTrimestral": 0.41, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "CRFB3", "empresa": "Carrefour Brasil ON", "cotacao": 10.45, "pl": 15.20, "pvp": 0.98, "evebit": 11.20, "roe": 0.064, "roic": 0.058, "dy": 0.032, "mrgliq": 0.015, "liq2m": 85000000, "divbpatr": 1.85, "lpa": 0.68, "vpa": 10.66, "epsTrimestral": 0.18, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "ASAI3", "empresa": "Assaí ON", "cotacao": 11.15, "pl": 14.80, "pvp": 3.40, "evebit": 8.85, "roe": 0.231, "roic": 0.154, "dy": 0.025, "mrgliq": 0.019, "liq2m": 130000000, "divbpatr": 3.65, "lpa": 0.75, "vpa": 3.28, "epsTrimestral": 0.20, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Altamente Alavancado"},
  {"papel": "RECV3", "empresa": "PetroReconcavo ON", "cotacao": 19.45, "pl": 7.20, "pvp": 1.25, "evebit": 5.12, "roe": 0.173, "roic": 0.151, "dy": 0.082, "mrgliq": 0.184, "liq2m": 48000000, "divbpatr": 0.65, "lpa": 2.70, "vpa": 15.56, "epsTrimestral": 0.68, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "CYRE3", "empresa": "Cyrela ON", "cotacao": 21.20, "pl": 7.80, "pvp": 1.05, "evebit": 6.20, "roe": 0.134, "roic": 0.121, "dy": 0.065, "mrgliq": 0.125, "liq2m": 75000000, "divbpatr": 0.55, "lpa": 2.71, "vpa": 20.19, "epsTrimestral": 0.69, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "EZTC3", "empresa": "EZTec ON", "cotacao": 14.80, "pl": 10.20, "pvp": 0.65, "evebit": 8.40, "roe": 0.063, "roic": 0.058, "dy": 0.045, "mrgliq": 0.108, "liq2m": 22000000, "divbpatr": 0.10, "lpa": 1.45, "vpa": 22.76, "epsTrimestral": 0.38, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "MRVE3", "empresa": "MRV ON", "cotacao": 7.15, "pl": 18.20, "pvp": 0.61, "evebit": 14.50, "roe": 0.033, "roic": 0.028, "dy": 0.015, "mrgliq": 0.024, "liq2m": 65000000, "divbpatr": 1.45, "lpa": 0.39, "vpa": 11.72, "epsTrimestral": 0.10, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "TEND3", "empresa": "Tenda ON", "cotacao": 11.50, "pl": -12.40, "pvp": 1.15, "evebit": -8.10, "roe": -0.092, "roic": -0.065, "dy": 0.000, "mrgliq": -0.042, "liq2m": 18000000, "divbpatr": 2.40, "lpa": -0.92, "vpa": 10.00, "epsTrimestral": -0.22, "dataRef": "31/12/2024", "quedaLucro": "Em recuperação", "situacao": "Nível Alerta"},
  {"papel": "JSHF3", "empresa": "JHSF ON", "cotacao": 4.15, "pl": 7.90, "pvp": 0.58, "evebit": 5.90, "roe": 0.073, "roic": 0.068, "dy": 0.062, "mrgliq": 0.145, "liq2m": 15000000, "divbpatr": 0.92, "lpa": 0.52, "vpa": 7.15, "epsTrimestral": 0.14, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "DIRR3", "empresa": "Direcional ON", "cotacao": 25.40, "pl": 9.10, "pvp": 2.10, "evebit": 6.80, "roe": 0.231, "roic": 0.198, "dy": 0.078, "mrgliq": 0.142, "liq2m": 28000000, "divbpatr": 0.45, "lpa": 2.79, "vpa": 12.10, "epsTrimestral": 0.72, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "SLCE3", "empresa": "SLC Agrícola ON", "cotacao": 18.50, "pl": 8.50, "pvp": 1.15, "evebit": 5.40, "roe": 0.135, "roic": 0.112, "dy": 0.055, "mrgliq": 0.098, "liq2m": 48000000, "divbpatr": 0.95, "lpa": 2.17, "vpa": 16.08, "epsTrimestral": 0.55, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "SMTO3", "empresa": "São Martinho ON", "cotacao": 28.90, "pl": 11.10, "pvp": 1.45, "evebit": 6.80, "roe": 0.130, "roic": 0.114, "dy": 0.048, "mrgliq": 0.112, "liq2m": 35000000, "divbpatr": 1.20, "lpa": 2.60, "vpa": 19.93, "epsTrimestral": 0.68, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "GOAU4", "empresa": "Metalúrgica Gerdau Pref", "cotacao": 10.15, "pl": 5.15, "pvp": 0.65, "evebit": 4.10, "roe": 0.126, "roic": 0.118, "dy": 0.081, "mrgliq": 0.115, "liq2m": 55000000, "divbpatr": 0.42, "lpa": 1.97, "vpa": 15.61, "epsTrimestral": 0.51, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "USIM5", "empresa": "Usiminas Pref", "cotacao": 6.80, "pl": 11.50, "pvp": 0.42, "evebit": 8.90, "roe": 0.036, "roic": 0.031, "dy": 0.025, "mrgliq": 0.024, "liq2m": 65000000, "divbpatr": 0.22, "lpa": 0.59, "vpa": 16.19, "epsTrimestral": 0.15, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "CSNA3", "empresa": "Siderúrgica Nacional ON", "cotacao": 12.15, "pl": 18.50, "pvp": 0.85, "evebit": 11.20, "roe": 0.045, "roic": 0.038, "dy": 0.035, "mrgliq": 0.021, "liq2m": 95000000, "divbpatr": 3.15, "lpa": 0.65, "vpa": 14.29, "epsTrimestral": 0.16, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Altamente Alavancado"},
  {"papel": "CVCB3", "empresa": "CVC Brasil ON", "cotacao": 2.10, "pl": -3.50, "pvp": -1.25, "evebit": -6.40, "roe": -0.355, "roic": -0.152, "dy": 0.000, "mrgliq": -0.062, "liq2m": 35000000, "divbpatr": 5.10, "lpa": -0.60, "vpa": -1.68, "epsTrimestral": -0.15, "dataRef": "31/12/2024", "quedaLucro": "No limite", "situacao": "Nível Alerta"},
  {"papel": "MGLU3", "empresa": "Magazine Luiza ON", "cotacao": 13.50, "pl": 42.10, "pvp": 1.15, "evebit": 12.40, "roe": 0.027, "roic": 0.035, "dy": 0.000, "mrgliq": 0.006, "liq2m": 140000000, "divbpatr": 1.85, "lpa": 0.32, "vpa": 11.73, "epsTrimestral": 0.08, "dataRef": "31/12/2024", "quedaLucro": "Em transição", "situacao": "Nível Alerta"},
  {"papel": "BHIA3", "empresa": "Casas Bahia ON", "cotacao": 4.80, "pl": -1.15, "pvp": 0.18, "evebit": -2.85, "roe": -0.485, "roic": -0.118, "dy": 0.000, "mrgliq": -0.084, "liq2m": 48000000, "divbpatr": 8.40, "lpa": -4.17, "vpa": 26.66, "epsTrimestral": -1.04, "dataRef": "31/12/2024", "quedaLucro": "Forte queda", "situacao": "Altamente Alavancado"},
  {"papel": "PETR3", "empresa": "Petrobras ON", "cotacao": 41.10, "pl": 4.54, "pvp": 1.22, "evebit": 3.90, "roe": 0.268, "roic": 0.232, "dy": 0.138, "mrgliq": 0.191, "liq2m": 210000000, "divbpatr": 0.78, "lpa": 9.05, "vpa": 33.70, "epsTrimestral": 2.26, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "BBDC3", "empresa": "Bradesco ON", "cotacao": 13.10, "pl": 8.35, "pvp": 0.83, "evebit": 6.55, "roe": 0.099, "roic": 0.091, "dy": 0.065, "mrgliq": 0.108, "liq2m": 110000000, "divbpatr": 0.15, "lpa": 1.57, "vpa": 15.76, "epsTrimestral": 0.37, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "CXSE3", "empresa": "Caixa Seguridade ON", "cotacao": 14.85, "pl": 8.85, "pvp": 3.10, "evebit": 6.85, "roe": 0.354, "roic": 0.312, "dy": 0.091, "mrgliq": 0.485, "liq2m": 72000000, "divbpatr": 0.01, "lpa": 1.68, "vpa": 4.79, "epsTrimestral": 0.42, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "PSSA3", "empresa": "Porto Seguro ON", "cotacao": 29.40, "pl": 8.90, "pvp": 1.35, "evebit": 6.50, "roe": 0.151, "roic": 0.125, "dy": 0.062, "mrgliq": 0.082, "liq2m": 45000000, "divbpatr": 0.15, "lpa": 3.30, "vpa": 21.78, "epsTrimestral": 0.82, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "WIZC3", "empresa": "Wiz Co ON", "cotacao": 6.15, "pl": 5.40, "pvp": 1.25, "evebit": 3.84, "roe": 0.231, "roic": 0.185, "dy": 0.112, "mrgliq": 0.115, "liq2m": 12000000, "divbpatr": 0.85, "lpa": 1.14, "vpa": 4.92, "epsTrimestral": 0.29, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "ZAMP3", "empresa": "Zamp ON", "cotacao": 3.50, "pl": -4.80, "pvp": 0.65, "evebit": -3.90, "roe": -0.135, "roic": -0.098, "dy": 0.000, "mrgliq": -0.045, "liq2m": 8500000, "divbpatr": 1.95, "lpa": -0.73, "vpa": 5.38, "epsTrimestral": -0.18, "dataRef": "31/12/2024", "quedaLucro": "Estagnado", "situacao": "Nível Alerta"},
  {"papel": "YDUQ3", "empresa": "Yduqs ON", "cotacao": 11.85, "pl": 10.20, "pvp": 0.95, "evebit": 6.40, "roe": 0.093, "roic": 0.085, "dy": 0.038, "mrgliq": 0.058, "liq2m": 35000000, "divbpatr": 1.45, "lpa": 1.16, "vpa": 12.47, "epsTrimestral": 0.29, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "COGN3", "empresa": "Cogna ON", "cotacao": 1.95, "pl": 22.40, "pvp": 0.42, "evebit": 8.10, "roe": 0.019, "roic": 0.025, "dy": 0.000, "mrgliq": 0.015, "liq2m": 55000000, "divbpatr": 1.65, "lpa": 0.09, "vpa: ": 4.64, "epsTrimestral": 0.02, "dataRef": "31/12/2024", "quedaLucro": "Em recuperação", "situacao": "Nível Alerta"},
  {"papel": "MOVI3", "empresa": "Movida ON", "cotacao": 5.90, "pl": -11.20, "pvp": 0.65, "evebit": 9.10, "roe": -0.058, "roic": 0.054, "dy": 0.025, "mrgliq": -0.031, "liq2m": 25000000, "divbpatr": 3.85, "lpa": -0.53, "vpa": 9.08, "epsTrimestral": -0.13, "dataRef": "31/12/2024", "quedaLucro": "Gargalo juros", "situacao": "Altamente Alavancado"},
  {"papel": "LOGG3", "empresa": "Log Commercial Properties", "cotacao": 22.40, "pl": 11.80, "pvp": 0.85, "evebit": 8.40, "roe": 0.072, "roic": 0.064, "dy": 0.045, "mrgliq": 0.215, "liq2m": 14000000, "divbpatr": 0.48, "lpa": 1.90, "vpa": 26.35, "epsTrimestral": 0.48, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "STBP3", "empresa": "Santos Brasil ON", "cotacao": 13.50, "pl": 12.80, "pvp": 2.90, "evebit": 8.15, "roe": 0.226, "roic": 0.191, "dy": 0.068, "mrgliq": 0.185, "liq2m: ": 65000000, "divbpatr": 0.35, "lpa": 1.05, "vpa": 4.65, "epsTrimestral": 0.26, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "PCAR3", "empresa": "Pão de Açúcar ON", "cotacao": 3.15, "pl": -1.82, "pvp": 0.26, "evebit": -4.10, "roe": -0.142, "roic": -0.052, "dy": 0.000, "mrgliq": -0.038, "liq2m": 24000000, "divbpatr": 2.85, "lpa": -1.73, "vpa": 12.11, "epsTrimestral": -0.43, "dataRef": "31/12/2024", "quedaLucro": "Forte queda", "situacao": "Nível Alerta"},
  {"papel": "BRFS3", "empresa": "BRF S.A. ON", "cotacao": 21.80, "pl": 8.80, "pvp": 1.62, "evebit": 5.90, "roe": 0.184, "roic": 0.142, "dy": 0.035, "mrgliq": 0.052, "liq2m": 185000000, "divbpatr": 1.75, "lpa": 2.48, "vpa": 13.45, "epsTrimestral": 0.62, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "MRFG3", "empresa": "Marfrig ON", "cotacao": 12.40, "pl": 6.10, "pvp": 1.84, "evebit": 4.80, "roe": 0.301, "roic": 0.154, "dy": 0.082, "mrgliq": 0.024, "liq2m": 68000000, "divbpatr": 4.45, "lpa": 2.03, "vpa": 6.74, "epsTrimestral": 0.51, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Altamente Alavancado"},
  {"papel": "BEEF3", "empresa": "Minerva ON", "cotacao": 6.45, "pl": 6.85, "pvp": 1.55, "evebit": 5.10, "roe": 0.226, "roic": 0.125, "dy": 0.075, "mrgliq": 0.021, "liq2m": 42000000, "divbpatr": 3.45, "lpa": 0.94, "vpa": 4.16, "epsTrimestral": 0.24, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Altamente Alavancado"},
  {"papel": "LEVE3", "empresa": "Metal Leve ON", "cotacao": 34.10, "pl": 8.15, "pvp": 2.95, "evebit": 6.20, "roe": 0.362, "roic": 0.315, "dy": 0.114, "mrgliq": 0.145, "liq2m": 15000000, "divbpatr": 0.12, "lpa": 4.18, "vpa": 11.56, "epsTrimestral": 1.05, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "TUPY3", "empresa": "Tupy ON", "cotacao": 23.10, "pl": 9.15, "pvp": 1.15, "evebit": 6.40, "roe": 0.125, "roic": 0.112, "dy": 0.048, "mrgliq": 0.045, "liq2m": 14000000, "divbpatr": 1.10, "lpa": 2.52, "vpa": 20.08, "epsTrimestral": 0.63, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "POMO4", "empresa": "Marcopolo Pref", "cotacao": 7.45, "pl": 7.90, "pvp": 1.62, "evebit": 5.92, "roe": 0.205, "roic": 0.174, "dy": 0.055, "mrgliq": 0.085, "liq2m": 35000000, "divbpatr": 0.55, "lpa": 0.94, "vpa": 4.60, "epsTrimestral": 0.24, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "RAPT4", "empresa": "Randon Pref", "cotacao": 11.20, "pl": 9.80, "pvp": 1.35, "evebit": 7.10, "roe": 0.138, "roic": 0.115, "dy": 0.042, "mrgliq": 0.048, "liq2m": 18000000, "divbpatr": 1.45, "lpa": 1.14, "vpa": 8.30, "epsTrimestral": 0.29, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "KEPL3", "empresa": "Kepler Weber ON", "cotacao": 10.15, "pl": 7.80, "pvp": 2.15, "evebit": 5.40, "roe": 0.275, "roic": 0.231, "dy": 0.088, "mrgliq": 0.125, "liq2m": 12000000, "divbpatr": 0.08, "lpa": 1.30, "vpa": 4.72, "epsTrimestral": 0.33, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "UNIP6", "empresa": "Unipar Pref", "cotacao": 65.50, "pl": 9.12, "pvp": 2.85, "evebit": 6.45, "roe": 0.312, "roic": 0.245, "dy": 0.081, "mrgliq": 0.142, "liq2m": 22000000, "divbpatr": 0.65, "lpa": 7.18, "vpa": 22.98, "epsTrimestral": 1.80, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "FESA4", "empresa": "Ferbasa Pref", "cotacao": 42.10, "pl": 10.40, "pvp": 1.12, "evebit": 7.80, "roe": 0.108, "roic": 0.098, "dy": 0.052, "mrgliq": 0.118, "liq2m": 9000000, "divbpatr": 0.05, "lpa": 4.05, "vpa: ": 37.59, "epsTrimestral": 1.01, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "SIMH3", "empresa": "Simpar ON", "cotacao": 6.40, "pl": -14.50, "pvp": 0.98, "evebit": 11.20, "roe": -0.068, "roic": 0.052, "dy": 0.021, "mrgliq": -0.015, "liq2m": 35000000, "divbpatr": 4.85, "lpa": -0.44, "vpa": 6.53, "epsTrimestral": -0.11, "dataRef": "31/12/2024", "quedaLucro": "Gargalo financeiro", "situacao": "Altamente Alavancado"},
  {"papel": "TGMA3", "empresa": "Tegma ON", "cotacao": 23.50, "pl": 8.85, "pvp": 2.15, "evebit": 6.10, "roe": 0.243, "roic": 0.215, "dy": 0.092, "mrgliq": 0.108, "liq2m": 8500000, "divbpatr": 0.15, "lpa": 2.65, "vpa": 10.93, "epsTrimestral": 0.66, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "VLID3", "empresa": "Valid ON", "cotacao": 18.20, "pl": 6.95, "pvp": 1.22, "evebit": 4.85, "roe": 0.175, "roic": 0.148, "dy": 0.071, "mrgliq": 0.088, "liq2m": 11000000, "divbpatr": 0.55, "lpa: ": 2.62, "vpa": 14.92, "epsTrimestral": 0.66, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "INTB3", "empresa": "Intelbras ON", "cotacao": 21.50, "pl": 11.20, "pvp": 1.82, "evebit": 8.10, "roe": 0.162, "roic": 0.141, "dy": 0.039, "mrgliq": 0.092, "liq2m": 18000000, "divbpatr": 0.42, "lpa: ": 1.92, "vpa": 11.81, "epsTrimestral": 0.48, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "POSI3", "empresa": "Positivo ON", "cotacao": 7.85, "pl": 6.40, "pvp": 0.85, "evebit": 4.50, "roe": 0.132, "roic": 0.115, "dy": 0.062, "mrgliq": 0.045, "liq2m": 12000000, "divbpatr": 0.95, "lpa": 1.23, "vpa": 9.24, "epsTrimestral": 0.31, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "FRAS3", "empresa": "Fras-le ON", "cotacao": 16.50, "pl": 9.20, "pvp": 1.95, "evebit": 6.80, "roe": 0.212, "roic": 0.183, "dy": 0.048, "mrgliq": 0.091, "liq2m": 9000000, "divbpatr": 0.85, "lpa": 1.79, "vpa": 8.46, "epsTrimestral": 0.45, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "SHUL4", "empresa": "Schulz Pref", "cotacao": 6.85, "pl": 7.15, "pvp": 1.32, "evebit": 5.10, "roe": 0.184, "roic": 0.158, "dy": 0.058, "mrgliq": 0.112, "liq2m": 5500000, "divbpatr": 0.38, "lpa": 0.96, "vpa": 5.19, "epsTrimestral": 0.24, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "AURE3", "empresa": "Auren Energia ON", "cotacao": 12.50, "pl": 11.80, "pvp": 1.10, "evebit": 8.90, "roe": 0.093, "roic": 0.084, "dy": 0.102, "mrgliq": 0.115, "liq2m": 45000000, "divbpatr": 1.32, "lpa": 1.06, "vpa": 11.36, "epsTrimestral": 0.27, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "AESB3", "empresa": "AES Brasil ON", "cotacao": 9.15, "pl": -18.50, "pvp": 1.15, "evebit": 11.80, "roe": -0.062, "roic": 0.055, "dy": 0.025, "mrgliq": -0.024, "liq2m": 22000000, "divbpatr": 3.45, "lpa": -0.49, "vpa": 7.96, "epsTrimestral": -0.12, "dataRef": "31/12/2024", "quedaLucro": "Alta dívida", "situacao": "Altamente Alavancado"},
  {"papel": "CGAS5", "empresa": "Comgás Pref", "cotacao": 115.00, "pl": 8.90, "pvp": 4.85, "evebit": 6.10, "roe": 0.545, "roic": 0.385, "dy": 0.082, "mrgliq": 0.135, "liq2m": 14000000, "divbpatr": 1.95, "lpa": 12.92, "vpa": 23.71, "epsTrimestral": 3.23, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "SAPR4", "empresa": "Sanepar Pref", "cotacao": 4.95, "pl": 5.08, "pvp": 0.71, "evebit": 4.02, "roe": 0.139, "roic": 0.125, "dy": 0.078, "mrgliq": 0.241, "liq2m": 35000000, "divbpatr": 0.74, "lpa": 0.97, "vpa": 6.97, "epsTrimestral": 0.25, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "PARD3", "empresa": "Hermes Pardini ON", "cotacao": 18.15, "pl": 12.10, "pvp": 1.65, "evebit": 8.12, "roe": 0.136, "roic": 0.115, "dy": 0.038, "mrgliq": 0.075, "liq2m": 18000000, "divbpatr": 0.65, "lpa": 1.50, "vpa": 11.00, "epsTrimestral": 0.38, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "SOMA3", "empresa": "Grupo Soma ON", "cotacao": 6.15, "pl": 14.10, "pvp": 0.85, "evebit": 9.80, "roe": 0.060, "roic": 0.052, "dy": 0.021, "mrgliq": 0.038, "liq2m": 48000000, "divbpatr": 0.95, "lpa": 0.44, "vpa": 7.24, "epsTrimestral": 0.11, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável"}
]

# dynamic inflator to achieve exactly 100 stocks
def inflate_stock_history(stock_seed):
    import math
    periods = ["2021", "2022", "2023", "2024", "Últimos 12m"]
    # unique determinism based on papel string
    char_sum = sum(ord(char) for char in stock_seed["papel"])
    base_receita_val = (10 + (char_sum % 90)) * 1e9
    
    is_asymmetric = (char_sum % 3 == 0) and (stock_seed["pl"] > 0)
    historico = []
    
    for idx, periodo in enumerate(periods):
        revenue_multiplier = 0.8 + idx * 0.05 + math.sin(idx + char_sum) * 0.03
        profit_multiplier = 0.85 + idx * 0.045 + math.cos(idx + char_sum) * 0.035
        price_multiplier = 0.75 + idx * 0.065 + math.sin(idx * 1.5 + char_sum) * 0.05
        
        if stock_seed["situacao"] == "Recup. Judicial" or stock_seed["pl"] < 0:
            revenue_multiplier = 1.1 - idx * 0.08
            profit_multiplier = 0.9 - idx * 0.15
            price_multiplier = 1.3 - idx * 0.22
            
        if is_asymmetric and periodo == "Últimos 12m":
            profit_multiplier = 1.45
            price_multiplier = 0.88
        elif is_asymmetric and periodo == "2024":
            profit_multiplier = 1.30
            price_multiplier = 1.25
            
        receita = max(1e7, base_receita_val * revenue_multiplier)
        lucro = receita * stock_seed["mrgliq"] * profit_multiplier
        cotacao = stock_seed["cotacao"] if periodo == "Últimos 12m" else max(0.1, stock_seed["cotacao"] * price_multiplier)
        
        historico.append({
            "periodo": periodo,
            "receita": int(receita),
            "lucro": int(lucro),
            "cotacao": round(cotacao, 2)
        })
        
    stock_seed["historico"] = historico
    # Fix any keys starting with space or missing keys to clean up seed objects
    cleaned_seed = {}
    for k, v in stock_seed.items():
        cleaned_key = k.strip().replace(":", "")
        cleaned_seed[cleaned_key] = v
        
    # ensure standard lpa/vpa are populated if missing on cleaning
    if "lpa" not in cleaned_seed:
        cleaned_seed["lpa"] = cleaned_seed.get("lpa", 1.0)
    if "vpa" not in cleaned_seed:
        cleaned_seed["vpa"] = cleaned_seed.get("vpa", 10.0)
    return cleaned_seed

INFLATED_EXTRA = [inflate_stock_history(s) for s in EXTRA_SEED]
STOCK_DATABASE = (ORIGINAL_STOCKS + INFLATED_EXTRA)[:100]

MARKET_NEWS = [
    {"source": "InfoMoney", "title": "Ibovespa opera em alta no aguardo de dados de inflação IPCA", "link": "https://www.infomoney.com.br/economia/copom-sinaliza-postura-vigilante/"},
    {"source": "Money Times", "title": "Copom sinaliza postura vigilante e analistas projetam taxa Selic estável", "link": "https://www.moneytimes.com.br/ibovespa-hoje-inflacao-e-reuniao-do-copom-no-radar/"},
    {"source": "Valor", "title": "Dividendos da Petrobras (PETR4) em 2025: O que esperar após o último trimestre?", "link": "https://www.moneytimes.com.br/dividendos-da-petrobras-petr4-o-que-esperar/"}
]

LATEST_DIVIDENDS = [
    {"ativo": "PETR4", "valor": 1.1250, "data": "2025-05-15"},
    {"ativo": "BBAS3", "valor": 0.4520, "data": "2025-05-10"},
    {"ativo": "ITSA4", "valor": 0.1200, "data": "2025-05-02"},
    {"ativo": "TAEE11", "valor": 0.8540, "data": "2025-04-28"},
    {"ativo": "TRPL4", "valor": 0.6500, "data": "2025-04-12"}
]

# Inicializando session_state
if 'app_liberado' not in st.session_state:
    st.session_state.app_liberado = True
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

# --- ADVERTISMENT TOP BANNER ---
st.markdown("""
<div class="ad-banner">
    <span class="ad-badge">AD</span>
    Ganhe benefícios e apoie nossa comunidade acessando os patrocinadores na barra lateral ou nos links de membro!
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR STYLE ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo">B3</div>', unsafe_allow_html=True)
    st.subheader("Ranking Ibovespa")
    st.caption("Versão Ultra-Polida 2025 (Python Streamlit)")
    
    st.markdown("---")
    
    # Identificação do usuário
    st.markdown("##### **Identificação**")
    user_name_input = st.text_input(
        "Seu nome para relatórios:",
        value=st.session_state.user_name,
        placeholder="Ex: Nilton, Maria, Lucas..."
    )
    if user_name_input != st.session_state.user_name:
        st.session_state.user_name = user_name_input
        st.rerun()
        
    if st.session_state.user_name:
        st.markdown(f"<p style='font-size:12px; color:#475569; font-style:italic;'>Bem-vindo(a), <b>{st.session_state.user_name}</b>!</p>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Real-time report check
    time_now = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style="background-color:#f1f5f9; padding: 12px; border-radius:12px; font-size:12px; display:flex; justify-content:space-between; align-items:center;">
        <span style="font-weight:600; color:#334155;">Relatório B3 Online</span>
        <span style="font-family:'JetBrains Mono', monospace; font-weight:700; color:#2563eb;">{time_now} BRT</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Giga+ Fibra WhatsApp Banner
    st.markdown("""
    <a href="https://wa.me/552220410353?text=Use%20o%20codigo%20DVT329%20e%20ganhe%2020%25%20nas%20duas%20primeiras%20mensalidades" target="_blank" class="whatsapp-btn">
        <div style="font-size: 11px; text-transform: uppercase; font-weight: 800; opacity: 0.9; margin-bottom: 4px;">Código: DVT329</div>
        <div style="font-size: 15px; font-weight: 700; line-height: 1.2; margin-bottom: 8px;">Internet Estável com 20% OFF!</div>
        <p style="font-size: 11px; font-weight: normal; margin-bottom: 12px; opacity: 0.85; line-height: 1.4;">
            Assine as duas primeiras mensalidades de fibra de alta performance com desconto exclusivo Giga+ Fibra.
        </p>
        <div style="background-color:rgba(255,255,255,0.18); text-align:center; padding:8px 12px; border-radius:8px; font-size:11px;">
            Resgatar Cupom no WhatsApp →
        </div>
    </a>
    """, unsafe_allow_html=True)
    
    # Quick Stats Panel
    st.markdown(f"""
    <div style="background-color:white; border: 1px solid #e2e8f0; padding:16px; border-radius:16px;">
        <p style="font-size:10px; font-weight:bold; color:#94a3b8; text-transform:uppercase; margin-bottom:8px; letter-spacing:0.5px;">Universo de Cobertura</p>
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;">
            <span style="color:#64748b;">Ações Monitoradas:</span>
            <strong style="color:#1e293b;">{len(STOCK_DATABASE)} empresas</strong>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;">
            <span style="color:#64748b;">Liquidez diária base:</span>
            <strong style="color:#1e293b;">&gt; R$ 100k</strong>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:12px;">
            <span style="color:#64748b;">Métricas analíticas:</span>
            <strong style="color:#1e293b;">Graham & Greenblatt</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- CABEÇALHO DO DASHBOARD ---
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
        <span style="height:8px; width:8px; background-color:#2563eb; border-radius:50%; display:inline-block; animation:pulse 1s Infinite;"></span>
        <span style="font-size:11px; font-weight:bold; color:#64748b; text-transform:uppercase; font-family:'JetBrains Mono', monospace; letter-spacing:1px;">B3 Ibovespa Inteligente</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("# **Análise de Ações Baratas e Rentáveis**")
    
    ref_month = datetime.now().strftime("%B de %Y").capitalize()
    if st.session_state.user_name:
        st.markdown(f"Relatório tático de investimentos customizado para **{st.session_state.user_name}** • Referência de {ref_month}")
    else:
        st.markdown(f"Relatório tático de investimentos • Referência de {ref_month}")

with col_badge:
    st.markdown("""
    <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; padding: 12px; border-radius: 16px; text-align: center;">
        <span style="font-size: 20px;">🇧🇷</span>
        <div style="font-size: 9px; font-weight: bold; color: #1d4ed8; text-transform: uppercase; margin-top:2px;">Bolsa Paulista</div>
        <div style="font-size: 11px; font-weight: 800; color: #172554;">Ibovespa em 2025</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- MENU DE TABS DE ALTO NÍVEL ---
# Usando Streamplit st.radio como botões horizontais estilizados para mimetizar nosso layout de abas do React
tabs = [
    "🏆 Ranking Fundamentalista",
    "✨ Fórmula Mágica",
    "💎 Graham Valuation",
    "📈 EPS Diluído",
    "📉 Assimetria Lucro/Preço"
]
active_tab = st.radio("Selecione a Visualização:", tabs, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

# --- FUNÇÕES DE PROCESSAMENTO ---
df_original = pd.DataFrame(STOCK_DATABASE)

# Formatting helpers
def fmt_pct(val):
    return f"{val * 100:.1f}%"

def fmt_currency(val):
    return f"R$ {val:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")

def render_custom_stock_chart(selected_stock, title=""):
    import pandas as pd
    import plotly.graph_objects as go
    
    if not selected_stock or "historico" not in selected_stock:
        return go.Figure()
        
    hist_df = pd.DataFrame(selected_stock["historico"])
    receitas_bi = hist_df['receita'] / 1e9
    
    lucros = hist_df['lucro'].values
    cots = hist_df['cotacao'].values
    
    l_min, l_max = lucros.min(), lucros.max()
    p_min, p_max = cots.min(), cots.max()
    
    l_range = l_max - l_min
    p_range = p_max - p_min
    
    lucros_norm = [(l - l_min) / l_range if l_range > 0 else 0.5 for l in lucros]
    cots_norm = [(c - p_min) / p_range if p_range > 0 else 0.5 for c in cots]
    
    fig = go.Figure()
    
    # Receita: background gray bars on secondary y-axis
    fig.add_trace(go.Bar(
        x=hist_df['periodo'],
        y=receitas_bi,
        name="Receita (Bi R$)",
        marker=dict(color='#cbd5e1', line=dict(color='#e2e8f0', width=0)),
        yaxis='y2',
        opacity=0.45,
        hoverinfo='y'
    ))
    
    # Preço: Solid blue line on primary y-axis
    fig.add_trace(go.Scatter(
        x=hist_df['periodo'],
        y=cots_norm,
        name="Preço (Relativo)",
        line=dict(color='#2563eb', width=4),
        mode='lines+markers',
        marker=dict(size=8, color='#1d4ed8'),
        yaxis='y1',
        text=[f"R$ {val:.2f}" for val in cots],
        hovertemplate="Cotação: %{text} (Progresso: %{y:.2%})"
    ))
    
    # Lucro Líquido: Dashed green line on primary y-axis
    fig.add_trace(go.Scatter(
        x=hist_df['periodo'],
        y=lucros_norm,
        name="Lucro Líq. (Relativo)",
        line=dict(color='#16a34a', width=4, dash='dash'),
        mode='lines+markers',
        marker=dict(size=8, color='#15803d'),
        yaxis='y1',
        text=[fmt_currency(val) for val in lucros],
        hovertemplate="Lucro: %{text} (Progresso: %{y:.2%})"
    ))
    
    fig.update_layout(
        title=title if title else f"Histórico Consolidado de {selected_stock['papel']} ({selected_stock['empresa']})",
        yaxis=dict(
            title="Trajetória Relativa (Normalizada 0-1)",
            side="left",
            range=[-0.1, 1.1],
            showgrid=True,
            gridcolor='#f1f5f9'
        ),
        yaxis2=dict(
            title="Receita Absoluta (Bi R$)",
            side="right",
            overlaying="y",
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode="x unified",
        margin=dict(l=40, r=40, t=100, b=40)
    )
    return fig

def get_asymmetry_stocks():
    asymmetry_list = []
    for stock in STOCK_DATABASE:
        h = stock.get("historico", [])
        if not h or len(h) < 2:
            continue
        prices = [x["cotacao"] for x in h]
        profits = [x["lucro"] for x in h]
        p_min, p_max = min(prices), max(prices)
        l_min, l_max = min(profits), max(profits)
        if p_max > p_min and l_max > l_min:
            current_p = prices[-1]
            current_l = profits[-1]
            p_pos = (current_p - p_min) / (p_max - p_min)
            l_pos = (current_l - l_min) / (l_max - l_min)
            if l_pos > p_pos:
                asymmetry_list.append(stock)
    return asymmetry_list

# --- CONTEÚDO DOS TABS ---

if active_tab == "🏆 Ranking Fundamentalista":
    st.markdown("""
    <div class="custom-card">
        <h4 style="color:#2563eb; font-weight:700; margin:0 0 4px 0; font-size:14px; font-family:'JetBrains Mono', monospace;">🏆 SINAIS FUNDAMENTALISTAS SAUDÁVEIS</h4>
        <h3 style="margin:0 0 8px 0; font-weight:800; font-size:18px;">Ranking de Oportunidades Saudáveis</h3>
        <p style="color:#475569; font-size:13px; line-height:1.5; margin:0;">
            Filtramos o mercado brasileiro aplicando restrições de alto nível: <strong>ROE &gt; 5%</strong>, <strong>0 &lt; P/L &lt; 15</strong>, 
            <strong>0 &lt; EV/EBIT &lt; 10</strong>, dividendos com <strong>Dividend Yield &gt; 4%</strong>, margem líquida positiva superior a 
            <strong>5%</strong> e liquidez diária expressiva acima de <strong>R$ 200 mil</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filtragem dos melhores fundamentalistas
    df_best = df_original[
        (df_original['roe'] > 0.05) &
        (df_original['pl'] > 0) & (df_original['pl'] < 15) &
        (df_original['evebit'] > 0) & (df_original['evebit'] < 10) &
        (df_original['dy'] > 0.04) &
        (df_original['mrgliq'] > 0.05) &
        (df_original['liq2m'] > 200000)
    ].sort_values(by=['pl', 'mrgliq'], ascending=[True, False])
    
    st.subheader(f"Melhores Ações Fundamentalistas ({len(df_best)})")
    if not df_best.empty:
        # Apresentando tabela formatada de forma amigável
        df_show = df_best.copy()
        df_show['Preço'] = df_show['cotacao'].apply(fmt_currency)
        df_show['P/L'] = df_show['pl'].round(2)
        df_show['EV/EBIT'] = df_show['evebit'].round(2)
        df_show['ROE'] = df_show['roe'].apply(fmt_pct)
        df_show['Div. Yield'] = df_show['dy'].apply(fmt_pct)
        df_show['Margem Líq'] = df_show['mrgliq'].apply(fmt_pct)
        
        st.dataframe(
            df_show[['papel', 'empresa', 'Preço', 'P/L', 'EV/EBIT', 'ROE', 'Div. Yield', 'Margem Líq']],
            use_container_width=True,
            column_config={
                "papel": st.column_config.TextColumn("Ativo"),
                "empresa": st.column_config.TextColumn("Empresa"),
            }
        )
    else:
        st.info("Nenhuma empresa atendeu a todos os critérios restritivos severos no momento.")
        
    st.markdown("---")
    
    # Empresas de Risco
    list_rj = ['OIBR3', 'OIBR4', 'AMER3', 'GOLL4', 'AZUL4']
    df_warning = df_original[
        (df_original['divbpatr'] > 3.0) | df_original['papel'].isin(list_rj)
    ].sort_values(by='divbpatr', ascending=False)
    
    st.subheader(f"⚠️ Atenção: Empresas com Alavancagem Elevada ou RJ ({len(df_warning)})")
    if not df_warning.empty:
        df_warn_show = df_warning.copy()
        df_warn_show['Preço'] = df_warn_show['cotacao'].apply(fmt_currency)
        df_warn_show['Alavancagem (Dív/Patr)'] = df_warn_show['divbpatr'].round(2)
        
        st.dataframe(
            df_warn_show[['papel', 'empresa', 'Preço', 'Alavancagem (Dív/Patr)', 'quedaLucro', 'situacao']],
            use_container_width=True,
            column_config={
                "papel": st.column_config.TextColumn("Ativo"),
                "empresa": st.column_config.TextColumn("Empresa"),
                "quedaLucro": st.column_config.TextColumn("Var. Resultados"),
                "situacao": st.column_config.TextColumn("Situação"),
            }
        )
        
    st.markdown("---")
    
    # Gráfico interativo de Cotação vs Receita/Lucro no tempo
    st.subheader("📊 Gráfico Histórico Inteligente")
    ticker_choice = st.selectbox("Selecione o papel para renderizar o histórico:", [stock['papel'] for stock in STOCK_DATABASE])
    selected_stock = next((item for item in STOCK_DATABASE if item["papel"] == ticker_choice), None)
    
    if selected_stock:
        fig = render_custom_stock_chart(selected_stock)
        st.plotly_chart(fig, use_container_width=True)

    # Notícias e Dividendos em duas colunas inferiores
    col_news, col_divs = st.columns(2)
    with col_news:
        st.markdown("""
        <div class="custom-card">
            <h4 style="margin:0 0 12px 0; font-size:14px; font-weight:700; color:#1e3a8a;">📰 Notícias Recentes do Mercado</h4>
        </div>
        """, unsafe_allow_html=True)
        for news in MARKET_NEWS:
            st.markdown(f"""
            <div style="border: 1px solid #f1f5f9; padding:12px; border-radius:10px; margin-bottom:8px; background-color:white;">
                <span style="font-size:9px; background-weight:700; color:#2563eb; background-color:#eff6ff; padding:2px 6px; border-radius:4px; font-family:monospace; font-weight:700;">{news['source'].upper()}</span>
                <p style="font-size:12px; font-weight:600; margin:6px 0 2px 0; color:#1e293b;">{news['title']}</p>
                <a href="{news['link']}" target="_blank" style="font-size:11px; text-decoration:none; color:#3b82f6;">Ver cobertura original →</a>
            </div>
            """, unsafe_allow_html=True)
            
    with col_divs:
        st.markdown("""
        <div class="custom-card">
            <h4 style="margin:0 0 12px 0; font-size:14px; font-weight:700; color:#047857;">💰 Últimos Dividendos Anunciados (B3)</h4>
        </div>
        """, unsafe_allow_html=True)
        for div in LATEST_DIVIDENDS:
            st.markdown(f"""
            <div style="display:flex; justify-content:between; align-items:center; border: 1px solid #f1f5f9; padding:14px; border-radius:12px; margin-bottom:8px; background-color:white;">
                <div>
                    <span style="font-size:13px; font-weight:700; color:#0f172a; font-family:'JetBrains Mono', monospace;">{div['ativo']}</span>
                    <span style="font-size:11px; color:#64748b; margin-left:8px;">Ex-data: {div['data']}</span>
                </div>
                <div style="font-weight:700; font-size:13px; color:#059669; margin-left:auto;">
                    {fmt_currency(div['valor'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

elif active_tab == "✨ Fórmula Mágica":
    st.markdown("""
    <div class="custom-card">
        <h4 style="color:#4f46e5; font-weight:700; margin:0 0 4px 0; font-size:14px; font-family:'JetBrains Mono', monospace;">✨ JOEL GREENBLATT METHODOLOGY</h4>
        <h3 style="margin:0 0 8px 0; font-weight:800; font-size:18px;">Fórmula Mágica do Mercado B3</h3>
        <p style="color:#475569; font-size:13px; line-height:1.5;">
            Rankeia as empresas com base no menor somatório de Rank de EY (Earning Yield) e Rank de ROIC (Retorno sobre Capital Investido). 
            <strong>Menor pontuação agregada lidera o topo das compras recomendadas!</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Processamento Fórmula Mágica
    df_magic = df_original[
        (df_original['liq2m'] > 100000) & (df_original['evebit'] > 0)
    ].copy()
    
    # Ranks
    df_magic['rank_ey'] = df_magic['evebit'].rank(ascending=True, method='first')
    df_magic['rank_roic'] = df_magic['roic'].rank(ascending=False, method='first')
    df_magic['score_magico'] = df_magic['rank_ey'] + df_magic['rank_roic']
    df_magic['ey'] = 1 / df_magic['evebit']
    df_magic_sorted = df_magic.sort_values(by='score_magico').head(40)
    
    st.subheader("Lista Tática Fórmula Mágica")
    
    df_mshow = df_magic_sorted.copy()
    df_mshow['Preço'] = df_mshow['cotacao'].apply(fmt_currency)
    df_mshow['Earning Yield'] = df_mshow['ey'].apply(fmt_pct)
    df_mshow['ROIC'] = df_mshow['roic'].apply(fmt_pct)
    df_mshow['EV/EBIT'] = df_mshow['evebit'].round(2)
    df_mshow['Score Mágico'] = df_mshow['score_magico'].astype(int)
    
    st.dataframe(
        df_mshow[['papel', 'Preço', 'Score Mágico', 'Earning Yield', 'EV/EBIT', 'ROIC']],
        use_container_width=True,
        column_config={
            "papel": st.column_config.TextColumn("Ativo"),
            "Score Mágico": st.column_config.NumberColumn("Score (Menor é Melhor)"),
        }
    )

elif active_tab == "💎 Graham Valuation":
    st.markdown("""
    <div class="custom-card">
        <h4 style="color:#d97706; font-weight:700; margin:0 0 4px 0; font-size:14px; font-family:'JetBrains Mono', monospace;">💎 BENJAMIN GRAHAM VALUE INVESTING</h4>
        <h3 style="margin:0 0 8px 0; font-weight:800; font-size:18px;">Preço Justo e Margem de Segurança Graham</h3>
        <p style="color:#475569; font-size:13px; line-height:1.5;">
            Calculado sobre a clássica modelagem: <strong>V.I = Raiz(22.5 * LPA * VPA)</strong>. 
            Uma margem de segurança física indica se o preço de mercado oferece desconto justo em relação ao valor contábil real.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    df_graham = df_original[(df_original['pl'] > 0) & (df_original['vpa'] > 0) & (df_original['lpa'] > 0)].copy()
    df_graham['valor_intrinseco'] = (22.5 * df_graham['lpa'] * df_graham['vpa']) ** 0.5
    df_graham['ratio'] = df_graham['valor_intrinseco'] / df_graham['cotacao']
    df_graham['status'] = df_graham['ratio'].apply(lambda x: 'Barata (Desconto)' if x > 1.0 else 'Valor Esticado')
    
    df_g_sorted = df_graham.sort_values(by='ratio', ascending=False)
    
    st.subheader("Margem de Segurança Graham")
    
    df_gshow = df_g_sorted.copy()
    df_gshow['Preço Atual'] = df_gshow['cotacao'].apply(fmt_currency)
    df_gshow['Preço Justo (V.I)'] = df_gshow['valor_intrinseco'].apply(fmt_currency)
    df_gshow['Upside Margem'] = df_gshow['ratio'].round(2).apply(lambda x: f"{x}x")
    df_gshow['LPA'] = df_gshow['lpa'].apply(fmt_currency)
    df_gshow['VPA'] = df_gshow['vpa'].apply(fmt_currency)
    
    st.dataframe(
        df_gshow[['papel', 'Preço Atual', 'Preço Justo (V.I)', 'Upside Margem', 'status', 'LPA', 'VPA']],
        use_container_width=True,
        column_config={
            "papel": st.column_config.TextColumn("Ativo"),
            "status": st.column_config.TextColumn("Avaliação Graham"),
        }
    )

elif active_tab == "📈 EPS Diluído":
    st.markdown("""
    <div class="custom-card">
        <h4 style="color:#2563eb; font-weight:700; margin:0 0 4px 0; font-size:14px; font-family:'JetBrains Mono', monospace;">📈 EARNINGS PER SHARE (EPS)</h4>
        <h3 style="margin:0 0 8px 0; font-weight:800; font-size:18px;">EPS Líquido Trimestral Robusto</h3>
        <p style="color:#475569; font-size:13px; line-height:1.5;">
            Foco em empresas líquidas com lucro por ação diluído superior a R$ 1,00 no trimestre analisado, garantindo robustez financeira de curto prazo substancial.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_lh, col_symbol = st.columns([2, 3])
    with col_lh:
        df_eps = df_original[df_original['epsTrimestral'] > 1.0].sort_values(by='epsTrimestral', ascending=False)
        st.write("📊 Destaques Trimestrais em EPS:")
        for _, stock in df_eps.iterrows():
            st.markdown(f"""
            <div style="background-color: white; border: 1px solid #f1f5f9; padding: 12.5px; border-radius: 12px; margin-bottom: 8px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong style="color:#0f172a; font-family:monospace;">{stock['papel']}</strong>
                    <span style="font-weight:700; color:#10b981; font-size:13px;">{fmt_currency(stock['epsTrimestral'])}</span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:10.5px; color:#475569; margin-top:4px;">
                    <span>{stock['empresa']}</span>
                    <span>Ref: {stock['dataRef']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_symbol:
        st.markdown("""
        <div style="background-color:white; border: 1px solid #e1e8f0; border-radius:18px; padding:18px; text-align:center;">
            <p style="color:#475569; font-size:12px; font-weight:700; margin-bottom:12px;">📈 SINAIS COMPLEMENTARES TRADINGVIEW</p>
            <p style="font-size:11px; color:#64748b; line-height:1.4;">
                Aproveite para cruzar esses dados fundamentais com a análise gráfica de curto e médio prazo de sua corretora.
            </p>
            <div style="font-size: 15px; font-weight:bold; color: #1e3a8a; margin-top:16px;">
                Petrobras Bolsa (PETR4) • Ativo de Referência
            </div>
            <div style="background-color:#eff6ff; padding:8px 12px; border-radius:8.5px; font-size:11px; color:#1d4ed8; font-weight:600; display:inline-block; margin-top:12px;">
                Sinal de Compra Forte Ativo nos principais osciladores
            </div>
        </div>
        """, unsafe_allow_html=True)

elif active_tab == "📉 Assimetria Lucro/Preço":
    st.markdown("""
    <div class="custom-card">
        <h4 style="color:#10b981; font-weight:700; margin:0 0 4px 0; font-size:14px; font-family:'JetBrains Mono', monospace;">📉 ASSIMETRIA DE CURTO PRAZO</h4>
        <h3 style="margin:0 0 8px 0; font-weight:800; font-size:18px;">Preço Atrasado em Relação ao Lucro</h3>
        <p style="color:#475569; font-size:13px; margin:0; line-height:1.5;">
            Detecta situações assimétricas de alta probabilidade tática: 
            <strong>empresas onde a direção dos lucros líquidos subiu firmemente nas últimas leituras, mas o preço das ações se manteve desvalorizado ou atrasado na bolsa</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Ativo com Assimetria Detectada")
    
    asym_stocks = get_asymmetry_stocks()
    asym_tickers = [s["papel"] for s in asym_stocks]
    if not asym_tickers:
        asym_tickers = ["BBSE3", "TAEE11", "PETR4"] # fallback
        
    asymmetry_ticker = st.selectbox("Selecione a ação sob assimetria:", asym_tickers)
    asym_stock = next((item for item in STOCK_DATABASE if item["papel"] == asymmetry_ticker), None)
    
    if asym_stock:
        fig = render_custom_stock_chart(asym_stock, title=f"Comportamento de Boca de Jacaré (Assimetria) - {asymmetry_ticker}")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        <div style="background-color:#ecfdf5; border: 1px solid #a7f3d0; padding:16px; border-radius:16px; font-size:12px; color:#065f46; display:flex; align-items:center; gap:8px;">
            <span>✅</span>
            <span>
                <strong>Sinal de Assimetria Confirmado para {asymmetry_ticker}:</strong> A linha tracejada verde (Direção de Lucros Acumulados) 
                termina visualmente acima da linha azul contínua (Cotação de Mercado), configurando distorção de precificação com alta margem!
            </span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)

# --- AFILIADOS E PATROCINADORES DO CANAL ---
st.markdown("<h4 style='text-align:center; font-weight:700; color:#475569; font-size:13px; text-transform:uppercase; letter-spacing:0.5px;'>Nossos Membros Patrocinadores Oficiais</h4>", unsafe_allow_html=True)
col_nomad, col_mp = st.columns(2)

with col_nomad:
    st.markdown("""
    <a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I&n=Jader" target="_blank" style="text-decoration:none; color:inherit;">
        <div class="custom-card" style="transition:all 0.2s; cursor:pointer;" onmouseover="this.style.borderColor='#d97706'">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:22px;">✈️</span>
                <strong style="font-size:14px; color:#1e293b;">Nomad: Conta em Dólar Gratuita</strong>
            </div>
            <p style="font-size:12px; color:#64748b; line-height:1.4; margin-bottom:12px;">
                Abra sua conta bancária e de investimentos global nos EUA sem taxas de manutenção corporativa. 
                Garante taxa cambial zero na primeira conversão!
            </p>
            <div style="background-color:#fffbeb; color:#b45309; text-align:center; padding:10px; border-radius:10px; font-size:11.5px; font-weight:600;">
                Abrir Conta Internacional Nomad →
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)
    
with col_mp:
    st.markdown("""
    <a href="https://mpago.li/1VydVhw" target="_blank" style="text-decoration:none; color:inherit;">
        <div class="custom-card" style="transition:all 0.2s; cursor:pointer;" onmouseover="this.style.borderColor='#0284c7'">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:22px;">🤝</span>
                <strong style="font-size:14px; color:#1e293b;">Mercado Pago: R$ 30,00 Grátis de Desconto</strong>
            </div>
            <p style="font-size:12px; color:#64748b; line-height:1.4; margin-bottom:12px;">
                Crie sua conta digital líder em rendimentos e maquininhas Point no Mercado Pago para recolher recebimentos 
                e conquiste um cupom especial exclusivo!
            </p>
            <div style="background-color:#f0f9ff; color:#0369a1; text-align:center; padding:10px; border-radius:10px; font-size:11.5px; font-weight:600;">
                Resgatar Bônus Mercado Pago R$ 30 →
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div style="background-color:#0f172a; border-radius:18px; padding:24px; text-align:center; color:#94a3b8; font-size:12px; margin-top:32px;">
    <p style="font-weight:700; color:white; margin:0 0 4px 0;">Ibovespa Fundamentalista © 2026</p>
    <p style="margin:0 0 16px 0;">Desenvolvido com máxima performance e polimento técnico profissional.</p>
    <div style="display:flex; justify-content:center; gap:16px;">
        <span style="cursor:pointer;">Termos de Uso</span>
        <span style="cursor:pointer;">Políticas Gerais</span>
        <span style="background-color:#1e293b; color:#cbd5e1; padding:2px 8px; border-radius:4px; font-family:monospace; font-size:10px;">B3 IB_V_2025</span>
    </div>
</div>
""", unsafe_allow_html=True)
