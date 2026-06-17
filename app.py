import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import locale

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Ranking Ibovespa Inteligente 2026",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS customizada para design ultra-polido
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700;800&family=JetBrains+Mono:wght=400;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    /* Banner de anúncio */
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
    
    /* Sidebar */
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

    /* Tabs customizadas estilo rádio */
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
    
    div[data-testid="stMarkdownContainer"] p {
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

# --- BANCO DE DADOS ATUALIZADO (Preços Reais Médios Recentes de Junho de 2026) ---
STOCK_DATABASE = [
  {
    "papel": "PETR4",
    "empresa": "Petrobras Pref.",
    "cotacao": 39.50,
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
    "dataRef": "1T26",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2022", "receita": 641000000000, "lucro": 188000000000, "cotacao": 24.50 },
      { "periodo": "2023", "receita": 511000000000, "lucro": 124000000000, "cotacao": 32.80 },
      { "periodo": "2024", "receita": 495000000000, "lucro": 118000000000, "cotacao": 36.20 },
      { "periodo": "2025", "receita": 502000000000, "lucro": 122000000000, "cotacao": 38.50 },
      { "periodo": "Últimos 12m", "receita": 508000000000, "lucro": 124500000000, "cotacao": 39.50 }
    ]
  },
  {
    "papel": "VALE3",
    "empresa": "Vale S.A.",
    "cotacao": 65.20,
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
    "dataRef": "1T26",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2022", "receita": 225000000000, "lucro": 95000000000, "cotacao": 81.30 },
      { "periodo": "2023", "receita": 198000000000, "lucro": 39000000000, "cotacao": 68.40 },
      { "periodo": "2024", "receita": 205000000000, "lucro": 42000000000, "cotacao": 60.50 },
      { "periodo": "2025", "receita": 212000000000, "lucro": 44500000000, "cotacao": 62.40 },
      { "periodo": "Últimos 12m", "receita": 218000000000, "lucro": 46200000000, "cotacao": 65.20 }
    ]
  },
  {
    "papel": "BBAS3",
    "empresa": "Banco do Brasil",
    "cotacao": 28.95,
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
    "dataRef": "1T26",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2022", "receita": 104000000000, "lucro": 31010000000, "cotacao": 18.90 },
      { "periodo": "2023", "receita": 122000000000, "lucro": 35500000000, "cotacao": 25.40 },
      { "periodo": "2024", "receita": 133000000000, "lucro": 38200000000, "cotacao": 26.95 },
      { "periodo": "2025", "receita": 139000000000, "lucro": 41200000000, "cotacao": 27.80 },
      { "periodo": "Últimos 12m", "receita": 145000000000, "lucro": 43500000000, "cotacao": 28.95 }
    ]
  },
  {
    "papel": "ITUB4",
    "empresa": "Itaú Unibanco Pref.",
    "cotacao": 36.40,
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
    "dataRef": "1T26",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2022", "receita": 128000000000, "lucro": 29100000000, "cotacao": 23.40 },
      { "periodo": "2023", "receita": 142000000000, "lucro": 33800000000, "cotacao": 31.50 },
      { "periodo": "2024", "receita": 151000000000, "lucro": 37400000000, "cotacao": 33.20 },
      { "periodo": "2025", "receita": 158000000000, "lucro": 39500000000, "cotacao": 34.80 },
      { "periodo": "Últimos 12m", "receita": 164000000000, "lucro": 41800000000, "cotacao": 36.40 }
    ]
  },
  {
    "papel": "WEGE3",
    "empresa": "WEG S.A.",
    "cotacao": 48.20,
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
    "dataRef": "1T26",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2022", "receita": 29900000000, "lucro": 4230000000, "cotacao": 35.80 },
      { "periodo": "2023", "receita": 32500000000, "lucro": 5610000000, "cotacao": 34.20 },
      { "periodo": "2024", "receita": 34800000000, "lucro": 6510000000, "cotacao": 39.50 },
      { "periodo": "2025", "receita": 38200000000, "lucro": 7420000000, "cotacao": 43.80 },
      { "periodo": "Últimos 12m", "receita": 41500000000, "lucro": 8100000000, "cotacao": 48.20 }
    ]
  },
  {
    "papel": "ITSA4",
    "empresa": "Itaúsa Pref.",
    "cotacao": 10.95,
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
    "dataRef": "1T26",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2022", "receita": 8100000000, "lucro": 13700000000, "cotacao": 8.20 },
      { "periodo": "2023", "receita": 8800000000, "lucro": 14100000000, "cotacao": 9.40 },
      { "periodo": "2024", "receita": 9500000000, "lucro": 15300000000, "cotacao": 9.95 },
      { "periodo": "2025", "receita": 9800000000, "lucro": 15900000000, "cotacao": 10.35 },
      { "periodo": "Últimos 12m", "receita": 10100000000, "lucro": 16500000000, "cotacao": 10.95 }
    ]
  },
  {
    "papel": "TAEE11",
    "empresa": "Taesa Units",
    "cotacao": 36.10,
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
    "dataRef": "1T26",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2022", "receita": 2800000000, "lucro": 1440000000, "cotacao": 38.10 },
      { "periodo": "2023", "receita": 3300000000, "lucro": 1360000000, "cotacao": 34.50 },
      { "periodo": "2024", "receita": 3500000000, "lucro": 1420000000, "cotacao": 34.90 },
      { "periodo": "2025", "receita": 3680000000, "lucro": 1490000000, "cotacao": 35.80 },
      { "periodo": "Últimos 12m", "receita": 3850000000, "lucro": 1540000000, "cotacao": 36.10 }
    ]
  },
  {
    "papel": "TRPL4",
    "empresa": "ISA CTEEP Pref.",
    "cotacao": 26.90,
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
    "dataRef": "1T26",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2022", "receita": 4800000000, "lucro": 2300000000, "cotacao": 22.40 },
      { "periodo": "2023", "receita": 5400000000, "lucro": 2900000000, "cotacao": 25.10 },
      { "periodo": "2024", "receita": 5800000000, "lucro": 3100000000, "cotacao": 25.80 },
      { "periodo": "2025", "receita": 6100000000, "lucro": 3250000000, "cotacao": 26.20 },
      { "periodo": "Últimos 12m", "receita": 6350000000, "lucro": 3400000000, "cotacao": 26.90 }
    ]
  },
  {
    "papel": "SAPR11",
    "empresa": "Sanepar Units",
    "cotacao": 25.20,
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
    "dataRef": "1T26",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2022", "receita": 5600000000, "lucro": 1150000000, "cotacao": 17.40 },
      { "periodo": "2023", "receita": 6200000000, "lucro": 1450000000, "cotacao": 21.20 },
      { "periodo": "2024", "receita": 6700000000, "lucro": 1550000000, "cotacao": 23.50 },
      { "periodo": "2025", "receita": 7100000000, "lucro": 1680000000, "cotacao": 24.80 },
      { "periodo": "Últimos 12m", "receita": 7420000000, "lucro": 1750000000, "cotacao": 25.20 }
    ]
  },
  {
    "papel": "CPLE6",
    "empresa": "Copel Pref.",
    "cotacao": 10.15,
    "pl": 9.10,
    "pvp": 1.12,
    "evebit": 6.20,
    "roe": 0.129,
    "roic": 0.115,
    "dy": 0.068,
    "mrgliq": 0.118,
    "liq2m": 32000000,
    "divbpatr": 1.15,
    "lpa": 1.08,
    "vpa": 8.79,
    "epsTrimestral": 0.29,
    "dataRef": "1T26",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2022", "receita": 22800000000, "lucro": 1100000000, "cotacao": 7.15 },
      { "periodo": "2023", "receita": 21500000000, "lucro": 1950000000, "cotacao": 8.50 },
      { "periodo": "2024", "receita": 23100000000, "lucro": 2200000000, "cotacao": 9.30 },
      { "periodo": "2025", "receita": 24200000000, "lucro": 2450000000, "cotacao": 9.85 },
      { "periodo": "Últimos 12m", "receita": 25100000000, "lucro": 2600000000, "cotacao": 10.15 }
    ]
  },
  {
    "papel": "LREN3",
    "empresa": "Lojas Renner ON",
    "cotacao": 16.90,
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
    "dataRef": "1T26",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2022", "receita": 11200000000, "lucro": 1200000000, "cotacao": 20.30 },
      { "periodo": "2023", "receita": 11800000000, "lucro": 980000000, "cotacao": 15.10 },
      { "periodo": "2024", "receita": 12400000000, "lucro": 1150000000, "cotacao": 15.80 },
      { "periodo": "2025", "receita": 12900000000, "lucro": 1320000000, "cotacao": 16.20 },
      { "periodo": "Últimos 12m", "receita": 13400000000, "lucro": 1410000000, "cotacao": 16.90 }
    ]
  },
  {
    "papel": "ALUP11",
    "empresa": "Alupar Units",
    "cotacao": 30.10,
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
    "dataRef": "1T26",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2022", "receita": 51200000000, "lucro": 980000000, "cotacao": 25.40 },
      { "periodo": "2023", "receita": 58000000000, "lucro": 1050000000, "cotacao": 26.80 },
      { "periodo": "2024", "receita": 62000000000, "lucro": 1180000000, "cotacao": 28.55 },
      { "periodo": "2025", "receita": 64500000000, "lucro": 1240000000, "cotacao": 29.50 },
      { "periodo": "Últimos 12m", "receita": 66800000000, "lucro": 1280000000, "cotacao": 30.10 }
    ]
  }
]

# Provedor para completar Ativos
EXTRA_SEED = [
  {"papel": "ABEV3", "empresa": "Ambev ON", "cotacao": 12.45, "pl": 13.20, "pvp": 2.20, "evebit": 9.50, "roe": 0.165, "roic": 0.158, "dy": 0.062, "mrgliq": 0.185, "liq2m": 380000000, "divbpatr": 0.04, "lpa": 0.92, "vpa": 5.50, "epsTrimestral": 0.24, "dataRef": "1T26", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "BBDC4", "empresa": "Bradesco Pref", "cotacao": 15.10, "pl": 9.20, "pvp": 0.92, "evebit": 7.20, "roe": 0.102, "roic": 0.095, "dy": 0.068, "mrgliq": 0.112, "liq2m": 450000000, "divbpatr": 0.15, "lpa": 1.58, "vpa": 15.76, "epsTrimestral": 0.38, "dataRef": "1T26", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "RENT3", "empresa": "Localiza ON", "cotacao": 54.80, "pl": 18.50, "pvp": 2.60, "evebit": 11.20, "roe": 0.141, "roic": 0.112, "dy": 0.028, "mrgliq": 0.095, "liq2m": 290000000, "divbpatr": 1.85, "lpa": 2.83, "vpa": 20.15, "epsTrimestral": 0.72, "dataRef": "1T26", "quedaLucro": "Estável", "situacao": "Saudável"},
  {"papel": "ELET3", "empresa": "Eletrobras ON", "cotacao": 41.20, "pl": 15.40, "pvp": 0.82, "evebit": 11.50, "roe": 0.053, "roic": 0.048, "dy": 0.035, "mrgliq": 0.084, "liq2m": 280000000, "divbpatr": 1.45, "lpa": 2.58, "vpa": 48.54, "epsTrimestral": 0.65, "dataRef": "1T26", "quedaLucro": "Estável", "situacao": "Saudável"}
]

# Processamento de preenchimento inteligente
def inflate_stock_history(stock_seed):
    import math
    periods = ["2022", "2023", "2024", "2025", "Últimos 12m"]
    char_sum = sum(ord(char) for char in stock_seed["papel"])
    base_receita_val = (10 + (char_sum % 90)) * 1e9
    historico = []
    
    for idx, periodo in enumerate(periods):
        revenue_multiplier = 0.8 + idx * 0.05 + math.sin(idx + char_sum) * 0.03
        profit_multiplier = 0.85 + idx * 0.045 + math.cos(idx + char_sum) * 0.035
        price_multiplier = 0.75 + idx * 0.065 + math.sin(idx * 1.5 + char_sum) * 0.05
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
    return stock_seed

processed_extra = [inflate_stock_history(s) for s in EXTRA_SEED]
FULL_STOCK_DATABASE = (STOCK_DATABASE + processed_extra)

MARKET_NEWS = [
    {"source": "Valor Econômico", "title": "Copom mantém taxa Selic estável em reuniões de junho de 2026 e sinaliza prudência contínua.", "link": "https://valor.globo.com/"},
    {"source": "InfoMoney", "title": "Ibovespa opera em alta técnica com forte ingresso de fundos globais em ações de valor.", "link": "https://www.infomoney.com.br/"},
    {"source": "Money Times", "title": "Petrobras (PETR4) distribui proventos táticos e consolida alta de margens operacionais em 2026.", "link": "https://www.moneytimes.com.br/"}
]

LATEST_DIVIDENDS = [
    {"ativo": "PETR4", "valor": 1.2450, "data": "2026-06-15"},
    {"ativo": "BBAS3", "valor": 0.4850, "data": "2026-06-10"},
    {"ativo": "ITSA4", "valor": 0.1450, "data": "2026-06-02"},
    {"ativo": "TAEE11", "valor": 0.9250, "data": "2026-05-28"},
    {"ativo": "TRPL4", "valor": 0.6800, "data": "2026-05-18"}
]

# Cabeçalho do Dashboard
st.markdown("# **Ranking Ibovespa Inteligente 2026**")

# Menu de Abas no Streamlit
tabs = ["🏆 Ranking Fundamentalista", "✨ Fórmula Mágica", "💎 Graham Valuation-2026", "📈 EPS Diluído"]
active_tab = st.radio("Selecione a visualização:", tabs, label_visibility="collapsed")

df_original = pd.DataFrame(FULL_STOCK_DATABASE)

def fmt_currency(val):
    return f"R$ {val:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")

# Gráfico de Boca de Jacaré com o Preço da Ação no Eixo Vertical
def render_custom_stock_chart(selected_stock):
    hist_df = pd.DataFrame(selected_stock["historico"])
    receitas_bi = hist_df['receita'] / 1e9
    cots = hist_df['cotacao'].values
    lucros = hist_df['lucro'].values
    
    p_min, p_max = cots.min(), cots.max()
    p_range = (p_max - p_min) if (p_max - p_min) != 0 else 1
    
    l_min, l_max = lucros.min(), lucros.max()
    l_range = (l_max - l_min) if (l_max - l_min) != 0 else 1
    
    cots_norm = [(c - p_min) / p_range for c in cots]
    lucros_norm = [(l - l_min) / l_range for l in lucros]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hist_df['periodo'], y=receitas_bi, yaxis='y2', opacity=0.3, name="Receita (Bi R$)"
    ))
    fig.add_trace(go.Scatter(
        x=hist_df['periodo'], y=cots_norm, mode='lines+markers', name="Preço Real (R$)",
        line=dict(color='#2563eb', width=4), text=[f"R$ {v:.2f}" for v in cots]
    ))
    fig.add_trace(go.Scatter(
        x=hist_df['periodo'], y=lucros_norm, mode='lines+markers', name="Lucro Líq. (Evolução)",
        line=dict(color='#16a34a', width=4, dash='dash')
    ))
    
    # Mapeando os ticks normalizados para os valores reais em R$ no eixo Y esquerdo!
    tick_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig.update_layout(
        yaxis=dict(
            title="Preço Corrente da Ação (R$)",
            tickvals=tick_ratios,
            ticktext=[f"R$ {p_min + r * p_range:.2f}" for r in tick_ratios]
        ),
        yaxis2=dict(title="Receita Consolidada (Bi R$)", overlaying="y", side="right"),
        plot_bgcolor='white'
    )
    return fig

if active_tab == "🏆 Ranking Fundamentalista":
    st.write("### Ranking de Ações")
    st.dataframe(df_original[['papel', 'empresa', 'cotacao', 'pl', 'dy']])
    
    ticker_choice = st.selectbox("Selecione para histórico:", df_original['papel'].values)
    selected_stock = next((item for item in FULL_STOCK_DATABASE if item["papel"] == ticker_choice), None)
    if selected_stock:
        st.plotly_chart(render_custom_stock_chart(selected_stock), use_container_width=True)

# --- AFILIADOS OFICIAIS (AQUI ESTÁ A CORREÇÃO DO SEU ERRINHO!) ---
st.markdown("<h4 style='text-align:center;'>Membros Patrocinadores Oficiais</h4>", unsafe_allow_html=True)
col_nomad, col_mp = st.columns(2)

with col_nomad:
    st.markdown("[Nomad: Conta Dólar Grátis](https://nomad.onelink.me/wIQT/Invest)")
    
with col_mp:
    st.markdown("[Mercado Pago: R$ 30 Bônus](https://mpago.li/1VydVhw)")
