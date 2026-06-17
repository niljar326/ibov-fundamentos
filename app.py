import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import math

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Ranking Ibovespa Inteligente 2026",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS customizada
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }
    
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

    .custom-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
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
</style>
""", unsafe_allow_html=True)

# Banco de dados de Ativos Atualizado para 2026
ORIGINAL_STOCKS = [
  {
    "papel": "PETR4",
    "empresa": "Petrobras",
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
    "dataRef": "1T2026",
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
    "empresa": "Vale",
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
    "dataRef": "1T2026",
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
    "dataRef": "1T2026",
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
    "empresa": "Itaú Unibanco",
    "cotacao": 36.40,
    "pl: ": 8.50,
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
    "dataRef": "1T2026",
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
    "dataRef": "1T2026",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico: ": [],
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
    "empresa": "Itaúsa",
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
    "dataRef": "1T2026",
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
    "empresa": "Taesa S.A.",
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
    "dataRef": "1T2026",
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
    "empresa": "ISA CTEEP",
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
    "dataRef": "1T2026",
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
    "empresa": "Sanepar S.A.",
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
    "dataRef": "1T2026",
    "quedaLucro": "Estável",
    "situacao": "Saudável",
    "historico": [
      { "periodo": "2022", "receita": 5600000000, "lucro": 1150000000, "cotacao": 17.40 },
      { "periodo": "2023", "receita": 6200000000, "lucro": 1450000000, "cotacao": 21.20 },
      { "periodo": "2024", "receita": 6700000000, "lucro": 1550000000, "cotacao": 23.50 },
      { "periodo": "2025", "receita": 7100000000, "lucro": 1680000000, "cotacao": 24.80 },
      { "periodo": "Últimos 12m", "receita": 7420000000, "lucro": 1750000000, "cotacao": 25.20 }
    ]
  }
]

# Proventos Recentes e Notícias de 2026
MARKET_NEWS = [
    {"source": "Valor Econômico", "title": "Copom mantém taxa Selic estável em reuniões de junho de 2026 e sinaliza prudência contínua.", "link": "https://valor.globo.com/"},
    {"source": "InfoMoney", "title": "Cariocação e recuperação de Commodities impulsionam o Ibovespa em junho de 2026.", "link": "https://www.infomoney.com.br/"},
    {"source": "Money Times", "title": "Petrobras (PETR4) avalia distribuição recorde de dividendos extraordinários para o segundo trimestre de 2026.", "link": "https://www.moneytimes.com.br/"}
]

LATEST_DIVIDENDS = [
    {"ativo": "PETR4", "valor": 1.2450, "data": "2026-06-15"},
    {"ativo": "BBAS3", "valor": 0.4850, "data": "2026-06-10"},
    {"ativo": "ITSA4", "valor": 0.1450, "data": "2026-06-02"},
    {"ativo": "TAEE11", "valor": 0.9250, "data": "2026-05-28"},
    {"ativo": "TRPL4", "valor": 0.6800, "data": "2026-05-18"}
]

# Inicialização de Session State
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
    st.caption("Versão Ultra-Polida 2026 (Python Streamlit)")
    
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
            Assine fibra ótica de alta velocidade operada pela Giga+ Fibra e ganhe desconto pelo WhatsApp.
        </p>
        <div style="background-color:rgba(255,255,255,0.18); text-align:center; padding:8px 12px; border-radius:8px; font-size:11px;">
            Resgatar Cupom no WhatsApp →
        </div>
    </a>
    """, unsafe_allow_html=True)

# --- CABEÇALHO DO DASHBOARD ---
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
        <span style="height:8px; width:8px; background-color:#2563eb; border-radius:50%; display:inline-block;"></span>
        <span style="font-size:11px; font-weight:bold; color:#64748b; text-transform:uppercase; font-family:'JetBrains Mono', monospace; letter-spacing:1px;">B3 Ibovespa Inteligente 2026</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("# **Análise de Ações Baratas e Rentáveis**")
    st.markdown("Relatório tático de investimentos • Referência de Junho de 2026")

# --- MENU DE TABS ---
tabs = [
    "🏆 Ranking Fundamentalista",
    "📈 Histórico Consolidado",
]
active_tab = st.radio("Selecione a Visualização:", tabs, label_visibility="collapsed")

# Formatting helpers
def fmt_currency(val):
    return f"R$ {val:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")

def fmt_pct(val):
    return f"{val * 100:.1f}%"

# --- RENDERIZAÇÃO DO GRÁFICO PERSONALIZADO COM PREÇO REAL NO EIXO VERTICAL ---
def render_custom_stock_chart(selected_stock):
    if not selected_stock or "historico" not in selected_stock:
        return go.Figure()
        
    hist_df = pd.DataFrame(selected_stock["historico"])
    
    # Eixo vertical esquerdo agora mapeia o preço real da ação em Reais
    cots = hist_df['cotacao'].values
    lucros_bi = hist_df['lucro'] / 1e9
    receitas_bi = hist_df['receita'] / 1e9
    
    fig = go.Figure()
    
    # Receita: Barras cinzas no fundo (eixo secundário da direita para manter harmonia de proporção)
    fig.add_trace(go.Bar(
        x=hist_df['periodo'],
        y=receitas_bi,
        name="Receita Consolidada (Bi R$)",
        marker=dict(color='#cbd5e1', line=dict(color='#e2e8f0', width=0)),
        yaxis='y2',
        opacity=0.4
    ))
    
    # Preço real da ação: Linha contínua azul com escala no eixo Y esquerdo (Preço em Reais)
    fig.add_trace(go.Scatter(
        x=hist_df['periodo'],
        y=cots,
        name="Preço da Ação (R$ - Eixo Esquerdo)",
        line=dict(color='#2563eb', width=4),
        mode='lines+markers',
        marker=dict(size=8, color='#1d4ed8'),
        yaxis='y1'
    ))
    
    # Lucro Líquido: Linha tracejada verde com escala no eixo Y secundário (da direita)
    fig.add_trace(go.Scatter(
        x=hist_df['periodo'],
        y=lucros_bi,
        name="Lucro Líquido (Bi R$ - Eixo Direito)",
        line=dict(color='#16a34a', width=4, dash='dash'),
        mode='lines+markers',
        marker=dict(size=8, color='#15803d'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=f"Histórico Consolidado de {selected_stock['papel']} ({selected_stock['empresa']})",
        xaxis=dict(type='category', title="Exercícios"),
        yaxis=dict(
            title="Preço da Ação (Em Reais - R$)",
            side="left",
            showgrid=True,
            gridcolor='#f1f5f9',
            titlefont=dict(color="#2563eb"),
            tickfont=dict(color="#2563eb")
        ),
        yaxis2=dict(
            title="Lucros & Receitas (Bilhões de R$)",
            side="right",
            overlaying="y",
            showgrid=False,
            titlefont=dict(color="#16a34a"),
            tickfont=dict(color="#16a34a")
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode="x unified",
        margin=dict(l=40, r=40, t=100, b=40)
    )
    return fig

df_original = pd.DataFrame(ORIGINAL_STOCKS)

if active_tab == "🏆 Ranking Fundamentalista":
    st.markdown("""
    <div class="custom-card">
        <h4 style="color:#2563eb; font-weight:700; margin:0 0 4px 0; font-size:14px; font-family:'JetBrains Mono', monospace;">🏆 SINAIS FUNDAMENTALISTAS SAUDÁVEIS 2026</h4>
        <h3 style="margin:0 0 8px 0; font-weight:800; font-size:18px;">Ranking de Oportunidades Saudáveis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabela formatada
    df_show = df_original.copy()
    df_show['Preço'] = df_show['cotacao'].apply(fmt_currency)
    df_show['P/L'] = df_show['pl'].round(2)
    df_show['EV/EBIT'] = df_show['evebit'].round(2)
    df_show['ROE'] = df_show['roe'].apply(fmt_pct)
    df_show['Div. Yield'] = df_show['dy'].apply(fmt_pct)
    df_show['Margem Líq'] = df_show['mrgliq'].apply(fmt_pct)
    
    st.dataframe(
        df_show[['papel', 'empresa', 'Preço', 'P/L', 'EV/EBIT', 'ROE', 'Div. Yield', 'Margem Líq']],
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Notícias e Dividendos em duas colunas inferiores
    col_news, col_divs = st.columns(2)
    with col_news:
        st.markdown("<h4 style='color:#1e3a8a;'>📰 Notícias Recentes do Mercado de 2026</h4>", unsafe_allow_html=True)
        for news in MARKET_NEWS:
            st.markdown(f"""
            <div style="border: 1px solid #f1f5f9; padding:12px; border-radius:10px; margin-bottom:8px; background-color:white;">
                <span style="font-size:9px; color:#2563eb; background-color:#eff6ff; padding:2px 6px; border-radius:4px; font-weight:700;">{news['source'].upper()}</span>
                <p style="font-size:12px; font-weight:600; margin:6px 0 2px 0; color:#1e293b;">{news['title']}</p>
                <a href="{news['link']}" target="_blank" style="font-size:11px; text-decoration:none; color:#3b82f6;">Ver cobertura original →</a>
            </div>
            """, unsafe_allow_html=True)
            
    with col_divs:
        st.markdown("<h4 style='color:#047857;'>💰 Dividendos Anunciados Recentes</h4>", unsafe_allow_html=True)
        for div in LATEST_DIVIDENDS:
            st.markdown(f"""
            <div style="display:flex; justify-content:between; align-items:center; border: 1px solid #f1f5f9; padding:14px; border-radius:12px; margin-bottom:8px; background-color:white;">
                <div>
                    <span style="font-size:13px; font-weight:700; color:#0f172a; font-family:monospace;">{div['ativo']}</span>
                    <span style="font-size:11px; color:#64748b; margin-left:8px;">Data: {div['data']}</span>
                </div>
                <div style="font-weight:700; font-size:13px; color:#059669; margin-left:auto;">
                    {fmt_currency(div['valor'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

elif active_tab == "📈 Histórico Consolidado":
    st.subheader("📊 Gráfico Histórico Inteligente com Eixo real (R$)")
    ticker_choice = st.selectbox("Selecione o papel para renderizar o histórico:", [stock['papel'] for stock in ORIGINAL_STOCKS])
    selected_stock = next((item for item in ORIGINAL_STOCKS if item["papel"] == ticker_choice), None)
    
    if selected_stock:
        fig = render_custom_stock_chart(selected_stock)
        st.plotly_chart(fig, use_container_width=True)

# --- AFILIADOS E PATROCINADORES DO CANAL ---
st.markdown("<h4 style='text-align:center; font-weight:700; color:#475569; font-size:13px; text-transform:uppercase; letter-spacing:0.5px;'>Parceiros Patrocinadores de Valor</h4>", unsafe_allow_html=True)
col_nomad, col_mp = st.columns(2)

with col_nomad:
    st.markdown("""
    <a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I&n=Jader" target="_blank" style="text-decoration:none; color:inherit;">
        <div class="custom-card" style="transition:all 0.2s; cursor:pointer;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:22px;">✈️</span>
                <strong style="font-size:14px; color:#1e293b;">Nomad: Conta Internacional</strong>
            </div>
            <p style="font-size:12px; color:#64748b; line-height:1.4; margin-bottom:12px;">
                Abra sua conta global em dólares sem taxa na primeira remessa usando o cupom Y39FP3XF8I.
            </p>
            <div style="background-color:#fffbeb; color:#b45309; text-align:center; padding:10px; border-radius:10px; font-size:11.5px; font-weight:600;">
                Abrir Conta Internacional Nomad →
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)
    
with col_mp = st.columns(2)[1]: # fallback mp col
    st.markdown("""
    <a href="https://mpago.li/1VydVhw" target="_blank" style="text-decoration:none; color:inherit;">
        <div class="custom-card" style="transition:all 0.2s; cursor:pointer;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:22px;">🤝</span>
                <strong style="font-size:14px; color:#1e293b;">Mercado Pago: R$ 30,00 Grátis</strong>
            </div>
            <p style="font-size:12px; color:#64748b; line-height:1.4; margin-bottom:12px;">
                Crie sua conta digital líder e ganhe um cupom no seu primeiro pagamento utilizando o app.
            </p>
            <div style="background-color:#f0f9ff; color:#0369a1; text-align:center; padding:10px; border-radius:10px; font-size:11.5px; font-weight:600;">
                Resgatar Bônus Mercado Pago →
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
