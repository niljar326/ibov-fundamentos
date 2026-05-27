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
STOCK_DATABASE = [
    {
        "papel": "PETR4", "empresa": "Petrobras", "cotacao": 38.50, "pl": 4.25, "pvp": 1.15, "evebit": 3.65,
        "roe": 0.284, "roic": 0.245, "dy": 0.142, "mrgliq": 0.195, "liq2m": 1540000000, "divbpatr": 0.78,
        "lpa": 9.05, "vpa": 33.47, "epsTrimestral": 2.38, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável",
        "historico": [
            {"periodo": "2021", "receita": 452e9, "lucro": 106e9, "cotacao": 22.10},
            {"periodo": "2022", "receita": 641e9, "lucro": 188e9, "cotacao": 24.50},
            {"periodo": "2023", "receita": 511e9, "lucro": 124e9, "cotacao": 32.80},
            {"periodo": "2024", "receita": 495e9, "lucro": 118e9, "cotacao": 36.20},
            {"periodo": "Últimos 12m", "receita": 488e9, "lucro": 115e9, "cotacao": 38.50}
        ]
    },
    {
        "papel": "VALE3", "empresa": "Vale", "cotacao": 62.40, "pl": 6.80, "pvp": 1.45, "evebit": 4.90,
        "roe": 0.213, "roic": 0.187, "dy": 0.088, "mrgliq": 0.224, "liq2m": 1250000000, "divbpatr": 0.52,
        "lpa": 9.17, "vpa": 43.03, "epsTrimestral": 1.84, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável",
        "historico": [
            {"periodo": "2021", "receita": 293e9, "lucro": 121e9, "cotacao": 77.20},
            {"periodo": "2022", "receita": 225e9, "lucro": 95e9, "cotacao": 81.30},
            {"periodo": "2023", "receita": 198e9, "lucro": 39e9, "cotacao": 68.40},
            {"periodo": "2024", "receita": 205e9, "lucro": 42e9, "cotacao": 60.50},
            {"periodo": "Últimos 12m", "receita": 211e9, "lucro": 45e9, "cotacao": 62.40}
        ]
    },
    {
        "papel": "BBAS3", "empresa": "Banco do Brasil", "cotacao": 27.80, "pl": 4.10, "pvp": 0.78, "evebit": 3.10,
        "roe": 0.215, "roic": 0.198, "dy": 0.104, "mrgliq": 0.165, "liq2m": 480000000, "divbpatr": 0.12,
        "lpa": 6.78, "vpa": 35.64, "epsTrimestral": 1.62, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável",
        "historico": [
            {"periodo": "2021", "receita": 82e9, "lucro": 19.7e9, "cotacao": 15.60},
            {"periodo": "2022", "receita": 104e9, "lucro": 31.01e9, "cotacao": 18.90},
            {"periodo": "2023", "receita": 122e9, "lucro": 35.5e9, "cotacao": 25.40},
            {"periodo": "2024", "receita": 133e9, "lucro": 38.2e9, "cotacao": 26.95},
            {"periodo": "Últimos 12m", "receita": 138e9, "lucro": 40.1e9, "cotacao": 27.80}
        ]
    },
    {
        "papel": "BBSE3", "empresa": "BB Seguridade", "cotacao": 33.15, "pl": 8.10, "pvp": 5.40, "evebit": 6.20,
        "roe": 0.655, "roic": 0.582, "dy": 0.096, "mrgliq": 0.442, "liq2m": 125000000, "divbpatr": 0.02,
        "lpa": 4.09, "vpa": 6.14, "epsTrimestral": 1.05, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável",
        "historico": [
            {"periodo": "2021", "receita": 12e9, "lucro": 3.9e9, "cotacao": 19.40},
            {"periodo": "2022", "receita": 15e9, "lucro": 6.0e9, "cotacao": 25.10},
            {"periodo": "2023", "receita": 18e9, "lucro": 7.7e9, "cotacao": 31.80},
            {"periodo": "2024", "receita": 19e9, "lucro": 8.1e9, "cotacao": 32.20},
            {"periodo": "Últimos 12m", "receita": 19.5e9, "lucro": 8.3e9, "cotacao": 33.15}
        ]
    },
    {
        "papel": "TAEE11", "empresa": "Taesa", "cotacao": 35.80, "pl": 9.80, "pvp": 1.72, "evebit": 8.10,
        "roe": 0.176, "roic": 0.125, "dy": 0.098, "mrgliq": 0.354, "liq2m": 65000000, "divbpatr": 1.82,
        "lpa": 3.65, "vpa": 20.81, "epsTrimestral": 0.92, "dataRef": "31/12/2024", "quedaLucro": "Estável", "situacao": "Saudável",
        "historico": [
            {"periodo": "2021", "receita": 3.2e9, "lucro": 2.2e9, "cotacao": 31.20},
            {"periodo": "2022", "receita": 2.9e9, "lucro": 1.4e9, "cotacao": 34.50},
            {"periodo": "2023", "receita": 3.4e9, "lucro": 1.3e9, "cotacao": 36.10},
            {"periodo": "2024", "receita": 3.6e9, "lucro": 1.2e9, "cotacao": 35.20},
            {"periodo": "Últimos 12m", "receita": 3.8e9, "lucro": 1.3e9, "cotacao": 35.80}
        ]
    },
    {
        "papel": "OIBR3", "empresa": "Oi S.A. (Rec Dev)", "cotacao": 0.85, "pl": -0.15, "pvp": -0.05, "evebit": -1.10,
        "roe": -0.842, "roic": -0.321, "dy": 0.0, "mrgliq": -0.925, "liq2m": 12000000, "divbpatr": 4.52,
        "lpa": -5.60, "vpa": -17.20, "epsTrimestral": -1.25, "dataRef": "31/12/2024", "quedaLucro": "Forte Queda", "situacao": "Recup. Judicial",
        "historico": [
            {"periodo": "2021", "receita": 17e9, "lucro": -8.4e9, "cotacao": 1.42},
            {"periodo": "2022", "receita": 12e9, "lucro": -19.2e9, "cotacao": 0.95},
            {"periodo": "2023", "receita": 9.5e9, "lucro": -5.4e9, "cotacao": 0.70},
            {"periodo": "2024", "receita": 8.1e9, "lucro": -4.1e9, "cotacao": 0.82},
            {"periodo": "Últimos 12m", "receita": 7.8e9, "lucro": -3.8e9, "cotacao": 0.85}
        ]
    }
]

MARKET_NEWS = [
    {"source": "InfoMoney", "title": "Ibovespa opera em alta impulsionado por commodities e expectativas fiscais", "link": "https://www.infomoney.com.br"},
    {"source": "Money Times", "title": "Dividendo robusto: Melhores pagadoras da semana anunciam cronograma de repasse", "link": "https://www.moneytimes.com.br"},
    {"source": "Valor", "title": "Análise Técnica: PETR4 rompe suporte histórico e mira R$ 41,00", "link": "https://valor.globo.com"}
]

LATEST_DIVIDENDS = [
    {"ativo": "PETR4", "valor": 1.4253, "data": "2025-02-15"},
    {"ativo": "BBAS3", "valor": 0.4521, "data": "2025-02-20"},
    {"ativo": "TAEE11", "valor": 0.8523, "data": "2025-02-28"}
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
    
    st.subheader(f"⚠️ Empresas com Alavancagem Elevada ou RJ ({len(df_warning)})")
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
    
    # Gráfico interativo
    st.subheader("📊 Gráfico Histórico Inteligente")
    ticker_choice = st.selectbox("Selecione o papel para renderizar o histórico:", [stock['papel'] for stock in STOCK_DATABASE])
    selected_stock = next((item for item in STOCK_DATABASE if item["papel"] == ticker_choice), None)
    
    if selected_stock and "historico" in selected_stock:
        hist_df = pd.DataFrame(selected_stock["historico"])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hist_df['periodo'], y=hist_df['receita'] / 1e9,
            name="Receita (Bi R$)", marker_color='#bfdbfe', yaxis='y1'
        ))
        fig.add_trace(go.Bar(
            x=hist_df['periodo'], y=hist_df['lucro'] / 1e9,
            name="Lucro Líquido (Bi R$)", marker_color='#3b82f6', yaxis='y1'
        ))
        fig.add_trace(go.Scatter(
            x=hist_df['periodo'], y=hist_df['cotacao'],
            name="Preço da Ação (R$)", line=dict(color='#ef4444', width=3), yaxis='y2'
        ))
        
        fig.update_layout(
            title=f"Histórico Financeiro - {ticker_choice} ({selected_stock['empresa']})",
            yaxis=dict(title="Valores Operacionais (Bilhões de R$)", side="left"),
            yaxis2=dict(title="Preço da Cotação (R$)", side="right", overlaying="y", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=100, b=40),
            plot_bgcolor='white',
            hovermode="x unified"
        )
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
                <span style="font-size:9px; color:#2563eb; background-color:#eff6ff; padding:2px 6px; border-radius:4px; font-family:monospace; font-weight:700;">{news['source'].upper()}</span>
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
            <div style="display:flex; justify-content:space-between; align-items:center; border: 1px solid #f1f5f9; padding:14px; border-radius:12px; margin-bottom:8px; background-color:white;">
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
    
    df_magic = df_original[
        (df_original['liq2m'] > 100000) & (df_original['evebit'] > 0)
    ].copy()
    
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
    asymmetry_ticker = st.selectbox("Selecione a ação sob assimetria:", ["BBSE3", "TAEE11", "PETR4"])
    
    asym_stock = next((item for item in STOCK_DATABASE if item["papel"] == asymmetry_ticker), None)
    if asym_stock and "historico" in asym_stock:
        hist_df = pd.DataFrame(asym_stock["historico"])
        
        fig = go.Figure()
        
        lucros_norm = (hist_df['lucro'] - hist_df['lucro'].min()) / (hist_df['lucro'].max() - hist_df['lucro'].min())
        cots_norm = (hist_df['cotacao'] - hist_df['cotacao'].min()) / (hist_df['cotacao'].max() - hist_df['cotacao'].min())
        
        fig.add_trace(go.Scatter(
            x=hist_df['periodo'], y=lucros_norm,
            name="Curva de Lucros (Normalizada)", line=dict(color='#10b981', width=4, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=hist_df['periodo'], y=cots_norm,
            name="Curva de Preços (Normalizada)", line=dict(color='#ef4444', width=4)
        ))
        
        fig.update_layout(
            title=f"Comportamento de Boca de Jacaré (Assimetria) - {asymmetry_ticker}",
            yaxis=dict(title="Progressão Relativa de Trajetórias", range=[-0.1, 1.1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='white',
            hovermode="x"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        <div style="background-color:#ecfdf5; border: 1px solid #a7f3d0; padding:16px; border-radius:16px; font-size:12px; color:#065f46; display:flex; align-items:center; gap:8px;">
            <span>✅</span>
            <span>
                <strong>Sinal de Assimetria Confirmado para {asymmetry_ticker}:</strong> A linha tracejada verde (Direção de Lucros Acumulados) 
                termina visualmente acima da linha vermelha contínua (Cotação de Mercado), configurando distorção de precificação com alta margem!
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
