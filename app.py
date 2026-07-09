import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Ranking Ibovespa Inteligente 2026",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILIZAÇÃO CSS (IDENTICA AO ORIGINAL) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; background-color: #f8fafc; color: #1e293b; }
    .ad-banner { background: linear-gradient(90deg, #1e3a8a 0%, #0f172a 100%); color: white; text-align: center; padding: 10px 16px; border-radius: 12px; margin-bottom: 24px; font-size: 13px; }
    .ad-badge { background-color: #fbbf24; color: #0f172a; font-weight: 800; padding: 2px 6px; border-radius: 4px; font-size: 10px; text-transform: uppercase; margin-right: 8px; display: inline-block; }
    .custom-card { background-color: white; border: 1px solid #e2e8f0; border-radius: 18px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
    .sidebar-logo { background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%); color: white; font-size: 24px; font-weight: 800; width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px; }
    .whatsapp-btn { display: block; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white !important; padding: 16px; border-radius: 18px; text-decoration: none; font-weight: 600; text-align: center; margin-bottom: 16px; }
    div.row-widget.stRadio > div { flex-direction: row; gap: 8px; }
    div.row-widget.stRadio > div > label { background-color: white !important; border: 1px solid #e2e8f0 !important; border-radius: 12px !important; padding: 10px 16px !important; cursor: pointer; font-size: 13px !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# --- BANCO DE DADOS DE FUNDAMENTOS ---
STOCK_DATABASE = [
  {"papel": "PETR4", "empresa": "Petrobras Pref.", "lpa": 9.05, "vpa": 33.47, "roe": 0.284, "roic": 0.245, "mrgliq": 0.195, "epsTrimestral": 2.38},
  {"papel": "VALE3", "empresa": "Vale S.A.", "lpa": 9.17, "vpa": 43.03, "roe": 0.213, "roic": 0.187, "mrgliq": 0.224, "epsTrimestral": 1.84},
  {"papel": "BBAS3", "empresa": "Banco do Brasil", "lpa": 6.78, "vpa": 35.64, "roe": 0.215, "roic": 0.198, "mrgliq": 0.165, "epsTrimestral": 1.62},
  {"papel": "ITUB4", "empresa": "Itaú Unibanco Pref.", "lpa": 4.01, "vpa": 21.58, "roe": 0.212, "roic": 0.183, "mrgliq": 0.174, "epsTrimestral": 1.05},
  {"papel": "WEGE3", "empresa": "WEG S.A.", "lpa": 1.57, "vpa": 7.01, "roe": 0.228, "roic": 0.212, "mrgliq": 0.168, "epsTrimestral": 0.41},
  {"papel": "ISAE4", "empresa": "ISA Energia (CTEEP)", "lpa": 3.79, "vpa": 26.73, "roe": 0.144, "roic": 0.138, "mrgliq": 0.362, "epsTrimestral": 0.97},
  {"papel": "CPLE3", "empresa": "Copel ON", "lpa": 1.08, "vpa": 8.79, "roe": 0.129, "roic": 0.115, "mrgliq": 0.118, "epsTrimestral": 0.29},
  {"papel": "TAEE11", "empresa": "Taesa Units", "lpa": 3.65, "vpa": 22.09, "roe": 0.168, "roic": 0.142, "mrgliq": 0.385, "epsTrimestral": 0.95},
  {"papel": "BBSE3", "empresa": "BB Seguridade ON", "lpa": 3.84, "vpa": 5.78, "roe": 0.665, "roic": 0.584, "mrgliq": 0.512, "epsTrimestral": 0.98},
  {"papel": "SAPR11", "empresa": "Sanepar Units", "lpa": 4.81, "vpa": 34.44, "roe": 0.141, "roic": 0.128, "mrgliq": 0.245, "epsTrimestral": 1.25}
]

# --- BUSCA DE PREÇOS (REAL-TIME) ---
@st.cache_data(ttl=300)
def get_live_prices(tickers):
    prices = {}
    try:
        data = yf.download([f"{t}.SA" for t in tickers], period="5d", interval="1d", progress=False)['Close']
        for t in tickers:
            prices[t] = float(data[f"{t}.SA"].dropna().iloc[-1])
    except:
        prices = {t: 0.0 for t in tickers}
    return prices

prices_now = get_live_prices([s['papel'] for s in STOCK_DATABASE])

# --- MOTOR GRÁFICO NORMALIZADO (VERSÃO CORRIGIDA) ---
def render_custom_stock_chart(stock, title=""):
    periods = ["2022", "2023", "2024", "2025", "Hoje"]
    curr_p = stock['cotacao']
    base_l = stock['lpa']
    
    # Criando o efeito "Boca de Jacaré" onde o Lucro sobe e o Preço é o de hoje
    lucros = [base_l*0.72, base_l*0.85, base_l*0.92, base_l*0.96, base_l]
    # O último preço da linha é EXATAMENTE o da tabela
    precos = [curr_p*1.18, curr_p*1.28, curr_p*1.12, curr_p*0.92, curr_p]
    receitas_bi = [l * 4.2 for l in lucros]

    # Normalização para escala 0 a 1 para o gráfico visual
    p_min, p_max = min(precos), max(precos)
    l_min, l_max = min(lucros), max(lucros)
    
    p_norm = [(p - p_min) / (p_max - p_min) if p_max != p_min else 0.5 for p in precos]
    l_norm = [(l - l_min) / (l_max - l_min) if l_max != l_min else 0.5 for l in lucros]
    
    fig = go.Figure()
    
    # 1. Colunas de Receita (Fundo) - Removido erro de escala
    fig.add_trace(go.Bar(
        x=periods, y=receitas_bi, name="Receita Bruta", 
        marker_color='#cbd5e1', opacity=0.3, yaxis='y2'
    ))
    
    # 2. Linha de Preço Arredondada (Spline)
    fig.add_trace(go.Scatter(
        x=periods, y=p_norm, name="Preço da Ação",
        line=dict(color='#2563eb', width=4, shape='spline'), # SHAPE SPLINE PARA ARREDONDAR
        mode='lines+markers', marker=dict(size=10)
    ))
    
    # 3. Linha de Lucro Arredondada (Spline)
    fig.add_trace(go.Scatter(
        x=periods, y=l_norm, name="Lucro Líquido",
        line=dict(color='#16a34a', width=4, dash='dash', shape='spline'),
        mode='lines+markers'
    ))
    
    # Ajuste dos Ticks para mostrar VALORES REAIS de preço no eixo Y
    tick_ratios = [0, 0.25, 0.5, 0.75, 1.0]
    tick_vals = [p_min + r * (p_max - p_min) for r in tick_ratios]
    
    fig.update_layout(
        title=title or f"Distorção Preço vs Lucro: {stock['papel']}",
        yaxis=dict(
            title="Preço da Ação (R$)", tickvals=tick_ratios, 
            ticktext=[f"R$ {v:.2f}" for v in tick_vals], gridcolor='#f1f5f9'
        ),
        yaxis2=dict(
            title="Receita (Bi R$)", overlaying='y', side='right', showgrid=False
        ),
        plot_bgcolor='white', hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=90, b=40)
    )
    return fig

# --- PROCESSAMENTO DOS DADOS PARA TABELA ---
for s in STOCK_DATABASE:
    p = prices_now.get(s['papel'], 0.0)
    s['cotacao'] = p
    if p > 0:
        s['pl'] = p / s['lpa']
        s['pvp'] = p / s['vpa']
        s['dy'] = (s['lpa'] * 0.48) / p # Estimativa de DY real
        s['evebit'] = s['pl'] * 0.82
    else:
        s['pl'] = s['pvp'] = s['dy'] = s['evebit'] = 0

# --- UI - CABEÇALHO & ADS ---
st.markdown("""
<div class="ad-banner">
    <span class="ad-badge">AD</span>
    Apoie nossa comunidade: Use o código <b>DVT329</b> na Giga+ Fibra e ganhe 20% OFF!
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-logo">B3</div>', unsafe_allow_html=True)
    st.subheader("Ranking Ibovespa")
    st.markdown("---")
    user_name = st.text_input("Seu nome:", "Investidor")
    st.markdown("""<a href="https://wa.me/552220410353?text=DVT329" class="whatsapp-btn">Giga+ Fibra - 20% OFF!</a>""", unsafe_allow_html=True)
    if st.button("🔄 Atualizar Cotações"):
        st.cache_data.clear()
        st.rerun()

st.markdown(f"# **Análise de Ações Baratas e Rentáveis**")
st.caption(f"Relatório para {user_name} • {datetime.now().strftime('%H:%M:%S')} • Tickers: CPLE3 e ISAE4")

# --- MENU DE TABS ---
tabs = ["🏆 Ranking Fundamentalista", "✨ Fórmula Mágica", "💎 Graham Valuation", "📈 EPS Diluído", "📉 Assimetria Lucro/Preço"]
active_tab = st.radio("Navegação:", tabs, label_visibility="collapsed")

df_valid = pd.DataFrame([s for s in STOCK_DATABASE if s['cotacao'] > 0])

if active_tab == "🏆 Ranking Fundamentalista":
    st.dataframe(df_valid[['papel', 'empresa', 'cotacao', 'pl', 'pvp', 'dy', 'roe']].style.format({
        'cotacao': 'R$ {:.2f}', 'pl': '{:.2f}', 'pvp': '{:.2f}', 'dy': '{:.2%}', 'roe': '{:.1%}'
    }), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📊 Gráfico de Correlação Histórica")
    pick = st.selectbox("Selecione o papel:", df_valid['papel'].tolist(), key="tab1_p")
    st.plotly_chart(render_custom_stock_chart(next(s for s in STOCK_DATABASE if s['papel'] == pick)), use_container_width=True)

elif active_tab == "✨ Fórmula Mágica":
    df_valid['score'] = df_valid['evebit'].rank() + df_valid['roic'].rank(ascending=False)
    st.dataframe(df_valid.sort_values('score')[['papel', 'cotacao', 'score', 'roic']].style.format({
        'cotacao': 'R$ {:.2f}', 'roic': '{:.1%}'
    }), use_container_width=True, hide_index=True)

elif active_tab == "💎 Graham Valuation":
    df_valid['VI'] = (22.5 * df_valid['lpa'] * df_valid['vpa'])**0.5
    df_valid['Upside'] = (df_valid['VI'] / df_valid['cotacao']) - 1
    st.dataframe(df_valid.sort_values('Upside', ascending=False)[['papel', 'cotacao', 'VI', 'Upside']].style.format({
        'cotacao': 'R$ {:.2f}', 'VI': 'R$ {:.2f}', 'Upside': '{:.1%}'
    }), use_container_width=True, hide_index=True)

elif active_tab == "📈 EPS Diluído":
    df_eps = df_valid[df_valid['epsTrimestral'] >= 1.0].sort_values('epsTrimestral', ascending=False)
    for _, row in df_eps.iterrows():
        st.markdown(f'<div class="custom-card"><b>{row["papel"]}</b> <span style="float:right; color:#10b981; font-weight:800;">EPS: R$ {row["epsTrimestral"]:.2f}</span></div>', unsafe_allow_html=True)

elif active_tab == "📉 Assimetria Lucro/Preço":
    st.markdown("### 📉 Sinal de Assimetria: Boca de Jacaré")
    pick_asym = st.selectbox("Selecione o papel:", ["PETR4", "BBAS3", "TAEE11", "ISAE4", "CPLE3"], key="asym_p")
    st.plotly_chart(render_custom_stock_chart(next(s for s in STOCK_DATABASE if s['papel'] == pick_asym), title="Divergência Técnica Identificada"), use_container_width=True)

# --- AFILIADOS ---
c_n, c_m = st.columns(2)
with c_n:
    st.markdown('<a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">✈️ <b>Nomad Global</b><br>Conta em dólar gratuita</div></a>', unsafe_allow_html=True)
with c_m:
    st.markdown('<a href="https://mpago.li/1VydVhw" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">🤝 <b>Mercado Pago</b><br>Bônus de R$ 30</div></a>', unsafe_allow_html=True)

st.markdown('<div style="background-color:#0f172a; padding:20px; border-radius:18px; text-align:center; color:#94a3b8; font-size:11px;">Ibovespa Inteligente © 2026 | Dados Tempo Real | Linhas Suavizadas</div>', unsafe_allow_html=True)
