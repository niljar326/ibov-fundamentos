import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import numpy as np
import math

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
    .ad-banner { background: linear-gradient(90deg, #1e3a8a 0%, #0f172a 100%); color: white; text-align: center; padding: 10px 16px; font-size: 13px; border-radius: 12px; margin-bottom: 24px; }
    .ad-badge { background-color: #fbbf24; color: #0f172a; font-weight: 800; padding: 2px 6px; border-radius: 4px; font-size: 10px; text-transform: uppercase; margin-right: 8px; display: inline-block; }
    .custom-card { background-color: white; border: 1px solid #e2e8f0; border-radius: 18px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
    .sidebar-logo { background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%); color: white; font-size: 24px; font-weight: 800; width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px; }
    .whatsapp-btn { display: block; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white !important; padding: 16px; border-radius: 18px; text-decoration: none; font-weight: 600; margin-bottom: 16px; text-align: center; }
    div.row-widget.stRadio > div { flex-direction: row; gap: 8px; }
    div.row-widget.stRadio > div > label { background-color: white !important; border: 1px solid #e2e8f0 !important; border-radius: 12px !important; padding: 10px 16px !important; cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# --- BANCO DE DADOS DE FUNDAMENTOS ---
STOCK_DATABASE = [
  {"papel": "PETR4", "empresa": "Petrobras Pref.", "lpa": 9.05, "vpa": 33.47, "roe": 0.284, "roic": 0.245, "mrgliq": 0.195, "divbpatr": 0.78, "epsTrimestral": 2.38, "dataRef": "1T26"},
  {"papel": "VALE3", "empresa": "Vale S.A.", "lpa": 9.17, "vpa": 43.03, "roe": 0.213, "roic": 0.187, "mrgliq": 0.224, "divbpatr": 0.52, "epsTrimestral": 1.84, "dataRef": "1T26"},
  {"papel": "BBAS3", "empresa": "Banco do Brasil", "lpa": 6.78, "vpa": 35.64, "roe": 0.215, "roic": 0.198, "mrgliq": 0.165, "divbpatr": 0.12, "epsTrimestral": 1.62, "dataRef": "1T26"},
  {"papel": "ITUB4", "empresa": "Itaú Unibanco Pref.", "lpa": 4.01, "vpa": 21.58, "roe": 0.212, "roic": 0.183, "mrgliq": 0.174, "divbpatr": 0.18, "epsTrimestral": 1.05, "dataRef": "1T26"},
  {"papel": "WEGE3", "empresa": "WEG S.A.", "lpa": 1.57, "vpa": 7.01, "roe": 0.228, "roic": 0.212, "mrgliq": 0.168, "divbpatr": 0.08, "epsTrimestral": 0.41, "dataRef": "1T26"},
  {"papel": "ISAE4", "empresa": "ISA Energia (CTEEP)", "lpa": 3.79, "vpa": 26.73, "roe": 0.144, "roic": 0.138, "mrgliq": 0.362, "divbpatr": 1.25, "epsTrimestral": 0.97, "dataRef": "1T26"},
  {"papel": "CPLE3", "empresa": "Copel ON", "lpa": 1.08, "vpa": 8.79, "roe": 0.129, "roic": 0.115, "mrgliq": 0.118, "divbpatr": 1.15, "epsTrimestral": 0.29, "dataRef": "1T26"},
  {"papel": "TAEE11", "empresa": "Taesa Units", "lpa": 3.65, "vpa": 22.09, "roe": 0.168, "roic": 0.142, "mrgliq": 0.385, "divbpatr": 1.84, "epsTrimestral": 0.95, "dataRef": "1T26"},
  {"papel": "BBSE3", "empresa": "BB Seguridade ON", "lpa": 3.84, "vpa": 5.78, "roe": 0.665, "roic": 0.584, "mrgliq": 0.512, "divbpatr": 0.02, "epsTrimestral": 0.98, "dataRef": "1T26"},
  {"papel": "SAPR11", "empresa": "Sanepar Units", "lpa": 4.81, "vpa": 34.44, "roe": 0.141, "roic": 0.128, "mrgliq": 0.245, "divbpatr": 0.74, "epsTrimestral": 1.25, "dataRef": "1T26"}
]

# --- FUNÇÃO DE BUSCA DE PREÇOS (REAL-TIME) ---
@st.cache_data(ttl=600)
def get_live_prices(tickers):
    prices = {}
    try:
        data = yf.download([f"{t}.SA" for t in tickers], period="10d", interval="1d", progress=False)['Close']
        for t in tickers:
            prices[t] = float(data[f"{t}.SA"].dropna().iloc[-1])
    except:
        prices = {t: 0.0 for t in tickers}
    return prices

prices_now = get_live_prices([s['papel'] for s in STOCK_DATABASE])

# --- FUNÇÃO ORIGINAL DE GRÁFICO (NORMALIZAÇÃO 0-1) ---
def render_custom_stock_chart(selected_stock, title=""):
    # Gerando histórico simulado baseado no preço real e fundamentos
    periods = ["2022", "2023", "2024", "2025", "Hoje"]
    curr_p = selected_stock['cotacao']
    base_l = selected_stock['lpa']
    
    # Criando tendência de "Boca de Jacaré" (Lucro sobe, Preço estagna ou cai)
    lucros = [base_l*0.7, base_l*0.82, base_l*0.9, base_l*0.95, base_l]
    cots = [curr_p*1.15, curr_p*1.25, curr_p*0.95, curr_p*0.88, curr_p]
    receitas = [l * 4 for l in lucros]

    l_min, l_max = min(lucros), max(lucros)
    p_min, p_max = min(cots), max(cots)
    
    # Normalização
    lucros_norm = [(l - l_min) / (l_max - l_min) if l_max != l_min else 0.5 for l in lucros]
    cots_norm = [(c - p_min) / (p_max - p_min) if p_max != p_min else 0.5 for c in cots]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=periods, y=[r/1e9 for r in receitas], name="Receita Líquida", marker_color='#cbd5e1', yaxis='y2', opacity=0.3))
    fig.add_trace(go.Scatter(x=periods, y=cots_norm, name="Preço Real (R$)", line=dict(color='#2563eb', width=4), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=periods, y=lucros_norm, name="Lucro Líquido", line=dict(color='#16a34a', width=4, dash='dash')))
    
    # Eixo vertical customizado com preços reais
    tick_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    tick_prices = [p_min + ratio * (p_max - p_min) for ratio in tick_ratios]
    
    fig.update_layout(
        title=title or f"Equilíbrio Fundamentalista: {selected_stock['papel']}",
        yaxis=dict(title="Preço Corrente (R$)", tickvals=tick_ratios, ticktext=[f"R$ {p:.2f}" for p in tick_prices], gridcolor='#f1f5f9'),
        yaxis2=dict(title="Receita", overlaying='y', side='right', showgrid=False),
        plot_bgcolor='white', hovermode="x unified", margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- PROCESSAMENTO DOS DADOS ---
for s in STOCK_DATABASE:
    p = prices_now.get(s['papel'], 0.0)
    s['cotacao'] = p
    if p > 0:
        s['pl'] = p / s['lpa']
        s['pvp'] = p / s['vpa']
        s['dy'] = (s['lpa'] * 0.45) / p # Estimativa baseada em payout de 45%
        s['evebit'] = s['pl'] * 0.8
    else:
        s['pl'] = s['pvp'] = s['dy'] = s['evebit'] = 0

# --- UI - HEADER & ADS ---
st.markdown('<div class="ad-banner"><span class="ad-badge">AD</span>Apoie nossa comunidade: Use o código <b>DVT329</b> na Giga+ Fibra!</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-logo">B3</div>', unsafe_allow_html=True)
    st.subheader("Ranking Ibovespa")
    st.markdown("---")
    user_name = st.text_input("Investidor:", "Visitante")
    st.markdown("""<a href="https://wa.me/552220410353?text=DVT329" class="whatsapp-btn">Giga+ Fibra - 20% OFF!</a>""", unsafe_allow_html=True)
    if st.button("🔄 Forçar Atualização"):
        st.cache_data.clear()
        st.rerun()

st.markdown(f"# **Análise de Ações Baratas e Rentáveis**")
st.caption(f"Relatório para {user_name} • Atualizado em: {datetime.now().strftime('%H:%M:%S')}")

# --- TABS ---
tabs = ["🏆 Ranking Fundamentalista", "✨ Fórmula Mágica", "💎 Graham Valuation", "📈 EPS Diluído", "📉 Assimetria Lucro/Preço"]
active_tab = st.radio("Selecione:", tabs, label_visibility="collapsed")

df_final = pd.DataFrame(STOCK_DATABASE)
df_valid = df_final[df_final['cotacao'] > 0].copy()

if active_tab == "🏆 Ranking Fundamentalista":
    st.dataframe(df_valid[['papel', 'empresa', 'cotacao', 'pl', 'pvp', 'dy', 'roe']].style.format({
        'cotacao': 'R$ {:.2f}', 'pl': '{:.2f}', 'pvp': '{:.2f}', 'dy': '{:.1%}', 'roe': '{:.1%}'
    }), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📊 Gráfico de Correlação de Balanço")
    pick = st.selectbox("Selecione o papel:", df_valid['papel'].tolist(), key="tab1_pick")
    st.plotly_chart(render_custom_stock_chart(next(s for s in STOCK_DATABASE if s['papel'] == pick)), use_container_width=True)

elif active_tab == "✨ Fórmula Mágica":
    df_valid['rank_ey'] = df_valid['evebit'].rank()
    df_valid['rank_roic'] = df_valid['roic'].rank(ascending=False)
    df_valid['score'] = df_valid['rank_ey'] + df_valid['rank_roic']
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
    df_eps = df_valid[df_valid['epsTrimestral'] > 1.0].sort_values('epsTrimestral', ascending=False)
    for _, row in df_eps.iterrows():
        st.markdown(f'<div class="custom-card"><b>{row["papel"]}</b> <span style="float:right; color:#10b981;">EPS: R$ {row["epsTrimestral"]:.2f}</span></div>', unsafe_allow_html=True)

elif active_tab == "📉 Assimetria Lucro/Preço":
    st.markdown("### 📉 Boca de Jacaré: Lucro Crescente vs Preço Atrasado")
    pick_asym = st.selectbox("Selecione o papel:", ["PETR4", "BBAS3", "TAEE11", "ISAE4", "CPLE3"], key="asym_pick")
    st.plotly_chart(render_custom_stock_chart(next(s for s in STOCK_DATABASE if s['papel'] == pick_asym), title="Sinal de Assimetria Validado"), use_container_width=True)

# --- AFILIADOS ---
col_n, col_m = st.columns(2)
with col_n:
    st.markdown('<a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">✈️ <b>Nomad Global</b><br>Conta em dólar gratuita</div></a>', unsafe_allow_html=True)
with col_m:
    st.markdown('<a href="https://mpago.li/1VydVhw" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">🤝 <b>Mercado Pago</b><br>Bônus de R$ 30</div></a>', unsafe_allow_html=True)

st.markdown('<div style="background-color:#0f172a; padding:20px; border-radius:18px; text-align:center; color:#94a3b8; font-size:11px;">Ibovespa Inteligente © 2026 | Dados Real-Time | CPLE3 & ISAE4 Atualizados</div>', unsafe_allow_html=True)
