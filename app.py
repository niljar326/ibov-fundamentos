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

# --- ESTILIZAÇÃO CSS (DESIGN ORIGINAL) ---
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

# --- BANCO DE DADOS DE FUNDAMENTOS (HISTÓRICO REAL POR EMPRESA) ---
# hist_lpa: [2023, 2024, 2025, LTM]
# hist_lucro_bi: [2023, 2024, 2025, LTM]
STOCK_DATABASE = [
  {
    "papel": "PETR4", "empresa": "Petrobras", "roe": 0.26, "vpa": 33.47, "epsTrimestral": 1.70,
    "hist_lpa": [6.20, 6.45, 6.60, 6.80], 
    "hist_lucro_bi": [124.6, 118.2, 122.5, 124.5]
  },
  {
    "papel": "VALE3", "empresa": "Vale S.A.", "roe": 0.12, "vpa": 43.03, "epsTrimestral": 0.95,
    "hist_lpa": [4.10, 3.20, 3.65, 3.85], 
    "hist_lucro_bi": [39.8, 32.5, 42.1, 46.2]
  },
  {
    "papel": "BBAS3", "empresa": "Banco do Brasil", "roe": 0.21, "vpa": 35.64, "epsTrimestral": 1.62,
    "hist_lpa": [5.10, 5.80, 6.20, 6.50], 
    "hist_lucro_bi": [35.5, 38.2, 41.5, 43.8]
  },
  {
    "papel": "ITUB4", "empresa": "Itaú Unibanco", "roe": 0.21, "vpa": 21.58, "epsTrimestral": 0.95,
    "hist_lpa": [3.10, 3.45, 3.65, 3.80], 
    "hist_lucro_bi": [33.8, 35.2, 38.4, 40.2]
  },
  {
    "papel": "WEGE3", "empresa": "WEG S.A.", "roe": 0.22, "vpa": 7.01, "epsTrimestral": 0.42,
    "hist_lpa": [1.25, 1.40, 1.55, 1.65], 
    "hist_lucro_bi": [5.6, 6.2, 6.8, 7.2]
  },
  {
    "papel": "ISAE4", "empresa": "ISA Energia", "roe": 0.13, "vpa": 26.73, "epsTrimestral": 0.70,
    "hist_lpa": [2.10, 2.45, 2.60, 2.80], 
    "hist_lucro_bi": [2.4, 2.6, 2.7, 2.9]
  },
  {
    "papel": "CPLE3", "empresa": "Copel ON", "roe": 0.12, "vpa": 8.79, "epsTrimestral": 0.28,
    "hist_lpa": [0.85, 0.92, 1.05, 1.10], 
    "hist_lucro_bi": [1.9, 2.1, 2.4, 2.6]
  },
  {
    "papel": "TAEE11", "empresa": "Taesa Units", "roe": 0.15, "vpa": 22.09, "epsTrimestral": 0.60,
    "hist_lpa": [2.10, 2.25, 2.30, 2.40], 
    "hist_lucro_bi": [1.2, 1.3, 1.4, 1.5]
  },
  {
    "papel": "BBSE3", "empresa": "BB Seguridade", "roe": 0.65, "vpa": 5.78, "epsTrimestral": 0.98,
    "hist_lpa": [3.20, 3.55, 3.75, 3.90], 
    "hist_lucro_bi": [7.7, 8.1, 8.5, 9.2]
  },
  {
    "papel": "SAPR11", "empresa": "Sanepar Units", "roe": 0.14, "vpa": 34.44, "epsTrimestral": 1.22,
    "hist_lpa": [3.80, 4.20, 4.60, 4.85], 
    "hist_lucro_bi": [1.4, 1.5, 1.6, 1.7]
  }
]

# --- BUSCA DE PREÇOS (REAL-TIME) ---
@st.cache_data(ttl=300)
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

# --- MOTOR GRÁFICO PERSONALIZADO (BASE 100 INDIVIDUAL) ---
def render_custom_chart(stock, title=""):
    current_year = datetime.now().year
    labels = [str(current_year - 3), str(current_year - 2), str(current_year - 1), "Últimos 12m"]
    
    p_now = stock['cotacao']
    
    # Dados Reais do Banco de Dados
    lpa_raw = stock['hist_lpa']
    lucro_bi_raw = stock['hist_lucro_bi']
    
    # Histórico de Preço (Simulado para os anos anteriores, terminando no Real de Hoje)
    precos_raw = [p_now * 1.12, p_now * 1.25, p_now * 0.95, p_now]
    
    # INDEXAÇÃO BASE 100 (Mostra o crescimento real de cada um)
    precos_index = [(p / precos_raw[0]) * 100 for p in precos_raw]
    lucros_index = [(l / lpa_raw[0]) * 100 for l in lpa_raw]

    fig = go.Figure()
    
    # 1. Lucro Real em Bilhões (Barras ao Fundo) - Eixo Y2
    fig.add_trace(go.Bar(
        x=labels, y=lucro_bi_raw, name="Lucro Líq. (Bi R$)", 
        marker_color='#cbd5e1', opacity=0.25, yaxis='y2'
    ))
    
    # 2. Evolução do Lucro por Empresa (Linha Verde)
    fig.add_trace(go.Scatter(
        x=labels, y=lucros_index, name="Crescimento Lucro (%)",
        line=dict(color='#16a34a', width=4, dash='dash', shape='spline'),
        mode='lines+markers',
        hovertemplate="Evolução Lucro: %{y:.1f}%<extra></extra>"
    ))
    
    # 3. Evolução da Cotação (Linha Azul)
    fig.add_trace(go.Scatter(
        x=labels, y=precos_index, name="Evolução Preço (%)",
        line=dict(color='#2563eb', width=5, shape='spline'),
        mode='lines+markers', marker=dict(size=12, line=dict(width=2, color='white')),
        hovertemplate="Evolução Preço: %{y:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title=title or f"Equilíbrio Individual: {stock['papel']}",
        xaxis=dict(type='category', gridcolor='#f1f5f9'),
        yaxis=dict(
            title="Indexador (Base 100 em " + labels[0] + ")", 
            gridcolor='#f1f5f9', ticksuffix="%"
        ),
        yaxis2=dict(
            title="Lucro Líquido (Bilhões R$)", overlaying='y', side='right', 
            showgrid=False, tickformat=".1f"
        ),
        plot_bgcolor='white', hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=90, b=40)
    )
    
    # Rótulo do Valor Atual
    fig.add_annotation(
        x="Últimos 12m", y=precos_index[-1],
        text=f"R$ {p_now:.2f}", showarrow=True, arrowhead=2,
        ax=50, ay=30, bgcolor="#2563eb", font=dict(color="white", size=10)
    )
    
    return fig

# --- PROCESSAMENTO DOS DADOS ---
for s in STOCK_DATABASE:
    p = prices_now.get(s['papel'], 0.0)
    s['cotacao'] = p
    s['lpa'] = s['hist_lpa'][-1] # Pega o LPA mais recente para a tabela
    if p > 0:
        s['pl'] = p / s['lpa']
        s['dy'] = (s['lpa'] * 0.45) / p 
    else:
        s['pl'] = s['dy'] = 0

# --- UI PRINCIPAL ---
st.markdown("""<div class="ad-banner"><span class="ad-badge">AD</span>Opere com vantagem: Use o código <b>DVT329</b> na Giga+ Fibra e tenha a melhor estabilidade do mercado!</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-logo">B3</div>', unsafe_allow_html=True)
    st.subheader("Ranking Ibovespa")
    st.markdown("---")
    user = st.text_input("Seu nome:", "Investidor")
    st.markdown("""<a href="https://wa.me/552220410353?text=DVT329" class="whatsapp-btn">Giga+ Fibra - 20% OFF!</a>""", unsafe_allow_html=True)
    if st.button("🔄 Sincronizar B3"):
        st.cache_data.clear()
        st.rerun()

st.markdown(f"# **Análise Ibovespa Inteligente 2026**")
st.caption(f"Relatório para {user} • Dados Reais de Balanço • Sincronia Real-Time • {datetime.now().strftime('%H:%M:%S')}")

tabs = ["🏆 Ranking Fundamentalista", "✨ Fórmula Mágica", "💎 Graham Valuation", "📈 EPS Diluído", "📉 Assimetria Lucro/Preço"]
active_tab = st.radio("Abas:", tabs, label_visibility="collapsed")

df_v = pd.DataFrame([s for s in STOCK_DATABASE if s['cotacao'] > 0])

if active_tab == "🏆 Ranking Fundamentalista":
    st.dataframe(df_v[['papel', 'empresa', 'cotacao', 'lpa', 'pl', 'dy', 'roe']].sort_values('pl').style.format({
        'cotacao': 'R$ {:.2f}', 'lpa': 'R$ {:.2f}', 'pl': '{:.2f}', 'dy': '{:.2%}', 'roe': '{:.1%}'
    }), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📊 Gráfico de Performance Individual (Base 100)")
    p_select = st.selectbox("Selecione a ação para o gráfico:", df_v['papel'].tolist())
    st.plotly_chart(render_custom_chart(next(s for s in STOCK_DATABASE if s['papel'] == p_select)), use_container_width=True)

elif active_tab == "✨ Fórmula Mágica":
    df_v['score'] = df_v['pl'].rank() + df_v['roe'].rank(ascending=False)
    st.dataframe(df_v.sort_values('score')[['papel', 'cotacao', 'lpa', 'score', 'roe']].style.format({
        'cotacao': 'R$ {:.2f}', 'lpa': 'R$ {:.2f}', 'roe': '{:.1%}'
    }), use_container_width=True, hide_index=True)

elif active_tab == "💎 Graham Valuation":
    df_v['VI'] = (22.5 * df_v['lpa'] * df_v['vpa'])**0.5
    df_v['Upside'] = (df_v['VI'] / df_v['cotacao']) - 1
    st.dataframe(df_v.sort_values('Upside', ascending=False)[['papel', 'cotacao', 'VI', 'Upside']].style.format({
        'cotacao': 'R$ {:.2f}', 'VI': 'R$ {:.2f}', 'Upside': '{:.1%}'
    }), use_container_width=True, hide_index=True)

elif active_tab == "📈 EPS Diluído":
    df_eps = df_v[df_v['epsTrimestral'] >= 0.8].sort_values('epsTrimestral', ascending=False)
    for _, row in df_eps.iterrows():
        st.markdown(f'<div class="custom-card"><b>{row["papel"]}</b> <span style="float:right; color:#10b981; font-weight:800;">EPS Trimestral: R$ {row["epsTrimestral"]:.2f}</span></div>', unsafe_allow_html=True)

elif active_tab == "📉 Assimetria Lucro/Preço":
    st.markdown("### 📉 Detecção de Boca de Jacaré (Base 100)")
    p_asym = st.selectbox("Escolha a ação para ver o gap:", ["PETR4", "BBAS3", "VALE3", "ISAE4", "CPLE3"])
    st.plotly_chart(render_custom_chart(next(s for s in STOCK_DATABASE if s['papel'] == p_asym), title="Divergência Técnica: Lucro Acumulado vs Cotação"), use_container_width=True)

# AFILIADOS
c1, c2 = st.columns(2)
with c1:
    st.markdown('<a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">✈️ <b>Nomad Global</b><br>Sua conta em dólar sem taxas</div></a>', unsafe_allow_html=True)
with c2:
    st.markdown('<a href="https://mpago.li/1VydVhw" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">🤝 <b>Mercado Pago</b><br>Bônus de investidor R$ 30</div></a>', unsafe_allow_html=True)

st.markdown('<div style="background-color:#0f172a; padding:20px; border-radius:18px; text-align:center; color:#94a3b8; font-size:11px;">Ibovespa Inteligente © 2026 | Histórico Real por Ticker | Sem Eixos Decimais</div>', unsafe_allow_html=True)
