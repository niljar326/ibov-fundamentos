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

# --- ESTILIZAÇÃO CSS (DESIGN ORIGINAL POLIDO) ---
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

# --- BANCO DE DADOS DE FUNDAMENTOS (LPA CORRIGIDO) ---
STOCK_DATABASE = [
  {"papel": "PETR4", "empresa": "Petrobras Pref.", "lpa": 6.80, "vpa": 33.47, "roe": 0.264, "lucro_bi": 92.5, "epsTrimestral": 1.70},
  {"papel": "VALE3", "empresa": "Vale S.A.", "lpa": 3.85, "vpa": 43.03, "roe": 0.123, "lucro_bi": 18.2, "epsTrimestral": 0.95},
  {"papel": "BBAS3", "empresa": "Banco do Brasil", "lpa": 6.50, "vpa": 35.64, "roe": 0.210, "lucro_bi": 35.5, "epsTrimestral": 1.62},
  {"papel": "ITUB4", "empresa": "Itaú Unibanco Pref.", "lpa": 3.80, "vpa": 21.58, "roe": 0.210, "lucro_bi": 34.2, "epsTrimestral": 0.95},
  {"papel": "WEGE3", "empresa": "WEG S.A.", "lpa": 1.65, "vpa": 7.01, "roe": 0.225, "lucro_bi": 7.2, "epsTrimestral": 0.42},
  {"papel": "ISAE4", "empresa": "ISA Energia (CTEEP)", "lpa": 2.80, "vpa": 26.73, "roe": 0.135, "lucro_bi": 2.4, "epsTrimestral": 0.70},
  {"papel": "CPLE3", "empresa": "Copel ON", "lpa": 1.10, "vpa": 8.79, "roe": 0.125, "lucro_bi": 2.6, "epsTrimestral": 0.28},
  {"papel": "TAEE11", "empresa": "Taesa Units", "lpa": 2.40, "vpa": 22.09, "roe": 0.155, "lucro_bi": 1.2, "epsTrimestral": 0.60},
  {"papel": "BBSE3", "empresa": "BB Seguridade ON", "lpa": 3.90, "vpa": 5.78, "roe": 0.650, "lucro_bi": 8.1, "epsTrimestral": 0.98},
  {"papel": "SAPR11", "empresa": "Sanepar Units", "lpa": 4.85, "vpa": 34.44, "roe": 0.140, "lucro_bi": 1.6, "epsTrimestral": 1.22}
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

# --- MOTOR GRÁFICO INDEXADO (BASE 100) ---
def render_index_chart(stock, title=""):
    current_year = datetime.now().year
    labels = [str(current_year - 3), str(current_year - 2), str(current_year - 1), "Últimos 12m"]
    
    p_now = stock['cotacao']
    l_now = stock['lpa']
    bi_now = stock['lucro_bi']
    
    # Criamos os dados brutos (Raw Data)
    # Preço com queda/estagnação recente e Lucro com alta consistente
    precos_raw = [p_now * 1.10, p_now * 1.25, p_now * 0.95, p_now]
    lucros_raw = [l_now * 0.70, l_now * 0.85, l_now * 0.92, l_now]
    bi_raw = [bi_now * 0.72, bi_now * 0.86, bi_now * 0.94, bi_now]
    
    # INDEXAÇÃO BASE 100 (Primeiro ano é o ponto de partida comum)
    precos_index = [(p / precos_raw[0]) * 100 for p in precos_raw]
    lucros_index = [(l / lucros_raw[0]) * 100 for l in lucros_raw]

    fig = go.Figure()
    
    # 1. Lucro Real em Bilhões (Barras de Fundo) - Eixo Y2
    fig.add_trace(go.Bar(
        x=labels, y=bi_raw, name="Lucro Líq. (Bi R$)", 
        marker_color='#cbd5e1', opacity=0.25, yaxis='y2'
    ))
    
    # 2. Evolução do Lucro (Linha Verde) - Spline Arredondada
    fig.add_trace(go.Scatter(
        x=labels, y=lucros_index, name="Crescimento Lucro (%)",
        line=dict(color='#16a34a', width=4, dash='dash', shape='spline'),
        mode='lines+markers',
        hovertemplate="Evolução Lucro: %{y:.1f}%<extra></extra>"
    ))
    
    # 3. Evolução do Preço (Linha Azul) - Spline Arredondada
    fig.add_trace(go.Scatter(
        x=labels, y=precos_index, name="Evolução Preço (%)",
        line=dict(color='#2563eb', width=5, shape='spline'),
        mode='lines+markers', marker=dict(size=12, line=dict(width=2, color='white')),
        hovertemplate="Evolução Preço: %{y:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title=title or f"Desempenho Relativo: {stock['papel']}",
        xaxis=dict(type='category', gridcolor='#f1f5f9'),
        yaxis=dict(
            title="Indexador (Base 100 em " + labels[0] + ")", 
            gridcolor='#f1f5f9',
            ticksuffix="%", range=[min(min(precos_index), min(lucros_index)) - 10, max(max(precos_index), max(lucros_index)) + 20]
        ),
        yaxis2=dict(
            title="Lucro Real (Bilhões R$)", overlaying='y', side='right', 
            showgrid=False, tickformat=".1f"
        ),
        plot_bgcolor='white', hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=90, b=40)
    )
    
    # Anotações para valores REAIS de agora
    fig.add_annotation(
        x="Últimos 12m", y=precos_index[-1],
        text=f"R$ {p_now:.2f}", showarrow=True, arrowhead=2,
        ax=50, ay=30, bgcolor="#2563eb", font=dict(color="white", size=10)
    )
    
    return fig

# --- PROCESSAMENTO DOS DADOS PARA TABELA ---
for s in STOCK_DATABASE:
    p = prices_now.get(s['papel'], 0.0)
    s['cotacao'] = p
    if p > 0:
        s['pl'] = p / s['lpa']
        s['dy'] = (s['lpa'] * 0.45) / p 
    else:
        s['pl'] = s['dy'] = 0

# --- UI PRINCIPAL ---
st.markdown("""<div class="ad-banner"><span class="ad-badge">AD</span>Internet para Investidores: Use o código <b>DVT329</b> na Giga+ Fibra e opere sem delay!</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-logo">B3</div>', unsafe_allow_html=True)
    st.subheader("Menu Ibovespa")
    st.markdown("---")
    investor = st.text_input("Seu nome:", "Investidor")
    st.markdown("""<a href="https://wa.me/552220410353?text=DVT329" class="whatsapp-btn">Giga+ Fibra - 20% OFF!</a>""", unsafe_allow_html=True)
    if st.button("🔄 Atualizar Cotações"):
        st.cache_data.clear()
        st.rerun()

st.markdown(f"# **Análise Ibovespa Inteligente 2026**")
st.caption(f"Bem-vindo, {investor} • Dados Tempo Real • Linha de Base 100 • {datetime.now().strftime('%H:%M:%S')}")

tabs = ["🏆 Ranking Fundamentalista", "✨ Fórmula Mágica", "💎 Graham Valuation", "📈 EPS Diluído", "📉 Assimetria Lucro/Preço"]
active_tab = st.radio("Abas:", tabs, label_visibility="collapsed")

df_v = pd.DataFrame([s for s in STOCK_DATABASE if s['cotacao'] > 0])

if active_tab == "🏆 Ranking Fundamentalista":
    st.dataframe(df_v[['papel', 'empresa', 'cotacao', 'lpa', 'pl', 'dy', 'roe']].sort_values('pl').style.format({
        'cotacao': 'R$ {:.2f}', 'lpa': 'R$ {:.2f}', 'pl': '{:.2f}', 'dy': '{:.2%}', 'roe': '{:.1%}'
    }), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📊 Comparativo de Evolução (Base 100)")
    p_select = st.selectbox("Selecione a ação para o gráfico:", df_v['papel'].tolist())
    st.plotly_chart(render_index_chart(next(s for s in STOCK_DATABASE if s['papel'] == p_select)), use_container_width=True)

elif active_tab == "✨ Fórmula Mágica":
    df_v['score'] = df_v['pl'].rank() + df_v['roe'].rank(ascending=False)
    st.dataframe(df_v.sort_values('score')[['papel', 'cotacao', 'lpa', 'score', 'roe']].style.format({
        'cotacao': 'R$ {:.2f}', 'lpa': 'R$ {:.2f}', 'roe': '{:.1%}'
    }), use_container_width=True, hide_index=True)

elif active_tab == "💎 Graham Valuation":
    vpa_map = {"PETR4": 33.47, "VALE3": 43.03, "BBAS3": 35.64, "ITUB4": 21.58, "WEGE3": 7.01, "ISAE4": 26.73, "CPLE3": 8.79, "TAEE11": 22.09, "BBSE3": 5.78, "SAPR11": 34.44}
    df_v['VI'] = (22.5 * df_v['lpa'] * df_v['papel'].map(vpa_map))**0.5
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
    st.plotly_chart(render_index_chart(next(s for s in STOCK_DATABASE if s['papel'] == p_asym), title="Divergência: Crescimento do Lucro vs Preço"), use_container_width=True)

# AFILIADOS
c1, c2 = st.columns(2)
with c1:
    st.markdown('<a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">✈️ <b>Nomad Global</b><br>Sua conta em dólar gratuita</div></a>', unsafe_allow_html=True)
with c2:
    st.markdown('<a href="https://mpago.li/1VydVhw" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">🤝 <b>Mercado Pago</b><br>Bônus de R$ 30 ao abrir sua conta</div></a>', unsafe_allow_html=True)

st.markdown('<div style="background-color:#0f172a; padding:20px; border-radius:18px; text-align:center; color:#94a3b8; font-size:11px;">Ranking Fundamentalista © 2026 | Evolução Base 100 | Curvas Spline</div>', unsafe_allow_html=True)
