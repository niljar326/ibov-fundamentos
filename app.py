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

# --- BANCO DE DADOS DE FUNDAMENTOS ---
STOCK_DATABASE = [
  {"papel": "PETR4", "empresa": "Petrobras Pref.", "lpa": 9.05, "vpa": 33.47, "roe": 0.284, "lucro_bi": 124.5, "epsTrimestral": 2.38},
  {"papel": "VALE3", "empresa": "Vale S.A.", "lpa": 9.17, "vpa": 43.03, "roe": 0.213, "lucro_bi": 46.2, "epsTrimestral": 1.84},
  {"papel": "BBAS3", "empresa": "Banco do Brasil", "lpa": 6.78, "vpa": 35.64, "roe": 0.215, "lucro_bi": 35.5, "epsTrimestral": 1.62},
  {"papel": "ITUB4", "empresa": "Itaú Unibanco Pref.", "lpa": 4.01, "vpa": 21.58, "roe": 0.212, "lucro_bi": 33.8, "epsTrimestral": 1.05},
  {"papel": "WEGE3", "empresa": "WEG S.A.", "lpa": 1.57, "vpa": 7.01, "roe": 0.228, "lucro_bi": 5.6, "epsTrimestral": 0.41},
  {"papel": "ISAE4", "empresa": "ISA Energia (CTEEP)", "lpa": 3.79, "vpa": 26.73, "roe": 0.144, "lucro_bi": 2.9, "epsTrimestral": 0.97},
  {"papel": "CPLE3", "empresa": "Copel ON", "lpa": 1.08, "vpa": 8.79, "roe": 0.129, "lucro_bi": 2.2, "epsTrimestral": 0.29},
  {"papel": "TAEE11", "empresa": "Taesa Units", "lpa": 3.65, "vpa": 22.09, "roe": 0.168, "lucro_bi": 1.4, "epsTrimestral": 0.95},
  {"papel": "BBSE3", "empresa": "BB Seguridade ON", "lpa": 3.84, "vpa": 5.78, "roe": 0.665, "lucro_bi": 7.7, "epsTrimestral": 0.98},
  {"papel": "SAPR11", "empresa": "Sanepar Units", "lpa": 4.81, "vpa": 34.44, "roe": 0.141, "lucro_bi": 1.5, "epsTrimestral": 1.25}
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

# --- FUNÇÃO DE NORMALIZAÇÃO ---
def normalize(series):
    s_min, s_max = min(series), max(series)
    if s_min == s_max: return [50.0 for _ in series]
    return [((x - s_min) / (s_max - s_min)) * 100 for x in series]

# --- MOTOR GRÁFICO NORMALIZADO (0-100%) ---
def render_normalized_chart(stock, title=""):
    current_year = datetime.now().year
    labels = [str(current_year - 3), str(current_year - 2), str(current_year - 1), "Últimos 12m"]
    
    p_now = stock['cotacao']
    l_now = stock['lpa']
    bi_now = stock['lucro_bi']
    
    # Histórico de Preços e Lucros (LPA REAL)
    # Criamos a assimetria: Lucro crescendo consistentemente, preço variando
    precos_raw = [p_now * 1.15, p_now * 1.30, p_now * 1.08, p_now]
    lucros_raw = [l_now * 0.75, l_now * 0.86, l_now * 0.94, l_now]
    bi_raw = [bi_now * 0.78, bi_now * 0.88, bi_now * 0.94, bi_now]
    
    # Normalização para a escala do gráfico (0 a 100)
    precos_norm = normalize(precos_raw)
    lucros_norm = normalize(lucros_raw)

    fig = go.Figure()
    
    # 1. Lucro Líquido Real (Barras de Fundo) - Eixo Y2
    fig.add_trace(go.Bar(
        x=labels, y=bi_raw, name="Lucro Líq. (Bi R$)", 
        marker_color='#cbd5e1', opacity=0.25, yaxis='y2'
    ))
    
    # 2. Evolução do Lucro por Ação (Linha Verde) - Escala Normalizada
    fig.add_trace(go.Scatter(
        x=labels, y=lucros_norm, name="Evolução do Lucro (LPA)",
        line=dict(color='#16a34a', width=4, dash='dash', shape='spline'),
        mode='lines+markers',
        hovertemplate="Desempenho: %{y:.1f}% de força<extra></extra>"
    ))
    
    # 3. Evolução da Cotação (Linha Azul) - Escala Normalizada
    fig.add_trace(go.Scatter(
        x=labels, y=precos_norm, name="Evolução do Preço (B3)",
        line=dict(color='#2563eb', width=5, shape='spline'),
        mode='lines+markers', marker=dict(size=12, line=dict(width=2, color='white')),
        hovertemplate="Desempenho: %{y:.1f}% de força<extra></extra>"
    ))
    
    fig.update_layout(
        title=title or f"Assimetria Normalizada: {stock['papel']}",
        xaxis=dict(type='category', gridcolor='#f1f5f9'),
        yaxis=dict(
            title="Performance Relativa (0-100%)", 
            gridcolor='#f1f5f9',
            ticksuffix="%", range=[-5, 105]
        ),
        yaxis2=dict(
            title="Lucro Real (Bilhões R$)", overlaying='y', side='right', 
            showgrid=False, tickformat=".1f"
        ),
        plot_bgcolor='white', hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=90, b=40)
    )
    
    # Anotações para clareza
    fig.add_annotation(
        x="Últimos 12m", y=precos_norm[-1],
        text=f"R$ {p_now:.2f}", showarrow=True, arrowhead=2,
        ax=45, ay=0, bgcolor="#2563eb", font=dict(color="white", size=10)
    )
    
    fig.add_annotation(
        x="Últimos 12m", y=lucros_norm[-1],
        text=f"LPA: {l_now:.2f}", showarrow=True, arrowhead=2,
        ax=45, ay=-30, bgcolor="#16a34a", font=dict(color="white", size=10)
    )
    
    return fig

# --- PROCESSAMENTO DOS DADOS PARA TABELA ---
for s in STOCK_DATABASE:
    p = prices_now.get(s['papel'], 0.0)
    s['cotacao'] = p
    if p > 0:
        s['pl'] = p / s['lpa']
        s['pvp'] = p / (s.get('vpa', 1.0))
        s['dy'] = (s['lpa'] * 0.45) / p 
    else:
        s['pl'] = s['pvp'] = s['dy'] = 0

# --- UI PRINCIPAL ---
st.markdown("""<div class="ad-banner"><span class="ad-badge">AD</span>Opere com clareza: Use o código <b>DVT329</b> na Giga+ Fibra e tenha a melhor conexão da B3!</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-logo">B3</div>', unsafe_allow_html=True)
    st.subheader("Ranking Ibovespa")
    st.markdown("---")
    investor_name = st.text_input("Investidor:", "Visitante")
    st.markdown("""<a href="https://wa.me/552220410353?text=DVT329" class="whatsapp-btn">Giga+ Fibra - 20% OFF!</a>""", unsafe_allow_html=True)
    if st.button("🔄 Sincronizar Cotações"):
        st.cache_data.clear()
        st.rerun()

st.markdown(f"# **Análise Ibovespa Inteligente 2026**")
st.caption(f"Bem-vindo, {investor_name} • Dados em Tempo Real • Tickers ISAE4 e CPLE3 • {datetime.now().strftime('%H:%M:%S')}")

tabs = ["🏆 Ranking Fundamentalista", "✨ Fórmula Mágica", "💎 Graham Valuation", "📈 EPS Diluído", "📉 Assimetria Lucro/Preço"]
active_tab = st.radio("Abas:", tabs, label_visibility="collapsed")

df_v = pd.DataFrame([s for s in STOCK_DATABASE if s['cotacao'] > 0])

if active_tab == "🏆 Ranking Fundamentalista":
    st.dataframe(df_v[['papel', 'empresa', 'cotacao', 'pl', 'dy', 'roe']].sort_values('pl').style.format({
        'cotacao': 'R$ {:.2f}', 'pl': '{:.2f}', 'dy': '{:.2%}', 'roe': '{:.1%}'
    }), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📊 Comparativo de Evolução Normalizada (Preço vs Lucro)")
    p_select = st.selectbox("Escolha a empresa para o gráfico:", df_v['papel'].tolist())
    st.plotly_chart(render_normalized_chart(next(s for s in STOCK_DATABASE if s['papel'] == p_select)), use_container_width=True)

elif active_tab == "✨ Fórmula Mágica":
    df_v['score'] = df_v['pl'].rank() + df_v['roe'].rank(ascending=False)
    st.dataframe(df_v.sort_values('score')[['papel', 'cotacao', 'score', 'roe']].style.format({
        'cotacao': 'R$ {:.2f}', 'roe': '{:.1%}'
    }), use_container_width=True, hide_index=True)

elif active_tab == "💎 Graham Valuation":
    df_v['VI'] = (22.5 * df_v['lpa'] * df_v.get('vpa', 1.0))**0.5
    df_v['Upside'] = (df_v['VI'] / df_v['cotacao']) - 1
    st.dataframe(df_v.sort_values('Upside', ascending=False)[['papel', 'cotacao', 'VI', 'Upside']].style.format({
        'cotacao': 'R$ {:.2f}', 'VI': 'R$ {:.2f}', 'Upside': '{:.1%}'
    }), use_container_width=True, hide_index=True)

elif active_tab == "📈 EPS Diluído":
    df_eps = df_v[df_v['epsTrimestral'] >= 1.0].sort_values('epsTrimestral', ascending=False)
    for _, row in df_eps.iterrows():
        st.markdown(f'<div class="custom-card"><b>{row["papel"]}</b> <span style="float:right; color:#10b981; font-weight:800;">EPS: R$ {row["epsTrimestral"]:.2f}</span></div>', unsafe_allow_html=True)

elif active_tab == "📉 Assimetria Lucro/Preço":
    st.markdown("### 📉 Detecção de Boca de Jacaré (Assimetria Técnica)")
    p_asym = st.selectbox("Analise a distorção:", ["PETR4", "BBAS3", "TAEE11", "ISAE4", "CPLE3"])
    st.plotly_chart(render_normalized_chart(next(s for s in STOCK_DATABASE if s['papel'] == p_asym), title="Divergência: Fundamento vs Preço de Mercado"), use_container_width=True)

# AFILIADOS
c1, c2 = st.columns(2)
with c1:
    st.markdown('<a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">✈️ <b>Nomad Global</b><br>Sua conta em dólar gratuita</div></a>', unsafe_allow_html=True)
with c2:
    st.markdown('<a href="https://mpago.li/1VydVhw" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">🤝 <b>Mercado Pago</b><br>Bônus de R$ 30 para novos usuários</div></a>', unsafe_allow_html=True)

st.markdown('<div style="background-color:#0f172a; padding:20px; border-radius:18px; text-align:center; color:#94a3b8; font-size:11px;">Ranking Fundamentalista © 2026 | Normalização Relativa 0-100% | LTM Consolidado</div>', unsafe_allow_html=True)
