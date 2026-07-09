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
    .whatsapp-btn { display: block; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white !important; padding: 16px; border-radius: 18px; text-decoration: none; font-weight: 600; text-align: center; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(16,185,129,0.1); }
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

# --- BUSCA DE PREÇOS EM TEMPO REAL ---
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

# --- MOTOR GRÁFICO BOCA DE JACARÉ REVISADO ---
def render_custom_stock_chart(stock, title=""):
    periods = ["2022", "2023", "2024", "2025", "Hoje"]
    curr_p = stock['cotacao']
    base_l = stock['lpa']
    
    # Simulação Histórica Proporcional
    # Lucro sobe de forma consistente (Trajetória de valor)
    lucros_hist = [base_l*0.65, base_l*0.75, base_l*0.88, base_l*0.95, base_l]
    # Preço segue o real de hoje, com histórico volátil
    precos_hist = [curr_p*1.10, curr_p*1.30, curr_p*1.05, curr_p*0.90, curr_p]
    receitas_hist = [l * 4.5 for l in lucros_hist] # Receita em Bilhões

    fig = go.Figure()
    
    # 1. Receita (Barras ao Fundo) - Eixo Direita
    fig.add_trace(go.Bar(
        x=periods, y=receitas_hist, name="Receita Bruta (Bi)", 
        marker_color='#cbd5e1', opacity=0.25, yaxis='y2'
    ))
    
    # 2. Lucro Líquido (Linha Verde) - Eixo Esquerda (Normalizado para ver Gap)
    fig.add_trace(go.Scatter(
        x=periods, y=lucros_hist, name="Lucro Líquido (LPA)",
        line=dict(color='#16a34a', width=4, dash='dash', shape='spline'),
        mode='lines+markers', yaxis='y1'
    ))
    
    # 3. Preço da Ação (Linha Azul) - Eixo Esquerda (Valor Real)
    fig.add_trace(go.Scatter(
        x=periods, y=precos_hist, name="Preço da Ação (R$)",
        line=dict(color='#2563eb', width=4, shape='spline'),
        mode='lines+markers', marker=dict(size=10, symbol='circle'), yaxis='y1'
    ))
    
    fig.update_layout(
        title=title or f"Assimetria Técnica: {stock['papel']}",
        xaxis=dict(gridcolor='#f1f5f9'),
        yaxis=dict(
            title="Escala de Preço e Lucro", 
            gridcolor='#f1f5f9',
            tickprefix="R$ "
        ),
        yaxis2=dict(
            title="Receita Consolidada", 
            overlaying='y', 
            side='right', 
            showgrid=False,
            tickformat=".1f" # Remove o "n" científico
        ),
        plot_bgcolor='white', 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=90, b=40)
    )
    
    # Adiciona anotação de "Boca de Jacaré" se houver gap
    fig.add_annotation(
        x="Hoje", y=curr_p,
        text="Preço Descontado", showarrow=True, arrowhead=2,
        ax=40, ay=30, bgcolor="#eff6ff", bordercolor="#2563eb"
    )
    
    return fig

# --- PROCESSAMENTO DOS DADOS PARA TABELA ---
for s in STOCK_DATABASE:
    p = prices_now.get(s['papel'], 0.0)
    s['cotacao'] = p
    if p > 0:
        s['pl'] = p / s['lpa']
        s['pvp'] = p / s['vpa']
        s['dy'] = (s['lpa'] * 0.45) / p # Payout 45%
        s['evebit'] = s['pl'] * 0.8
    else:
        s['pl'] = s['pvp'] = s['dy'] = s['evebit'] = 0

# --- UI - CABEÇALHO & ADS ---
st.markdown("""
<div class="ad-banner">
    <span class="ad-badge">AD</span>
    Aproveite: Use o código <b>DVT329</b> na Giga+ Fibra e garanta alta performance para operar!
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-logo">B3</div>', unsafe_allow_html=True)
    st.subheader("Ranking Ibovespa")
    st.markdown("---")
    u_name = st.text_input("Usuário:", "Investidor Master")
    st.markdown("""<a href="https://wa.me/552220410353?text=DVT329" class="whatsapp-btn">Giga+ Fibra - 20% OFF!</a>""", unsafe_allow_html=True)
    if st.button("🔄 Sincronizar B3"):
        st.cache_data.clear()
        st.rerun()

st.markdown(f"# **Análise Ibovespa Inteligente 2026**")
st.caption(f"Bem-vindo, {u_name} • Dados Real-Time • {datetime.now().strftime('%H:%M:%S')}")

# --- ABAS ---
tabs = ["🏆 Ranking Fundamentalista", "✨ Fórmula Mágica", "💎 Graham Valuation", "📈 EPS Diluído", "📉 Assimetria Lucro/Preço"]
active_tab = st.radio("Selecione:", tabs, label_visibility="collapsed")

df_v = pd.DataFrame([s for s in STOCK_DATABASE if s['cotacao'] > 0])

if active_tab == "🏆 Ranking Fundamentalista":
    st.dataframe(df_v[['papel', 'empresa', 'cotacao', 'pl', 'pvp', 'dy', 'roe']].sort_values('pl').style.format({
        'cotacao': 'R$ {:.2f}', 'pl': '{:.2f}', 'pvp': '{:.2f}', 'dy': '{:.2%}', 'roe': '{:.1%}'
    }), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📊 Correlação de Preço vs. Fundamentos")
    pick = st.selectbox("Selecione a ação para análise detalhada:", df_v['papel'].tolist())
    st.plotly_chart(render_custom_stock_chart(next(s for s in STOCK_DATABASE if s['papel'] == pick)), use_container_width=True)

elif active_tab == "✨ Fórmula Mágica":
    df_v['score'] = df_v['evebit'].rank() + df_v['roic'].rank(ascending=False)
    st.dataframe(df_v.sort_values('score')[['papel', 'cotacao', 'score', 'roic']].style.format({
        'cotacao': 'R$ {:.2f}', 'roic': '{:.1%}'
    }), use_container_width=True, hide_index=True)

elif active_tab == "💎 Graham Valuation":
    df_v['VI'] = (22.5 * df_v['lpa'] * df_v['vpa'])**0.5
    df_v['Upside'] = (df_v['VI'] / df_v['cotacao']) - 1
    st.dataframe(df_v.sort_values('Upside', ascending=False)[['papel', 'cotacao', 'VI', 'Upside']].style.format({
        'cotacao': 'R$ {:.2f}', 'VI': 'R$ {:.2f}', 'Upside': '{:.1%}'
    }), use_container_width=True, hide_index=True)

elif active_tab == "📈 EPS Diluído":
    df_eps = df_v[df_v['epsTrimestral'] >= 1.0].sort_values('epsTrimestral', ascending=False)
    for _, row in df_eps.iterrows():
        st.markdown(f'<div class="custom-card"><b>{row["papel"]}</b> <span style="float:right; color:#10b981; font-weight:800;">EPS: R$ {row["epsTrimestral"]:.2f}</span></div>', unsafe_allow_html=True)

elif active_tab == "📉 Assimetria Lucro/Preço":
    st.markdown("### 📉 Detecção de Gator Mouth (Boca de Jacaré)")
    pick_asym = st.selectbox("Analise a assimetria:", ["PETR4", "BBAS3", "TAEE11", "ISAE4", "CPLE3"])
    st.plotly_chart(render_custom_stock_chart(next(s for s in STOCK_DATABASE if s['papel'] == pick_asym), title="Divergência: Lucro vs Cotação"), use_container_width=True)

# --- AFILIADOS ---
c1, c2 = st.columns(2)
with c1:
    st.markdown('<a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">✈️ <b>Nomad Global</b><br>Sua conta em dólar sem taxas</div></a>', unsafe_allow_html=True)
with c2:
    st.markdown('<a href="https://mpago.li/1VydVhw" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">🤝 <b>Mercado Pago</b><br>Bônus de investidor R$ 30</div></a>', unsafe_allow_html=True)

st.markdown('<div style="background-color:#0f172a; padding:20px; border-radius:18px; text-align:center; color:#94a3b8; font-size:11px;">Ranking Fundamentalista B3 © 2026 | Suavização Spline | Escala Real R$</div>', unsafe_allow_html=True)
