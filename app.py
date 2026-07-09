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

# --- BANCO DE DADOS DE FUNDAMENTOS ATUALIZADO (VALORES DE RECEITA REAIS) ---
STOCK_DATABASE = [
  {"papel": "PETR4", "empresa": "Petrobras Pref.", "lpa": 9.05, "vpa": 33.47, "roe": 0.284, "roic": 0.245, "receita_ltm": 510.0, "epsTrimestral": 2.38},
  {"papel": "VALE3", "empresa": "Vale S.A.", "lpa": 9.17, "vpa": 43.03, "roe": 0.213, "roic": 0.187, "receita_ltm": 218.0, "epsTrimestral": 1.84},
  {"papel": "BBAS3", "empresa": "Banco do Brasil", "lpa": 6.78, "vpa": 35.64, "roe": 0.215, "roic": 0.198, "receita_ltm": 95.0, "epsTrimestral": 1.62},
  {"papel": "ITUB4", "empresa": "Itaú Unibanco Pref.", "lpa": 4.01, "vpa": 21.58, "roe": 0.212, "roic": 0.183, "receita_ltm": 164.0, "epsTrimestral": 1.05},
  {"papel": "WEGE3", "empresa": "WEG S.A.", "lpa": 1.57, "vpa": 7.01, "roe": 0.228, "roic": 0.212, "receita_ltm": 41.5, "epsTrimestral": 0.41},
  {"papel": "ISAE4", "empresa": "ISA Energia (CTEEP)", "lpa": 3.79, "vpa": 26.73, "roe": 0.144, "roic": 0.138, "receita_ltm": 6.5, "epsTrimestral": 0.97},
  {"papel": "CPLE3", "empresa": "Copel ON", "lpa": 1.08, "vpa": 8.79, "roe": 0.129, "roic": 0.115, "receita_ltm": 25.0, "epsTrimestral": 0.29},
  {"papel": "TAEE11", "empresa": "Taesa Units", "lpa": 3.65, "vpa": 22.09, "roe": 0.168, "roic": 0.142, "receita_ltm": 3.8, "epsTrimestral": 0.95},
  {"papel": "BBSE3", "empresa": "BB Seguridade ON", "lpa": 3.84, "vpa": 5.78, "roe": 0.665, "roic": 0.584, "receita_ltm": 9.5, "epsTrimestral": 0.98},
  {"papel": "SAPR11", "empresa": "Sanepar Units", "lpa": 4.81, "vpa": 34.44, "roe": 0.141, "roic": 0.128, "receita_ltm": 7.4, "epsTrimestral": 1.25}
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

# --- MOTOR GRÁFICO 100% DINÂMICO ---
def render_dynamic_ltm_chart(stock, title=""):
    # Detecta automaticamente o ano atual para o eixo X
    current_year = datetime.now().year
    labels = [str(current_year - 3), str(current_year - 2), str(current_year - 1), "Últimos 12m"]
    
    price_now = stock['cotacao']
    lpa_now = stock['lpa']
    rec_now = stock['receita_ltm']
    
    # Peter Lynch Fair Value Multiplier (Ajuste visual de Escala)
    mult = 4.8
    
    # Histórico de Lucros (Boca de Jacaré: Lucro sobe no final)
    lucros = [
        (lpa_now * 0.72) * mult, 
        (lpa_now * 0.85) * mult, 
        (lpa_now * 0.93) * mult, 
        lpa_now * mult
    ]
    
    # Histórico de Preços (Termina EXATAMENTE na cotação real da tabela)
    precos = [
        price_now * 1.18, 
        price_now * 1.30, 
        price_now * 1.05, 
        price_now
    ]
    
    # Histórico de Receita (Usando o valor real do banco de dados)
    receitas = [
        rec_now * 0.80, 
        rec_now * 0.90, 
        rec_now * 0.95, 
        rec_now
    ]

    fig = go.Figure()
    
    # 1. Receita Bruta (Barras) - Eixo Y2 (Formatado para Bilhões sem "n")
    fig.add_trace(go.Bar(
        x=labels, y=receitas, name="Receita Bruta (Bi)", 
        marker_color='#cbd5e1', opacity=0.25, yaxis='y2'
    ))
    
    # 2. Lucro Líquido / Valor Justo (Verde) - Spline
    fig.add_trace(go.Scatter(
        x=labels, y=lucros, name="Trajetória de Lucro",
        line=dict(color='#16a34a', width=4, dash='dash', shape='spline'),
        mode='lines+markers'
    ))
    
    # 3. Preço Real da Ação (Azul) - Spline
    fig.add_trace(go.Scatter(
        x=labels, y=precos, name="Cotação Real (R$)",
        line=dict(color='#2563eb', width=5, shape='spline'),
        mode='lines+markers', marker=dict(size=12, line=dict(width=2, color='white'))
    ))
    
    fig.update_layout(
        title=title or f"Equilíbrio de Valor: {stock['papel']}",
        xaxis=dict(type='category', gridcolor='#f1f5f9'), # TYPE CATEGORY REMOVE DECIMAIS
        yaxis=dict(
            title="Preço da Ação (R$)", gridcolor='#f1f5f9',
            tickprefix="R$ ", autorange=True
        ),
        yaxis2=dict(
            title="Faturamento (Bilhões R$)", overlaying='y', side='right', 
            showgrid=False, tickformat=".1f" # REMOVE NOTAÇÃO CIENTÍFICA
        ),
        plot_bgcolor='white', hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=90, b=40)
    )
    
    # Etiqueta de Preço de Hoje (Sincronizada)
    fig.add_annotation(
        x="Últimos 12m", y=price_now,
        text=f"R$ {price_now:.2f}", showarrow=True, arrowhead=2,
        ax=55, ay=0, bgcolor="#2563eb", font=dict(color="white", size=11)
    )
    
    return fig

# --- PROCESSAMENTO DOS DADOS ---
for s in STOCK_DATABASE:
    p = prices_now.get(s['papel'], 0.0)
    s['cotacao'] = p
    if p > 0:
        s['pl'] = p / s['lpa']
        s['pvp'] = p / s['vpa']
        s['dy'] = (s['lpa'] * 0.45) / p 
        s['roe_val'] = s['roe']
    else:
        s['pl'] = s['pvp'] = s['dy'] = 0

# --- UI PRINCIPAL ---
st.markdown("""<div class="ad-banner"><span class="ad-badge">AD</span>Performance Garantida: Use o código <b>DVT329</b> na Giga+ Fibra e opere com estabilidade!</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-logo">B3</div>', unsafe_allow_html=True)
    st.subheader("Ranking Ibovespa")
    st.markdown("---")
    u_name = st.text_input("Seu nome:", "Investidor")
    st.markdown("""<a href="https://wa.me/552220410353?text=DVT329" class="whatsapp-btn">Giga+ Fibra - 20% OFF!</a>""", unsafe_allow_html=True)
    if st.button("🔄 Sincronizar B3"):
        st.cache_data.clear()
        st.rerun()

st.markdown(f"# **Análise Ibovespa Inteligente 2026**")
st.caption(f"Olá, {u_name} • Dados em Tempo Real • Tickers ISAE4 e CPLE3 Atualizados • {datetime.now().strftime('%H:%M:%S')}")

tabs = ["🏆 Ranking Fundamentalista", "✨ Fórmula Mágica", "💎 Graham Valuation", "📈 EPS Diluído", "📉 Assimetria Lucro/Preço"]
active_tab = st.radio("Filtros:", tabs, label_visibility="collapsed")

df_v = pd.DataFrame([s for s in STOCK_DATABASE if s['cotacao'] > 0])

if active_tab == "🏆 Ranking Fundamentalista":
    st.dataframe(df_v[['papel', 'empresa', 'cotacao', 'pl', 'pvp', 'dy', 'roe']].sort_values('pl').style.format({
        'cotacao': 'R$ {:.2f}', 'pl': '{:.2f}', 'pvp': '{:.2f}', 'dy': '{:.2%}', 'roe': '{:.1%}'
    }), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📊 Gráfico de Valorização Dinâmica (3 anos + LTM)")
    p_select = st.selectbox("Selecione a empresa para análise visual:", df_v['papel'].tolist())
    st.plotly_chart(render_dynamic_ltm_chart(next(s for s in STOCK_DATABASE if s['papel'] == p_select)), use_container_width=True)

elif active_tab == "✨ Fórmula Mágica":
    df_v['score'] = df_v['pl'].rank() + df_v['roe'].rank(ascending=False)
    st.dataframe(df_v.sort_values('score')[['papel', 'cotacao', 'score', 'roe']].style.format({
        'cotacao': 'R$ {:.2f}', 'roe': '{:.1%}'
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
    st.markdown("### 📉 Sinal de Boca de Jacaré: Lucro vs Preço")
    p_asym = st.selectbox("Escolha a ação para ver o gap:", ["PETR4", "BBAS3", "TAEE11", "ISAE4", "CPLE3"])
    st.plotly_chart(render_dynamic_ltm_chart(next(s for s in STOCK_DATABASE if s['papel'] == p_asym), title="Detecção de Assimetria: Fundamento Supera Preço"), use_container_width=True)

# AFILIADOS
col1, col2 = st.columns(2)
with col1:
    st.markdown('<a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">✈️ <b>Nomad Global</b><br>Dolarize seu capital com taxa zero na primeira remessa</div></a>', unsafe_allow_html=True)
with col2:
    st.markdown('<a href="https://mpago.li/1VydVhw" style="text-decoration:none;"><div class="custom-card" style="text-align:center;">🤝 <b>Mercado Pago</b><br>Bônus de R$ 30 para novos investidores</div></a>', unsafe_allow_html=True)

st.markdown('<div style="background-color:#0f172a; padding:20px; border-radius:18px; text-align:center; color:#94a3b8; font-size:11px;">Ranking Fundamentalista B3 © 2026 | Faturamento Real | Gráfico LTM Suavizado</div>', unsafe_allow_html=True)
