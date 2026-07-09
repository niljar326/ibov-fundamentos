import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Ranking B3 Real-Time 2026",
    page_icon="🇧🇷",
    layout="wide"
)

# --- ESTILIZAÇÃO CSS (Design Profissional) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    .custom-card { background-color: white; border: 1px solid #e2e8f0; border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .metric-value { font-size: 24px; font-weight: 800; color: #1e293b; }
    .metric-label { font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600; }
    .stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# --- BANCO DE DADOS DE FUNDAMENTOS (Dados de Balanço 1T26) ---
# LPA (Lucro por Ação) e VPA (Valor Patrimonial por Ação) mudam apenas a cada 3 meses.
# O Dividendos_Ano é a soma dos proventos pagos nos últimos 12 meses.
BASE_DATA = [
    {"papel": "PETR4", "empresa": "Petrobras", "lpa": 9.05, "vpa": 33.47, "div_ano": 4.58, "roic": 0.245},
    {"papel": "VALE3", "empresa": "Vale S.A.", "lpa": 9.17, "vpa": 43.03, "div_ano": 5.45, "roic": 0.187},
    {"papel": "BBAS3", "empresa": "Banco do Brasil", "lpa": 6.78, "vpa": 35.64, "div_ano": 3.12, "roic": 0.198},
    {"papel": "ITUB4", "empresa": "Itaú Unibanco", "lpa": 4.01, "vpa": 21.58, "div_ano": 1.45, "roic": 0.183},
    {"papel": "WEGE3", "empresa": "WEG S.A.", "lpa": 1.57, "vpa": 7.01, "div_ano": 0.72, "roic": 0.212},
    {"papel": "ITSA4", "empresa": "Itaúsa", "lpa": 1.45, "vpa": 8.77, "div_ano": 0.88, "roic": 0.165},
    {"papel": "TAEE11", "empresa": "Taesa", "lpa": 3.65, "vpa": 22.09, "div_ano": 3.42, "roic": 0.142},
    {"papel": "TRPL4", "empresa": "ISA CTEEP", "lpa": 3.79, "vpa": 26.73, "div_ano": 2.55, "roic": 0.138},
    {"papel": "SAPR11", "empresa": "Sanepar", "lpa": 4.81, "vpa": 34.44, "div_ano": 1.95, "roic": 0.128},
    {"papel": "CPLE6", "empresa": "Copel", "lpa": 1.08, "vpa": 8.79, "div_ano": 0.65, "roic": 0.115},
    {"papel": "BBSE3", "empresa": "BB Seguridade", "lpa": 3.84, "vpa": 5.78, "div_ano": 3.25, "roic": 0.584},
    {"papel": "BBDC4", "empresa": "Bradesco", "lpa": 1.58, "vpa": 15.76, "div_ano": 0.95, "roic": 0.095},
    {"papel": "ABEV3", "empresa": "Ambev", "lpa": 0.92, "vpa": 5.50, "div_ano": 0.76, "roic": 0.158},
    {"papel": "EGIE3", "empresa": "Engie Brasil", "lpa": 4.25, "vpa": 14.50, "div_ano": 3.80, "roic": 0.185},
    {"papel": "B3SA3", "empresa": "B3 S.A.", "lpa": 0.85, "vpa": 3.20, "div_ano": 0.45, "roic": 0.170},
]

# --- FUNÇÃO PARA BUSCAR PREÇOS EM TEMPO REAL ---
@st.cache_data(ttl=600) # Atualiza o cache a cada 10 minutos
def get_live_prices(tickers):
    tickers_sa = [f"{t}.SA" for t in tickers]
    data = yf.download(tickers_sa, period="1d", interval="1m", progress=False)['Close']
    
    prices = {}
    for t in tickers:
        # Pega o último preço válido
        try:
            val = data[f"{t}.SA"].dropna().iloc[-1]
            prices[t] = val
        except:
            prices[t] = 0.0
    return prices

# --- PROCESSAMENTO DOS DADOS ---
tickers = [x['papel'] for x in BASE_DATA]
live_prices = get_live_prices(tickers)

df = pd.DataFrame(BASE_DATA)
df['preco_atual'] = df['papel'].map(live_prices)

# Cálculos Automáticos de Indicadores
df['PL'] = df['preco_atual'] / df['lpa']
df['PVP'] = df['preco_atual'] / df['vpa']
df['DY'] = (df['div_ano'] / df['preco_atual']) * 100
df['Graham'] = (22.5 * df['lpa'] * df['vpa'])**0.5
df['Margem_Graham'] = (df['Graham'] / df['preco_atual'] - 1) * 100

# --- UI - DASHBOARD ---
st.title("🇧🇷 Ranking Ibovespa Inteligente")
st.caption(f"Dados de mercado atualizados automaticamente em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# Top Cards (Destaques de agora)
col1, col2, col3, col4 = st.columns(4)
top_dy = df.sort_values('DY', ascending=False).iloc[0]
top_graham = df.sort_values('Margem_Graham', ascending=False).iloc[0]

with col1:
    st.markdown(f"""<div class="custom-card"><div class="metric-label">Maior Dividend Yield</div>
    <div class="metric-value">{top_dy['papel']}</div><div style="color:#10b981; font-weight:bold;">{top_dy['DY']:.2f}%</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="custom-card"><div class="metric-label">Maior Upside Graham</div>
    <div class="metric-value">{top_graham['papel']}</div><div style="color:#2563eb; font-weight:bold;">+{top_graham['Margem_Graham']:.1f}%</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="custom-card"><div class="metric-label">Preço Médio PETR4</div>
    <div class="metric-value">R$ {live_prices.get('PETR4', 0):.2f}</div><div style="color:#64748b;">Tempo Real</div></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="custom-card"><div class="metric-label">Ações Monitoradas</div>
    <div class="metric-value">{len(df)}</div><div style="color:#64748b;">Ibovespa + Selecionadas</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# Abas de Estratégias
tab1, tab2, tab3 = st.tabs(["💎 Preço Justo (Graham)", "✨ Fórmula Mágica (Greenblatt)", "💰 Dividend Yield"])

with tab1:
    st.subheader("Oportunidades por Valor Intrínseco")
    df_g = df.copy()
    df_g = df_g[['papel', 'empresa', 'preco_atual', 'Graham', 'Margem_Graham']]
    df_g = df_g.sort_values('Margem_Graham', ascending=False)
    
    st.dataframe(df_g.style.format({
        'preco_atual': 'R$ {:.2f}',
        'Graham': 'R$ {:.2f}',
        'Margem_Graham': '{:.1f}%'
    }), use_container_width=True)

with tab2:
    st.subheader("Ranking de Eficiência (Menor P/L + Maior ROIC)")
    df_magic = df.copy()
    df_magic['rank_pl'] = df_magic['PL'].rank(ascending=True)
    df_magic['rank_roic'] = df_magic['roic'].rank(ascending=False)
    df_magic['score'] = df_magic['rank_pl'] + df_magic['rank_roic']
    df_magic = df_magic.sort_values('score')
    
    st.dataframe(df_magic[['papel', 'empresa', 'preco_atual', 'PL', 'roic']].style.format({
        'preco_atual': 'R$ {:.2f}',
        'PL': '{:.2f}',
        'roic': '{:.1%}'
    }), use_container_width=True)

with tab3:
    st.subheader("Maiores Pagadoras de Dividendos (Real-Time)")
    df_dy = df.sort_values('DY', ascending=False)
    st.dataframe(df_dy[['papel', 'empresa', 'preco_atual', 'div_ano', 'DY']].style.format({
        'preco_atual': 'R$ {:.2f}',
        'div_ano': 'R$ {:.2f}',
        'DY': '{:.2f}%'
    }), use_container_width=True)

# Gráfico de Assimetria
st.markdown("### 📊 Gráfico de Cotação Atual vs. Valor de Graham")
fig = go.Figure()
fig.add_trace(go.Bar(x=df['papel'], y=df['preco_atual'], name="Preço de Mercado", marker_color='#2563eb'))
fig.add_trace(go.Scatter(x=df['papel'], y=df['Graham'], name="Preço Justo (Graham)", mode='lines+markers', line=dict(color='#10b981', width=3)))

fig.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor='white',
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig, use_container_width=True)

# Rodapé Informativo
st.sidebar.markdown("### Configurações de Dados")
st.sidebar.write("Os dados de balanço (LPA/VPA) são do **1T2026**.")
if st.sidebar.button("Atualizar Preços Agora"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.write("🚀 **Dica:** Ativos com Margem de Graham positiva e DY acima de 6% são considerados zona de oportunidade técnica.")
