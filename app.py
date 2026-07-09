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

# --- ESTILIZAÇÃO CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    .custom-card { background-color: white; border: 1px solid #e2e8f0; border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .metric-value { font-size: 24px; font-weight: 800; color: #1e293b; }
    .metric-label { font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- BANCO DE DADOS DE FUNDAMENTOS (Atualizado com ISAE4) ---
BASE_DATA = [
    {"papel": "PETR4", "empresa": "Petrobras", "lpa": 9.05, "vpa": 33.47, "div_ano": 4.58, "roic": 0.245},
    {"papel": "VALE3", "empresa": "Vale S.A.", "lpa": 9.17, "vpa": 43.03, "div_ano": 5.45, "roic": 0.187},
    {"papel": "BBAS3", "empresa": "Banco do Brasil", "lpa": 6.78, "vpa": 35.64, "div_ano": 3.12, "roic": 0.198},
    {"papel": "ITUB4", "empresa": "Itaú Unibanco", "lpa": 4.01, "vpa": 21.58, "div_ano": 1.45, "roic": 0.183},
    {"papel": "WEGE3", "empresa": "WEG S.A.", "lpa": 1.57, "vpa": 7.01, "div_ano": 0.72, "roic": 0.212},
    {"papel": "ITSA4", "empresa": "Itaúsa", "lpa": 1.45, "vpa": 8.77, "div_ano": 0.88, "roic": 0.165},
    {"papel": "TAEE11", "empresa": "Taesa", "lpa": 3.65, "vpa": 22.09, "div_ano": 3.42, "roic": 0.142},
    {"papel": "ISAE4", "empresa": "ISA Energia Brasil", "lpa": 3.79, "vpa": 26.73, "div_ano": 2.55, "roic": 0.138}, # TICKER ATUALIZADO
    {"papel": "SAPR11", "empresa": "Sanepar", "lpa": 4.81, "vpa": 34.44, "div_ano": 1.95, "roic": 0.128},
    {"papel": "CPLE6", "empresa": "Copel", "lpa": 1.08, "vpa": 8.79, "div_ano": 0.65, "roic": 0.115},
    {"papel": "BBSE3", "empresa": "BB Seguridade", "lpa": 3.84, "vpa": 5.78, "div_ano": 3.25, "roic": 0.584},
    {"papel": "BBDC4", "empresa": "Bradesco", "lpa": 1.58, "vpa": 15.76, "div_ano": 0.95, "roic": 0.095},
    {"papel": "ABEV3", "empresa": "Ambev", "lpa": 0.92, "vpa": 5.50, "div_ano": 0.76, "roic": 0.158},
    {"papel": "EGIE3", "empresa": "Engie Brasil", "lpa": 4.25, "vpa": 14.50, "div_ano": 3.80, "roic": 0.185},
    {"papel": "B3SA3", "empresa": "B3 S.A.", "lpa": 0.85, "vpa": 3.20, "div_ano": 0.45, "roic": 0.170},
]

# --- FUNÇÃO PARA BUSCAR PREÇOS EM TEMPO REAL ---
@st.cache_data(ttl=300) # Atualiza a cada 5 minutos
def get_live_prices(tickers):
    # Criar lista formatada para o Yahoo Finance (ex: ISAE4.SA)
    tickers_sa = [f"{t}.SA" for t in tickers]
    
    # Download dos dados
    try:
        data = yf.download(tickers_sa, period="1d", interval="1m", progress=False)
        # Se for um DataFrame com multi-index (vários tickers), pegamos o 'Close'
        close_data = data['Close']
        
        prices = {}
        for t in tickers:
            try:
                # Pega o último preço não nulo
                val = close_data[f"{t}.SA"].dropna().iloc[-1]
                prices[t] = float(val)
            except:
                prices[t] = 0.0
        return prices
    except Exception as e:
        st.error(f"Erro ao buscar cotações: {e}")
        return {t: 0.0 for t in tickers}

# --- PROCESSAMENTO ---
tickers = [x['papel'] for x in BASE_DATA]
live_prices = get_live_prices(tickers)

df = pd.DataFrame(BASE_DATA)
df['preco_atual'] = df['papel'].map(live_prices)

# Cálculo dos indicadores dinâmicos
df['PL'] = df['preco_atual'] / df['lpa']
df['PVP'] = df['preco_atual'] / df['vpa']
df['DY'] = (df['div_ano'] / df['preco_atual']) * 100
df['Graham'] = (22.5 * df['lpa'] * df['vpa'])**0.5
df['Margem_Graham'] = (df['Graham'] / df['preco_atual'] - 1) * 100

# --- UI DASHBOARD ---
st.title("🇧🇷 Ranking Ibovespa Real-Time")
st.caption(f"Cotações automáticas via Yahoo Finance • Atualizado em: {datetime.now().strftime('%H:%M:%S')}")

# Métricas de Destaque
c1, c2, c3, c4 = st.columns(4)
top_dy = df.sort_values('DY', ascending=False).iloc[0]
top_value = df.sort_values('Margem_Graham', ascending=False).iloc[0]

with c1:
    st.markdown(f'<div class="custom-card"><div class="metric-label">Top Dividend Yield</div><div class="metric-value">{top_dy["papel"]}</div><div style="color:#10b981; font-weight:bold;">{top_dy["DY"]:.2f}%</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="custom-card"><div class="metric-label">Desconto Graham</div><div class="metric-value">{top_value["papel"]}</div><div style="color:#2563eb; font-weight:bold;">+{top_value["Margem_Graham"]:.1f}%</div></div>', unsafe_allow_html=True)
with c3:
    # Mostra especificamente a nova ISAE4 como destaque de atualização
    isae_price = live_prices.get('ISAE4', 0)
    st.markdown(f'<div class="custom-card"><div class="metric-label">ISA Energia (ISAE4)</div><div class="metric-value">R$ {isae_price:.2f}</div><div style="color:#64748b;">Novo Ticker ✅</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="custom-card"><div class="metric-label">Ativos Monitorados</div><div class="metric-value">{len(df)}</div><div style="color:#64748b;">Bolsa Brasileira</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Tabelas de Ranking
tab1, tab2, tab3 = st.tabs(["💎 Graham (Preço Justo)", "✨ Fórmula Mágica", "💰 Dividendos"])

with tab1:
    df_g = df[['papel', 'empresa', 'preco_atual', 'Graham', 'Margem_Graham']].sort_values('Margem_Graham', ascending=False)
    st.dataframe(df_g.style.format({'preco_atual': 'R$ {:.2f}', 'Graham': 'R$ {:.2f}', 'Margem_Graham': '{:.1f}%'}), use_container_width=True, hide_index=True)

with tab2:
    df['rank_pl'] = df['PL'].rank(ascending=True)
    df['rank_roic'] = df['roic'].rank(ascending=False)
    df['score'] = df['rank_pl'] + df['rank_roic']
    df_magic = df[['papel', 'empresa', 'preco_atual', 'PL', 'roic', 'score']].sort_values('score')
    st.dataframe(df_magic.style.format({'preco_atual': 'R$ {:.2f}', 'PL': '{:.2f}', 'roic': '{:.1%}'}), use_container_width=True, hide_index=True)

with tab3:
    df_dy = df[['papel', 'empresa', 'preco_atual', 'div_ano', 'DY']].sort_values('DY', ascending=False)
    st.dataframe(df_dy.style.format({'preco_atual': 'R$ {:.2f}', 'div_ano': 'R$ {:.2f}', 'DY': '{:.2f}%'}), use_container_width=True, hide_index=True)

# Gráfico Comparativo
st.markdown("### 📊 Visão Geral: Preço Atual vs Valor de Graham")
fig = go.Figure()
fig.add_trace(go.Bar(x=df['papel'], y=df['preco_atual'], name="Preço de Mercado", marker_color='#2563eb'))
fig.add_trace(go.Scatter(x=df['papel'], y=df['Graham'], name="Preço Justo (Graham)", mode='lines+markers', line=dict(color='#10b981', width=3)))
fig.update_layout(legend=dict(orientation="h", y=1.1), plot_bgcolor='white', margin=dict(t=0, b=0))
st.plotly_chart(fig, use_container_width=True)

# Barra Lateral
st.sidebar.title("Configurações")
st.sidebar.info("Ticker TRPL4 atualizado para ISAE4 conforme nova listagem da B3.")
if st.sidebar.button("Forçar Atualização de Preços"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.write("Desenvolvido para análise fundamentalista 2026.")
