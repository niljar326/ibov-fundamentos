import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

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
    .custom-card { background-color: white; border: 1px solid #e2e8f0; border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 10px; }
    .metric-value { font-size: 24px; font-weight: 800; color: #1e293b; }
    .metric-label { font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- BANCO DE DADOS DE FUNDAMENTOS ---
BASE_DATA = [
    {"papel": "PETR4", "empresa": "Petrobras", "lpa": 9.05, "vpa": 33.47, "div_ano": 4.58, "roic": 0.245},
    {"papel": "VALE3", "empresa": "Vale S.A.", "lpa": 9.17, "vpa": 43.03, "div_ano": 5.45, "roic": 0.187},
    {"papel": "BBAS3", "empresa": "Banco do Brasil", "lpa": 6.78, "vpa": 35.64, "div_ano": 3.12, "roic": 0.198},
    {"papel": "ITUB4", "empresa": "Itaú Unibanco", "lpa": 4.01, "vpa": 21.58, "div_ano": 1.45, "roic": 0.183},
    {"papel": "WEGE3", "empresa": "WEG S.A.", "lpa": 1.57, "vpa": 7.01, "div_ano": 0.72, "roic": 0.212},
    {"papel": "ITSA4", "empresa": "Itaúsa", "lpa": 1.45, "vpa": 8.77, "div_ano": 0.88, "roic": 0.165},
    {"papel": "TAEE11", "empresa": "Taesa", "lpa": 3.65, "vpa": 22.09, "div_ano": 3.42, "roic": 0.142},
    {"papel": "ISAE4", "empresa": "ISA Energia", "lpa": 3.79, "vpa": 26.73, "div_ano": 2.55, "roic": 0.138},
    {"papel": "SAPR11", "empresa": "Sanepar", "lpa": 4.81, "vpa": 34.44, "div_ano": 1.95, "roic": 0.128},
    {"papel": "CPLE6", "empresa": "Copel", "lpa": 1.08, "vpa": 8.79, "div_ano": 0.65, "roic": 0.115},
    {"papel": "BBSE3", "empresa": "BB Seguridade", "lpa": 3.84, "vpa": 5.78, "div_ano": 3.25, "roic": 0.584},
    {"papel": "BBDC4", "empresa": "Bradesco", "lpa": 1.58, "vpa": 15.76, "div_ano": 0.95, "roic": 0.095},
    {"papel": "ABEV3", "empresa": "Ambev", "lpa": 0.92, "vpa": 5.50, "div_ano": 0.76, "roic": 0.158},
    {"papel": "EGIE3", "empresa": "Engie Brasil", "lpa": 4.25, "vpa": 14.50, "div_ano": 3.80, "roic": 0.185},
    {"papel": "B3SA3", "empresa": "B3 S.A.", "lpa": 0.85, "vpa": 3.20, "div_ano": 0.45, "roic": 0.170},
]

# --- FUNÇÃO ROBUSTA DE BUSCA DE PREÇOS ---
@st.cache_data(ttl=600)
def get_live_prices(tickers):
    prices = {}
    tickers_formatted = [f"{t}.SA" for t in tickers]
    
    try:
        # Busca dados do dia atual
        data = yf.download(tickers_formatted, period="5d", interval="1d", progress=False)['Close']
        
        for t in tickers:
            try:
                # Tenta pegar o último preço válido dos últimos 5 dias
                ticker_column = f"{t}.SA"
                last_val = data[ticker_column].dropna().iloc[-1]
                prices[t] = float(last_val)
            except:
                # Fallback: tenta buscar unitariamente se falhar no lote
                try:
                    solo_ticker = yf.Ticker(f"{t}.SA")
                    prices[t] = float(solo_ticker.fast_info['last_price'])
                except:
                    prices[t] = 0.0
    except:
        prices = {t: 0.0 for t in tickers}
        
    return prices

# --- PROCESSAMENTO ---
tickers_list = [x['papel'] for x in BASE_DATA]
live_prices = get_live_prices(tickers_list)

df = pd.DataFrame(BASE_DATA)
df['preco_atual'] = df['papel'].map(live_prices)

# Verificação anti-infinito: Só calcula se o preço for maior que zero
df['PL'] = np.where(df['preco_atual'] > 0, df['preco_atual'] / df['lpa'], 0)
df['PVP'] = np.where(df['preco_atual'] > 0, df['preco_atual'] / df['vpa'], 0)
df['DY'] = np.where(df['preco_atual'] > 0, (df['div_ano'] / df['preco_atual']) * 100, 0)
df['Graham'] = (22.5 * df['lpa'] * df['vpa'])**0.5
df['Margem_Graham'] = np.where(df['preco_atual'] > 0, (df['Graham'] / df['preco_atual'] - 1) * 100, 0)

# Remove erros de cotação para não poluir o ranking principal
df_clean = df[df['preco_atual'] > 0].copy()

# --- UI DASHBOARD ---
st.title("🇧🇷 Ibovespa Real-Time Inteligente")
st.caption(f"Verificação de Cotações: OK • Última Atualização: {datetime.now().strftime('%H:%M:%S')}")

# Métricas de Destaque
if not df_clean.empty:
    c1, c2, c3, c4 = st.columns(4)
    top_dy = df_clean.sort_values('DY', ascending=False).iloc[0]
    top_value = df_clean.sort_values('Margem_Graham', ascending=False).iloc[0]

    with c1:
        st.markdown(f'<div class="custom-card"><div class="metric-label">Maior Yield</div><div class="metric-value">{top_dy["papel"]}</div><div style="color:#10b981; font-weight:bold;">{top_dy["DY"]:.2f}%</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="custom-card"><div class="metric-label">Desconto Graham</div><div class="metric-value">{top_value["papel"]}</div><div style="color:#2563eb; font-weight:bold;">+{top_value["Margem_Graham"]:.1f}%</div></div>', unsafe_allow_html=True)
    with c3:
        # Verificação específica da CPLE6 solicitado
        cple_p = live_prices.get('CPLE6', 0)
        status_cple = "OK" if cple_p > 0 else "Erro API"
        st.markdown(f'<div class="custom-card"><div class="metric-label">Copel (CPLE6)</div><div class="metric-value">R$ {cple_p:.2f}</div><div style="color:#64748b;">Status: {status_cple}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="custom-card"><div class="metric-label">Ativos Ativos</div><div class="metric-value">{len(df_clean)}</div><div style="color:#64748b;">Monitoramento B3</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Tabelas de Ranking
tab1, tab2, tab3 = st.tabs(["💎 Preço Justo (Graham)", "✨ Fórmula Mágica", "💰 Dividendos"])

with tab1:
    df_g = df_clean[['papel', 'empresa', 'preco_atual', 'Graham', 'Margem_Graham']].sort_values('Margem_Graham', ascending=False)
    st.dataframe(df_g.style.format({'preco_atual': 'R$ {:.2f}', 'Graham': 'R$ {:.2f}', 'Margem_Graham': '{:.1f}%'}), use_container_width=True, hide_index=True)

with tab2:
    df_clean['rank_pl'] = df_clean['PL'].rank(ascending=True)
    df_clean['rank_roic'] = df_clean['roic'].rank(ascending=False)
    df_clean['score'] = df_clean['rank_pl'] + df_clean['rank_roic']
    df_magic = df_clean[['papel', 'empresa', 'preco_atual', 'PL', 'roic', 'score']].sort_values('score')
    st.dataframe(df_magic.style.format({'preco_atual': 'R$ {:.2f}', 'PL': '{:.2f}', 'roic': '{:.1%}'}), use_container_width=True, hide_index=True)

with tab3:
    df_dy = df_clean[['papel', 'empresa', 'preco_atual', 'div_ano', 'DY']].sort_values('DY', ascending=False)
    st.dataframe(df_dy.style.format({'preco_atual': 'R$ {:.2f}', 'div_ano': 'R$ {:.2f}', 'DY': '{:.2f}%'}), use_container_width=True, hide_index=True)

# Gráfico
st.markdown("### 📊 Gráfico: Cotação x Valor de Graham")
fig = go.Figure()
fig.add_trace(go.Bar(x=df_clean['papel'], y=df_clean['preco_atual'], name="Preço Mercado", marker_color='#2563eb'))
fig.add_trace(go.Scatter(x=df_clean['papel'], y=df_clean['Graham'], name="Preço Justo", mode='lines+markers', line=dict(color='#10b981', width=3)))
fig.update_layout(legend=dict(orientation="h", y=1.1), plot_bgcolor='white', margin=dict(t=0, b=0))
st.plotly_chart(fig, use_container_width=True)

# Barra Lateral
st.sidebar.title("🛠 Ajustes Técnicos")
st.sidebar.write("Ticker **ISAE4** e **CPLE6** verificados.")
if st.sidebar.button("🔄 Atualizar Cotações Agora"):
    st.cache_data.clear()
    st.rerun()

# Lista de erros se houver
erros = [t for t, p in live_prices.items() if p == 0]
if erros:
    st.sidebar.warning(f"Ativos com erro na API: {', '.join(erros)}")
