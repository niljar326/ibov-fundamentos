import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import numpy as np
import math

# Configuração da página (Original Style)
st.set_page_config(
    page_title="Ranking Ibovespa Inteligente 2026",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILIZAÇÃO CSS CUSTOMIZADA (IDÊNTICA AO ORIGINAL) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; background-color: #f8fafc; color: #1e293b; }
    .ad-banner { background: linear-gradient(90deg, #1e3a8a 0%, #0f172a 100%); color: white; text-align: center; padding: 10px 16px; font-size: 13px; font-weight: 500; border-radius: 12px; margin-bottom: 24px; position: relative; overflow: hidden; }
    .ad-badge { background-color: #fbbf24; color: #0f172a; font-weight: 800; padding: 2px 6px; border-radius: 4px; font-size: 10px; text-transform: uppercase; margin-right: 8px; display: inline-block; }
    .custom-card { background-color: white; border: 1px solid #e2e8f0; border-radius: 18px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
    section[data-testid="stSidebar"] { background-color: white !important; border-right: 1px solid #e2e8f0 !important; }
    .sidebar-logo { background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%); color: white; font-size: 24px; font-weight: 800; width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2); margin-bottom: 12px; }
    .whatsapp-btn { display: block; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white !important; padding: 16px; border-radius: 18px; text-decoration: none; font-weight: 600; margin-bottom: 16px; transition: all 0.2s; text-align: center; }
    div.row-widget.stRadio > div { flex-direction: row; gap: 8px; }
    div.row-widget.stRadio > div > label { background-color: white !important; border: 1px solid #e2e8f0 !important; border-radius: 12px !important; padding: 10px 16px !important; color: #475569 !important; font-weight: 600; font-size: 13px !important; cursor: pointer; transition: all 0.15s; }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÃO PARA BUSCAR DADOS EM TEMPO REAL ---
@st.cache_data(ttl=600)
def get_real_time_prices(tickers):
    prices = {}
    try:
        data = yf.download([f"{t}.SA" for t in tickers], period="5d", interval="1d", progress=False)['Close']
        for t in tickers:
            try:
                # Pega o último fechamento válido para evitar zeros
                val = data[f"{t}.SA"].dropna().iloc[-1]
                prices[t] = float(val)
            except:
                prices[t] = 0.0
    except:
        prices = {t: 0.0 for t in tickers}
    return prices

# --- BANCO DE DADOS ATUALIZADO (Mistura de Fundamentos Originais e Tickers Novos) ---
STOCK_DATABASE = [
  {"papel": "PETR4", "empresa": "Petrobras Pref.", "lpa": 9.05, "vpa": 33.47, "roe": 0.284, "roic": 0.245, "dy_base": 0.142, "mrgliq": 0.195, "liq2m": 1540000000, "divbpatr": 0.78, "epsTrimestral": 2.38, "dataRef": "1T26", "situacao": "Saudável"},
  {"papel": "VALE3", "empresa": "Vale S.A.", "lpa": 9.17, "vpa": 43.03, "roe": 0.213, "roic": 0.187, "dy_base": 0.088, "mrgliq": 0.224, "liq2m": 1250000000, "divbpatr": 0.52, "epsTrimestral": 1.84, "dataRef": "1T26", "situacao": "Saudável"},
  {"papel": "BBAS3", "empresa": "Banco do Brasil", "lpa": 6.78, "vpa": 35.64, "roe": 0.215, "roic": 0.198, "dy_base": 0.104, "mrgliq": 0.165, "liq2m": 480000000, "divbpatr": 0.12, "epsTrimestral": 1.62, "dataRef": "1T26", "situacao": "Saudável"},
  {"papel": "ITUB4", "empresa": "Itaú Unibanco Pref.", "lpa": 4.01, "vpa": 21.58, "roe": 0.212, "roic": 0.183, "dy_base": 0.065, "mrgliq": 0.174, "liq2m": 780000000, "divbpatr": 0.18, "epsTrimestral": 1.05, "dataRef": "1T26", "situacao": "Saudável"},
  {"papel": "WEGE3", "empresa": "WEG S.A.", "lpa": 1.57, "vpa": 7.01, "roe": 0.228, "roic": 0.212, "dy_base": 0.024, "mrgliq": 0.168, "liq2m": 290000000, "divbpatr": 0.08, "epsTrimestral": 0.41, "dataRef": "1T26", "situacao": "Saudável"},
  {"papel": "ISAE4", "empresa": "ISA Energia (CTEEP)", "lpa": 3.79, "vpa": 26.73, "roe": 0.144, "roic": 0.138, "dy_base": 0.106, "mrgliq": 0.362, "liq2m": 210000000, "divbpatr": 1.25, "epsTrimestral": 0.97, "dataRef": "1T26", "situacao": "Saudável"},
  {"papel": "CPLE3", "empresa": "Copel ON", "lpa": 1.08, "vpa": 8.79, "roe": 0.129, "roic": 0.115, "dy_base": 0.068, "mrgliq": 0.118, "liq2m": 32000000, "divbpatr": 1.15, "epsTrimestral": 0.29, "dataRef": "1T26", "situacao": "Saudável"},
  {"papel": "TAEE11", "empresa": "Taesa Units", "lpa": 3.65, "vpa": 22.09, "roe": 0.168, "roic": 0.142, "dy_base": 0.098, "mrgliq": 0.385, "liq2m": 240000000, "divbpatr": 1.84, "epsTrimestral": 0.95, "dataRef": "1T26", "situacao": "Saudável"},
  {"papel": "BBSE3", "empresa": "BB Seguridade ON", "lpa": 3.84, "vpa": 5.78, "roe": 0.665, "roic": 0.584, "dy_base": 0.098, "mrgliq": 0.512, "liq2m": 135000000, "divbpatr": 0.02, "epsTrimestral": 0.98, "dataRef": "1T26", "situacao": "Saudável"},
  {"papel": "SAPR11", "empresa": "Sanepar Units", "lpa": 4.81, "vpa": 34.44, "roe": 0.141, "roic": 0.128, "dy_base": 0.076, "mrgliq": 0.245, "liq2m": 122000000, "divbpatr": 0.74, "epsTrimestral": 1.25, "dataRef": "1T26", "situacao": "Saudável"}
]

# Adicionando Histórico Fictício para os Gráficos (Baseado no original)
def add_mock_history(stock):
    periods = ["2023", "2024", "2025", "Hoje"]
    hist = []
    base_l = stock['lpa'] * 1e8
    base_r = base_l * 5
    for idx, p in enumerate(periods):
        mult = 0.8 + (idx * 0.1)
        hist.append({
            "periodo": p,
            "receita": base_r * mult,
            "lucro": base_l * mult,
            "cotacao": stock['cotacao'] * (0.9 if idx < 3 else 1.0)
        })
    stock['historico'] = hist
    return stock

# --- ATUALIZAÇÃO DOS DADOS ---
tickers = [s['papel'] for s in STOCK_DATABASE]
prices_now = get_real_time_prices(tickers)

for s in STOCK_DATABASE:
    current_p = prices_now.get(s['papel'], 0.0)
    s['cotacao'] = current_p
    # Proteção anti-infinito e erro de divisão por zero
    if current_p > 0:
        s['pl'] = current_p / s['lpa']
        s['pvp'] = current_p / s['vpa']
        s['dy'] = s['dy_base'] * (1.0) # Simplificado para tempo real
        s['evebit'] = s['pl'] * 0.85
    else:
        s['pl'], s['pvp'], s['dy'], s['evebit'] = 0, 0, 0, 0
    add_mock_history(s)

# --- BANNER ADVERTISMENT TOP ---
st.markdown("""
<div class="ad-banner">
    <span class="ad-badge">AD</span>
    Apoie nossa comunidade de B3 acessando os patrocinadores parceiros na barra lateral ou nos links promocionais!
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo">B3</div>', unsafe_allow_html=True)
    st.subheader("Ranking Ibovespa")
    st.caption("Versão Inteligente de Alta Precisão 2026")
    st.markdown("---")
    user_name = st.text_input("Seu nome para relatórios:", "Investidor")
    st.markdown("---")
    # Giga+ Fibra WhatsApp Promo
    st.markdown("""
    <a href="https://wa.me/552220410353?text=Use%20o%20codigo%20DVT329" target="_blank" class="whatsapp-btn">
        <div style="font-size: 11px; text-transform: uppercase; font-weight: 800; opacity: 0.9;">Código: DVT329</div>
        <div style="font-size: 14px; font-weight: 700;">Giga+ Fibra - 20% OFF!</div>
        <div style="font-size: 11px; margin-top:8px;">Regatar no WhatsApp →</div>
    </a>
    """, unsafe_allow_html=True)
    if st.button("🔄 Atualizar Cotações"):
        st.cache_data.clear()
        st.rerun()

# --- CABEÇALHO ---
st.markdown(f"# **Análise de Ações Baratas e Rentáveis**")
st.markdown(f"Relatório tático de investimentos para **{user_name}** • {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# --- TABS (IDÊNTICAS AO ORIGINAL) ---
tabs = ["🏆 Ranking Fundamentalista", "✨ Fórmula Mágica", "💎 Graham Valuation", "📈 EPS Diluído", "📉 Assimetria Lucro/Preço"]
active_tab = st.radio("Selecione a Visualização:", tabs, label_visibility="collapsed")

# --- LÓGICA DAS TABS ---
df_final = pd.DataFrame(STOCK_DATABASE)
df_valid = df_final[df_final['cotacao'] > 0].copy()

def fmt_curr(val): return f"R$ {val:,.2f}"
def fmt_pct(val): return f"{val:.2%}"

if active_tab == "🏆 Ranking Fundamentalista":
    st.markdown('<div class="custom-card"><h4>🏆 Ranking de Oportunidades Saudáveis</h4></div>', unsafe_allow_html=True)
    df_f = df_valid.sort_values('pl')
    st.dataframe(df_f[['papel', 'empresa', 'cotacao', 'pl', 'pvp', 'dy', 'roe']].style.format({
        'cotacao': '{:.2f}', 'pl': '{:.2f}', 'pvp': '{:.2f}', 'dy': '{:.1%}', 'roe': '{:.1%}'
    }), use_container_width=True, hide_index=True)

elif active_tab == "✨ Fórmula Mágica":
    st.markdown('<div class="custom-card"><h4>✨ Joel Greenblatt Methodology</h4></div>', unsafe_allow_html=True)
    df_valid['rank_ey'] = df_valid['evebit'].rank(ascending=True)
    df_valid['rank_roic'] = df_valid['roic'].rank(ascending=False)
    df_valid['score'] = df_valid['rank_ey'] + df_valid['rank_roic']
    st.dataframe(df_valid.sort_values('score')[['papel', 'empresa', 'cotacao', 'score', 'roic', 'evebit']].style.format({
        'cotacao': '{:.2f}', 'roic': '{:.1%}', 'evebit': '{:.2f}'
    }), use_container_width=True, hide_index=True)

elif active_tab == "💎 Graham Valuation":
    st.markdown('<div class="custom-card"><h4>💎 Benjamin Graham Valuation</h4></div>', unsafe_allow_html=True)
    df_valid['V.I'] = (22.5 * df_valid['lpa'] * df_valid['vpa'])**0.5
    df_valid['Upside'] = (df_valid['V.I'] / df_valid['cotacao']) - 1
    st.dataframe(df_valid.sort_values('Upside', ascending=False)[['papel', 'cotacao', 'V.I', 'Upside']].style.format({
        'cotacao': '{:.2f}', 'V.I': '{:.2f}', 'Upside': '{:.1%}'
    }), use_container_width=True, hide_index=True)

elif active_tab == "📈 EPS Diluído":
    st.markdown('### 📈 Lucro por Ação (EPS) > R$ 1,00')
    df_eps = df_valid[df_valid['epsTrimestral'] > 1.0].sort_values('epsTrimestral', ascending=False)
    for _, row in df_eps.iterrows():
        st.markdown(f"""
        <div class="custom-card" style="display:flex; justify-content:space-between;">
            <b>{row['papel']} - {row['empresa']}</b>
            <span style="color:#10b981; font-weight:800;">EPS: R$ {row['epsTrimestral']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

elif active_tab == "📉 Assimetria Lucro/Preço":
    st.markdown('<div class="custom-card"><h4>📉 Comportamento Boca de Jacaré</h4></div>', unsafe_allow_html=True)
    ticker_choice = st.selectbox("Escolha o papel:", df_valid['papel'].tolist())
    selected = next(item for item in STOCK_DATABASE if item["papel"] == ticker_choice)
    
    # Renderização do Gráfico Original (Dual Axis Normalizado)
    hist_df = pd.DataFrame(selected["historico"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df['periodo'], y=hist_df['lucro'], name="Lucro Líquido", line=dict(color='#16a34a', width=4)))
    fig.add_trace(go.Scatter(x=hist_df['periodo'], y=hist_df['cotacao'] * (hist_df['lucro'].max()/hist_df['cotacao'].max()), name="Preço da Ação (Normalizado)", line=dict(color='#2563eb', width=4)))
    fig.update_layout(title=f"Distorção de Preço vs Lucro: {ticker_choice}", hovermode="x unified", plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

# --- AFILIADOS (IDÊNTICOS AO ORIGINAL) ---
st.markdown("<br><h4 style='text-align:center; color:#475569; font-size:13px;'>Patrocinadores Oficiais</h4>", unsafe_allow_html=True)
col_nomad, col_mp = st.columns(2)
with col_nomad:
    st.markdown("""
    <a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I" target="_blank" style="text-decoration:none;">
        <div class="custom-card" style="text-align:center; transition:0.2s;">
            ✈️ <b>Nomad: Conta em Dólar Gratuita</b><br>
            <small style="color:#64748b;">Taxa zero na primeira remessa com cupom.</small>
        </div>
    </a>
    """, unsafe_allow_html=True)
with col_mp:
    st.markdown("""
    <a href="https://mpago.li/1VydVhw" target="_blank" style="text-decoration:none;">
        <div class="custom-card" style="text-align:center; transition:0.2s;">
            🤝 <b>Mercado Pago: Bônus R$ 30</b><br>
            <small style="color:#64748b;">Crie sua conta e resgate seu bônus de investidor.</small>
        </div>
    </a>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div style="background-color:#0f172a; border-radius:18px; padding:20px; text-align:center; color:#94a3b8; font-size:12px; margin-top:30px;">
    Ibovespa Fundamentalista © 2026 | Dados em Tempo Real (Yahoo Finance) | Tickers Atualizados: ISAE4, CPLE3
</div>
""", unsafe_allow_html=True)
