import streamlit as st
import streamlit.components.v1 as components 
import pandas as pd
import plotly.graph_objects as go
import feedparser
import yfinance as yf
import datetime
from datetime import timedelta
from time import mktime
import json
import os
import uuid

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Melhores Ações Ibovespa 2025 | Ranking Fundamentalista e Dividendos",
    layout="wide",
    page_icon="🇧🇷"
)

# Tenta importar fundamentus
try: import fundamentus
except ImportError:
    st.error("Biblioteca 'fundamentus' não encontrada.")
    st.stop()

# --- 2. GERENCIAMENTO DE ESTADO ---
if 'access_key_tab1_vFinal' not in st.session_state: st.session_state.access_key_tab1_vFinal = False
if 'access_key_tab3_vFinal' not in st.session_state: st.session_state.access_key_tab3_vFinal = False
if 'tv_symbol' not in st.session_state: st.session_state.tv_symbol = "BMFBOVESPA:LREN3"
if 'expander_open' not in st.session_state: st.session_state.expander_open = True
if 'app_liberado' not in st.session_state: st.session_state.app_liberado = False

def unlock_tab1(): st.session_state.access_key_tab1_vFinal = True
def unlock_tab3(): st.session_state.access_key_tab3_vFinal = True
def liberar_acesso(): st.session_state.app_liberado = True
def close_expander(): st.session_state.expander_open = False

# --- CSS ---
st.markdown("""
    <style>
    [data-testid="stDataFrame"] table tr th { text-align: right !important; }
    [data-testid="stDataFrame"] table tr td { text-align: left !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 5px 5px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #ff4b4b; }
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    whatsapp_number = "552220410353"
    whatsapp_msg = "Use o codigo DVT329 e ganhe 20% nas duas primeiras mensalidades"
    whatsapp_url = f"https://wa.me/{whatsapp_number}?text={whatsapp_msg.replace(' ', '%20')}"
    st.markdown(f"""
        <a href="{whatsapp_url}" target="_blank" style="text-decoration: none;">
            <div style="background-color: #25D366; padding: 12px; border-radius: 10px; display: flex; align-items: center; justify-content: center; gap: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); margin-bottom: 20px; transition: transform 0.2s;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg"><path d="M12.0117 2C6.50574 2 2.02344 6.47837 2.02344 11.9841C2.02344 13.7432 2.48045 15.4608 3.36889 16.9982L2 22L7.1264 20.6558C8.59864 21.4589 10.2798 21.8805 11.9839 21.8805H12.0089C17.5149 21.8805 21.9953 17.4022 21.9953 11.8965C21.9953 9.25595 20.9666 6.77209 19.098 4.90706C17.2295 3.04204 14.7438 2.01235 12.0117 2Z"/></svg>
                <div style="color: white; font-weight: bold; font-size: 14px; line-height: 1.2;">Ganhe 20% OFF<br>Giga+ Fibra!</div>
            </div>
        </a>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("Desenvolvido com Streamlit")

# --- AUXILIARES ---
def clean_fundamentus_col(x):
    if pd.isna(x) or x == '': return 0.0
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        x = x.strip()
        if x.endswith('%'):
            x = x.replace('%', '').replace('.', '').replace(',', '.')
            try: return float(x) / 100
            except: return 0.0
        x = x.replace('.', '').replace(',', '.')
        try: return float(x)
        except: return 0.0
    return 0.0

# Formatação para o Gráfico (1 casa decimal)
def format_short_decimal(val):
    if pd.isna(val) or val == 0: return "0"
    abs_val = abs(val)
    if abs_val >= 1e9: return f"{val/1e9:.1f}B"
    elif abs_val >= 1e6: return f"{val/1e6:.1f}M"
    elif abs_val >= 1e3: return f"{val/1e3:.1f}K"
    return f"{val:.1f}"

def get_current_data():
    now = datetime.datetime.now()
    return now.strftime("%B"), now.year

# --- TELA DE BLOQUEIO (ADS) ---
def show_lock_screen(key_id):
    st.info("🔒 **Conteúdo Bloqueado:** Para visualizar os Rankings e Gráficos, por favor interaja com os parceiros abaixo.")
    
    # Script da Propaganda
    components.html("""
        <div style="display: flex; justify-content: center; align-items: center; width: 100%; height: 100%;">
            <script src="https://pl28325401.effectivegatecpm.com/1a/83/79/1a8379a4a8ddb94a327a5797257a9f02.js"></script>
        </div>
    """, height=130)
    
    # Banners de Afiliados
    col_ad1, col_ad2 = st.columns(2)
    with col_ad1:
        st.markdown("""
        <div style="background-color: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 10px; color: #333; height: 100%;">
            <h4 style="margin-top:0; color: #333;">✈️ Nomad: Taxa Zero em Dólar</h4>
            <p style="font-size: 14px;">Ganhe taxa zero na 1ª conversão.</p>
            <div style="text-align:center;"><a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I%26n=Jader" target="_blank" style="text-decoration: none; color: white; background-color: #1a1a1a; padding: 10px 15px; border-radius: 5px; font-size: 14px; display: inline-block; width: 100%;">➡️ <b>Abrir Conta Nomad</b></a></div>
        </div>
        """, unsafe_allow_html=True)
    with col_ad2:
        st.markdown("""
        <div style="background-color: #eaf6ff; border: 1px solid #bae0ff; padding: 15px; border-radius: 10px; color: #333; height: 100%;">
            <h4 style="margin-top:0; color: #009ee3;">🤝 Mercado Pago: R$ 30 OFF</h4>
            <p style="font-size: 14px;">Ganhe <b>R$ 30 de desconto</b>.</p>
            <div style="text-align:center;"><a href="https://mpago.li/1VydVhw" target="_blank" style="text-decoration: none; color: white; background-color: #009ee3; padding: 10px 15px; border-radius: 5px; font-size: 14px; display: inline-block; width: 100%;">➡️ <b>Resgatar R$ 30</b></a></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.button("🔓 Já visitei o anúncio / LIBERAR SITE", type="primary", on_click=liberar_acesso, key=f"btn_unlock_{key_id}")

# --- FUNÇÃO BANNERS (RODAPÉ) ---
def show_affiliate_banners():
    st.divider()
    col_ad1, col_ad2 = st.columns(2)
    with col_ad1:
        st.markdown("""
        <div style="background-color: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 10px; color: #333; height: 100%;">
            <h4 style="margin-top:0; color: #333;">✈️ Nomad: Taxa Zero em Dólar</h4>
            <p style="font-size: 14px;">Ganhe taxa zero na 1ª conversão.</p>
            <div style="text-align:center;"><a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I%26n=Jader" target="_blank" style="text-decoration: none; color: white; background-color: #1a1a1a; padding: 10px 15px; border-radius: 5px; font-size: 14px; display: inline-block; width: 100%;">➡️ <b>Abrir Conta Nomad</b></a></div>
        </div>
        """, unsafe_allow_html=True)
    with col_ad2:
        st.markdown("""
        <div style="background-color: #eaf6ff; border: 1px solid #bae0ff; padding: 15px; border-radius: 10px; color: #333; height: 100%;">
            <h4 style="margin-top:0; color: #009ee3;">🤝 Mercado Pago: R$ 30 OFF</h4>
            <p style="font-size: 14px;">Ganhe <b>R$ 30 de desconto</b>.</p>
            <div style="text-align:center;"><a href="https://mpago.li/1VydVhw" target="_blank" style="text-decoration: none; color: white; background-color: #009ee3; padding: 10px 15px; border-radius: 5px; font-size: 14px; display: inline-block; width: 100%;">➡️ <b>Resgatar R$ 30</b></a></div>
        </div>
        """, unsafe_allow_html=True)

# --- DADOS PRINCIPAIS ---
@st.cache_data(ttl=3600*6)
def get_ranking_data():
    try:
        df = fundamentus.get_resultado()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'papel'}, inplace=True)
        # Adicionado 'roic' na lista de colunas para tratar
        cols = ['pl', 'roe', 'dy', 'evebit', 'cotacao', 'liq2m', 'mrgliq', 'divbpatr', 'c5y', 'roic']
        for col in cols:
            if col in df.columns: df[col] = df[col].apply(clean_fundamentus_col)
            else: df[col] = 0.0
        return df
    except: return pd.DataFrame()

def apply_best_filters(df):
    if df.empty: return df
    df_f = df.copy()
    filtro = (
        (df_f['roe'] > 0.05) & (df_f['pl'] < 15) & (df_f['pl'] > 0) & 
        (df_f['evebit'] > 0) & (df_f['evebit'] < 10) &
        (df_f['dy'] > 0.04) & (df_f['mrgliq'] > 0.05) & (df_f['liq2m'] > 200000)
    )
    df_f = df_f[filtro].copy()
    df_f['dy'] = df_f['dy'] * 100
    df_f['mrgliq'] = df_f['mrgliq'] * 100
    df_f['roe'] = df_f['roe'] * 100
    df_f.rename(columns={
        'papel': 'Ativo', 'cotacao': 'Preço', 'pl': 'P/L', 
        'evebit': 'EV/EBIT', 'dy': 'DY', 'roe': 'ROE', 'mrgliq': 'Margem Líq.'
    }, inplace=True)
    return df_f.sort_values(by=['P/L', 'Margem Líq.'], ascending=[True, False]).reset_index(drop=True)

# --- FÓRMULA MÁGICA ---
@st.cache_data(ttl=3600*6)
def get_magic_formula_data(df_base):
    if df_base.empty: return pd.DataFrame()
    
    # Copia para não alterar o original
    df = df_base.copy()
    
    # 1. Filtros iniciais de consistência
    # Garantir liquidez mínima > 100k
    df = df[df['liq2m'] > 100000]
    # Remover empresas com EBIT/EV negativo (não aplicável para a fórmula padrão de "empresas baratas e boas")
    df = df[df['evebit'] > 0]
    
    # 2. Selecionar Top 400 por Liquidez
    df = df.sort_values(by='liq2m', ascending=False).head(400)
    
    # 3. Calcular Métricas
    # EY (Earning Yield) = EBIT / Valor da Empresa.
    # O fundamentus fornece EV/EBIT. Logo, EY = 1 / (EV/EBIT).
    df['ey_score'] = 1 / df['evebit']
    
    # 4. Rankear
    # EY: Maior é melhor (descending) -> Rank 1
    df['rank_ey'] = df['ey_score'].rank(ascending=False)
    
    # ROIC: Maior é melhor (descending) -> Rank 1
    df['rank_roic'] = df['roic'].rank(ascending=False)
    
    # 5. Soma dos Ranks (Fórmula Mágica)
    df['magic_points'] = df['rank_ey'] + df['rank_roic']
    
    # 6. Ordenar pelo menor Rank acumulado (Melhor empresa)
    df_final = df.sort_values(by='magic_points', ascending=True).head(40)
    
    # Formatação para exibição
    df_final['roic_fmt'] = (df_final['roic'] * 100).apply(lambda x: f"{x:.1f}%")
    df_final['ey_fmt'] = (df_final['ey_score'] * 100).apply(lambda x: f"{x:.1f}%")
    
    # Seleção de colunas (removida a liq2m da exibição final)
    cols_out = ['papel', 'cotacao', 'magic_points', 'rank_ey', 'rank_roic', 'ey_fmt', 'roic_fmt', 'evebit']
    df_final = df_final[cols_out].copy()
    
    df_final.rename(columns={
        'papel': 'Ativo',
        'cotacao': 'Preço',
        'magic_points': 'Score Mágico',
        'rank_ey': 'Rank EY',
        'rank_roic': 'Rank ROIC',
        'ey_fmt': 'Earning Yield',
        'roic_fmt': 'ROIC',
        'evebit': 'EV/EBIT'
    }, inplace=True)
    
    return df_final.reset_index(drop=True)

# --- GRAHAM VALUATION ---
@st.cache_data(ttl=3600*6)
def get_graham_data(df_base):
    if df_base.empty: return pd.DataFrame()
    
    df = df_base.copy()
    
    # 1. Filtro Universo 400 + Liquidez > 100k
    df = df[df['liq2m'] > 100000]
    df = df[df['pl'] > 0]
    df = df.sort_values(by='liq2m', ascending=False).head(400)
    
    # 2. Calcular LPA (Preço / PL)
    df['lpa'] = df['cotacao'] / df['pl']
    
    # 3. Parâmetros da Fórmula
    Y_rate = 6.0 
    growth_factor = 8.5 + (2 * 10) 
    const_graham = 4.4
    
    df['valor_intrinseco'] = (df['lpa'] * growth_factor * const_graham) / Y_rate
    
    # 4. Divisão do valor da ação por V (Ratio)
    df['ratio_graham'] = df['cotacao'] / df['valor_intrinseco']
    
    # 5. Classificação (Status)
    def classify(r):
        if r < 1.0: return "Barata"
        elif r > 1.0: return "Cara"
        else: return "Justo"
        
    df['Status'] = df['ratio_graham'].apply(classify)
    df = df.sort_values(by='ratio_graham', ascending=True)
    
    cols_out = ['papel', 'cotacao', 'valor_intrinseco', 'ratio_graham', 'Status', 'lpa']
    df_final = df[cols_out].copy()
    
    df_final.rename(columns={
        'papel': 'Ativo', 'cotacao': 'Preço Atual',
        'valor_intrinseco': 'Preço Justo (Graham)', 'ratio_graham': 'Ratio (P/V)', 'lpa': 'LPA'
    }, inplace=True)
    
    return df_final.reset_index(drop=True)

# --- NOVO: ABA 4 (LUCRO VS PREÇO) ---
@st.cache_data(ttl=3600*6)
def get_earnings_gap_data(df_base):
    if df_base.empty: return pd.DataFrame()
    
    df = df_base.copy()
    
    # Filtro Top 200 e Liquidez > 100k
    df = df[df['liq2m'] > 100000]
    # Filtra PL positivo (para ter lucro real)
    df = df[df['pl'] > 0]
    
    # Pega top 200 liquidez
    df = df.sort_values(by='liq2m', ascending=False).head(200)
    
    # Calcular LPA
    df['lpa'] = df['cotacao'] / df['pl']
    
    # Calcular Earnings Yield (Lucro / Preço) em %
    # Quanto maior, mais lucro a empresa entrega pelo preço que custa
    df['yield'] = (df['lpa'] / df['cotacao']) * 100
    
    # Ordenar do MAIOR yield (maior distância do lucro para o preço) para o menor
    df = df.sort_values(by='yield', ascending=False)
    
    cols_out = ['papel', 'cotacao', 'lpa', 'pl', 'yield']
    df_final = df[cols_out].copy()
    
    df_final.rename(columns={
        'papel': 'Ativo',
        'cotacao': 'Preço',
        'lpa': 'LPA (Lucro/Ação)',
        'pl': 'P/L',
        'yield': 'Earnings Yield (%)'
    }, inplace=True)
    
    return df_final.reset_index(drop=True)


@st.cache_data(ttl=3600*12)
def get_risk_table(df_original):
    if df_original.empty: return pd.DataFrame()
    lista_rj = ['OIBR3', 'OIBR4', 'AMER3', 'GOLL4', 'AZUL4', 'RCSL3', 'RCSL4', 'RSID3', 'TCNO3', 'TCNO4']
    mask_risk = (df_original['divbpatr'] > 3.0) | (df_original['papel'].isin(lista_rj))
    df_risk = df_original[mask_risk].copy()
    if df_risk.empty: return pd.DataFrame()
    risk_data = []
    top_risks = df_risk.sort_values(by='divbpatr', ascending=False).head(15)
    for idx, row in top_risks.iterrows():
        ticker = row['papel']
        status = "Recup. Judicial" if ticker in lista_rj else "Alta Alavancagem"
        lucro_queda_str, val_queda = "N/D", 0.0
        try:
            stock = yf.Ticker(ticker + ".SA")
            fin = stock.financials
            if not fin.empty:
                inc_row = None
                possible = ['Net Income', 'Net Income Common', 'Lucro Liquido', 'Net Income Continuous']
                for name in possible:
                    matches = [i for i in fin.index if name.lower() in str(i).lower()]
                    if matches:
                        inc_row = fin.loc[matches[0]]
                        break
                if inc_row is not None and len(inc_row) >= 2:
                    if inc_row.iloc[0] < inc_row.iloc[1]:
                        pct = ((inc_row.iloc[0] - inc_row.iloc[1]) / abs(inc_row.iloc[1])) * 100
                        val_queda = pct 
                        lucro_queda_str = f"{pct:.1f}%"
                    else: lucro_queda_str = "Estável"
        except: pass
        if val_queda < 0 or ticker in lista_rj:
            risk_data.append({'Ativo': ticker, 'Preço': row['cotacao'], 'Alavancagem': row['divbpatr'], 'Queda Lucro': lucro_queda_str, 'Situação': status})
    return pd.DataFrame(risk_data)

# --- GRÁFICO ---
@st.cache_data(ttl=3600*24)
def get_chart_data(ticker):
    try:
        stock = yf.Ticker(ticker + ".SA")
        
        # 1. Coleta dos Dataframes
        financials = stock.financials.T
        quarterly = stock.quarterly_financials.T
        hist = stock.history(period="5y")
        
        # Ajuste de Fuso Horário
        if not financials.empty: financials.index = pd.to_datetime(financials.index).tz_localize(None)
        if not quarterly.empty: quarterly.index = pd.to_datetime(quarterly.index).tz_localize(None)
        if not hist.empty: hist.index = pd.to_datetime(hist.index).tz_localize(None)

        # Ordenar (Do mais antigo para o mais novo)
        financials = financials.sort_index()
        quarterly = quarterly.sort_index()

        # Função auxiliar para encontrar colunas
        def find_col(df, candidates):
            cols_str = [str(c) for c in df.columns]
            for cand in candidates:
                for i, col_name in enumerate(cols_str):
                    if cand.lower() in col_name.lower():
                        return df.columns[i]
            return None

        rev_candidates = ['Total Revenue', 'Operating Revenue', 'Revenue', 'Receita']
        inc_candidates = ['Net Income', 'Net Income Common', 'Lucro', 'Net Income Continuous']

        rev_col = find_col(financials, rev_candidates)
        inc_col = find_col(financials, inc_candidates)
        
        if not rev_col or not inc_col: return None

        data_rows = []
        current_year = datetime.datetime.now().year
        
        # 2. Dados Anuais
        past_years = financials[financials.index.year < current_year]
        last_4_years = past_years.tail(4)
        
        for date, row in last_4_years.iterrows():
            year_str = str(date.year)
            price = 0.0
            if not hist.empty:
                mask = hist.index <= date
                if mask.any(): 
                    price = hist.loc[mask, 'Close'].iloc[-1]
                else: 
                    price = hist['Close'].iloc[0]
            
            data_rows.append({
                'Periodo': year_str,
                'Receita': row[rev_col],
                'Lucro': row[inc_col],
                'Cotação': price
            })

        # 3. Dados TTM
        ttm_rev = 0.0
        ttm_inc = 0.0
        has_ttm = False
        
        if not quarterly.empty:
            q_rev_col = find_col(quarterly, rev_candidates)
            q_inc_col = find_col(quarterly, inc_candidates)
            if q_rev_col and q_inc_col:
                last_4_quarters = quarterly.tail(4)
                if len(last_4_quarters) == 4:
                    ttm_rev = last_4_quarters[q_rev_col].sum()
                    ttm_inc = last_4_quarters[q_inc_col].sum()
                    has_ttm = True
        
        curr_price = 0.0
        if not hist.empty: curr_price = hist['Close'].iloc[-1]
            
        if has_ttm:
            data_rows.append({
                'Periodo': 'Últimos 12m',
                'Receita': ttm_rev,
                'Lucro': ttm_inc,
                'Cotação': curr_price
            })
        else:
            if len(data_rows) > 0:
                last_valid = data_rows[-1]
                data_rows.append({
                    'Periodo': 'Últimos 12m',
                    'Receita': last_valid['Receita'], 
                    'Lucro': last_valid['Lucro'],
                    'Cotação': curr_price
                })

        df_final = pd.DataFrame(data_rows)
        if df_final.empty: return None

        # Formatação Texto
        df_final['Receita_Texto'] = df_final['Receita'].apply(format_short_decimal)
        df_final['Lucro_Texto'] = df_final['Lucro'].apply(format_short_decimal)
        
        return df_final
    except Exception as e:
        return None

@st.cache_data(ttl=3600*6)
def get_latest_dividends(ticker_list):
    divs_data = []
    for ticker in ticker_list[:10]:
        try:
            stock = yf.Ticker(ticker + ".SA")
            d = stock.dividends
            if not d.empty: divs_data.append({'Ativo': ticker, 'Valor': d.iloc[-1], 'Data': d.index[-1]})
        except: continue
    if divs_data:
        df = pd.DataFrame(divs_data)
        df['Data'] = pd.to_datetime(df['Data']).dt.tz_localize(None)
        return df.sort_values('Data', ascending=False).head(5)
    return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_market_news():
    feeds = {'Money Times': 'https://www.moneytimes.com.br/feed/', 'InfoMoney': 'https://www.infomoney.com.br/feed/'}
    news_items = []
    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                news_items.append({'title': entry.title, 'link': entry.link, 'source': source})
        except: continue
    return news_items[:6]

def show_chart_widget(symbol_tv, interval="D"):
    html_code = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%", "height": 500, "symbol": "{symbol_tv}", "interval": "{interval}", 
        "timezone": "America/Sao_Paulo", "theme": "light", "style": "1", "locale": "br",
        "toolbar_bg": "#f1f3f6", "enable_publishing": false, "allow_symbol_change": true,
        "studies": ["MASimple@tv-basicstudies"], "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(html_code, height=500)

# ==========================================
# INTERFACE
# ==========================================

# Banner Superior
components.html("""
    <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
        <script src="https://pl28325401.effectivegatecpm.com/1a/83/79/1a8379a4a8ddb94a327a5797257a9f02.js"></script>
    </div>
""", height=90)

st.title("🇧🇷 Ranking de Ações Baratas e Rentáveis - B3")
mes_txt, ano_int = get_current_data()
st.markdown(f"**Referência:** {mes_txt}/{ano_int}")
st.warning("⚠️ Utilize os dados apenas para estudo.")

with st.spinner('Analisando mercado...'):
    df_raw = get_ranking_data()
    # Preparar DataFrame de Top 200 Liquidez > 100k para uso geral
    df_liquid_200 = df_raw[df_raw['liq2m'] > 100000].sort_values(by='liq2m', ascending=False).head(200)
    
    df_best = apply_best_filters(df_raw)
    df_warning = get_risk_table(df_raw)
    df_magic = get_magic_formula_data(df_raw)
    df_graham = get_graham_data(df_raw)
    df_earnings_gap = get_earnings_gap_data(df_raw)

tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking Fundamentalista", "✨ Fórmula Mágica", "💎 Graham", "💰 Lucro vs Preço"])

with tab1:
    st.markdown("Oportunidades Fundamentalistas.")
    if not st.session_state.app_liberado:
        show_lock_screen("tab1") 
    else:
        if not df_best.empty:
            st.subheader("🏆 Melhores Ações")
            cols_view = ['Ativo', 'Preço', 'EV/EBIT', 'P/L', 'ROE', 'DY', 'Margem Líq.']
            styler = df_best[cols_view].style.map(lambda x: 'background-color: #f2f2f2; color: black;', subset=['Preço','P/L']).format({
                "Preço": "R$ {:.2f}", 
                "P/L": "{:.1f}", 
                "DY": "{:.1f}", 
                "ROE": "{:.1f}", 
                "Margem Líq.": "{:.1f}",
                "EV/EBIT": "{:.1f}"
            })
            st.dataframe(styler, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("⚠️ Empresas em Risco")
        
        if not df_warning.empty:
            styler_risk = df_warning.style.map(lambda v: 'color: red;' if '-' in str(v) else '', subset=['Queda Lucro']).format({
                "Preço": "R$ {:.2f}",
                "Alavancagem": "{:.0f}"
            })
            st.dataframe(styler_risk, use_container_width=True, hide_index=True)
        else: st.info("Sem alertas hoje.")

        st.divider()
        st.subheader("📈 Gráfico Cotação vs Lucro/Receita (4 Anos + TTM)")
        
        # --- MUDANÇA: Lista ampliada para as 200 mais líquidas ---
        opts = df_liquid_200['Ativo'].tolist()
        # Garante que LREN3 está na lista se existir no DF original, caso a liquidez varie
        if 'LREN3' not in opts and not df_raw[df_raw['Ativo'] == 'LREN3'].empty:
            opts.append('LREN3')
            
        idx = opts.index('LREN3') if 'LREN3' in opts else 0
        with st.expander("🔎 Selecionar Ação (Top 200 Líquidas)", expanded=st.session_state.expander_open):
            selected = st.selectbox("Ativo:", opts, index=idx, on_change=close_expander)
        
        if selected:
            df_chart = get_chart_data(selected)
            if df_chart is not None:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_chart['Periodo'], y=df_chart['Receita'], name="Receita", 
                    marker_color='#A9A9A9', opacity=0.8, yaxis='y1',
                    text=df_chart['Receita_Texto'], textposition='outside', hovertemplate='Receita: %{text}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=df_chart['Periodo'], y=df_chart['Cotação'], name="Preço (R$)", 
                    line=dict(color='#0000FF', width=3, shape='spline', smoothing=1.3), mode='lines+markers', yaxis='y2',
                    hovertemplate='Preço: R$ %{y:.2f}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=df_chart['Periodo'], y=df_chart['Lucro'], name="Lucro Líq.", 
                    line=dict(color='#008000', width=3, dash='dot', shape='spline', smoothing=1.3), mode='lines+markers', yaxis='y3',
                    hovertemplate='Lucro: %{text}<extra></extra>', text=df_chart['Lucro_Texto']
                ))
                fig.update_layout(
                    title=f"{selected}: Receita vs Lucro vs Preço (Curto Prazo)",
                    xaxis=dict(title="Período", type='category'),
                    yaxis=dict(title="Receita", side="left", showgrid=False, color="#808080", showticklabels=False),
                    yaxis2=dict(title="Preço (R$)", side="right", overlaying="y", showgrid=True, color="#0000FF"),
                    yaxis3=dict(title="Lucro", anchor="x", overlaying="y", side="right", showgrid=False, showticklabels=False, color="#008000"),
                    hovermode="x unified", height=500, legend=dict(orientation="h", y=1.1, x=0), barmode='overlay', margin=dict(t=80) 
                )
                st.plotly_chart(fig, use_container_width=True)
            else: st.warning(f"Dados históricos insuficientes para montar o gráfico de {selected}.")
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📰 Notícias")
            for n in get_market_news(): st.markdown(f"[{n['title']}]({n['link']})")
        with c2:
            st.subheader("💰 Dividendos")
            df_d = get_latest_dividends(df_best['Ativo'].tolist())
            if not df_d.empty: 
                df_d['Valor'] = df_d['Valor'].apply(lambda x: f"R$ {x:.4f}")
                df_d['Data'] = df_d['Data'].dt.strftime('%d/%m/%Y')
                st.dataframe(df_d, hide_index=True)

    show_affiliate_banners()

with tab2:
    st.subheader("✨ Fórmula Mágica (Top 40)")
    st.markdown("""
    **Metodologia:**
    1. Universo das 400 ações mais líquidas do Ibovespa/B3 (Liquidez > 100k).
    2. Rank 1: **Earning Yield** (EBIT / Valor da Empresa) -> Do Maior para o Menor.
    3. Rank 2: **ROIC** -> Do Maior para o Menor.
    4. **Score Final** = Rank EY + Rank ROIC. (Menor pontuação indica melhor combinação de qualidade e preço).
    """)
    
    if not st.session_state.app_liberado:
        show_lock_screen("tab2") 
    else:
        if not df_magic.empty:
            st.success(f"Top 40 Empresas selecionadas de um universo de 400 papéis.")
            
            def color_magic_score(val):
                if val < 50: color = "#d4edda" 
                elif val < 150: color = "#fff3cd" 
                else: color = "#f8d7da" 
                return f'background-color: {color}; color: black;'

            styler_magic = df_magic.style.format({
                "Preço": "R$ {:.2f}", 
                "Score Mágico": "{:.0f}",
                "Rank EY": "{:.0f}", 
                "Rank ROIC": "{:.0f}",
                "EV/EBIT": "{:.2f}"
            }).map(color_magic_score, subset=['Score Mágico'])
            
            st.dataframe(styler_magic, use_container_width=True, hide_index=True)
        else:
            st.warning("Não foi possível calcular o ranking hoje.")

    show_affiliate_banners()

with tab3:
    st.subheader("💎 Valuation: Método Graham (Adaptado)")
    st.markdown("""
    **Fórmula de Valor Intrínseco:** $V = \\frac{LPA \\times (8.5 + 2g) \\times 4.4}{Y}$
    
    *   **LPA:** Lucro por Ação.
    *   **g:** Crescimento projetado (Considerado **10%** fixo neste modelo).
    *   **Y:** Rendimento de Títulos Corporativos AAA (IPCA + Spread). Considerado **6.0%** (Real Yield Atual).
    *   **Classificação:** Baseada no Ratio (Preço / Valor Intrínseco). Quanto menor, mais descontada.
    """)
    
    if not st.session_state.app_liberado:
        show_lock_screen("tab3") 
    else:
        if not df_graham.empty:
            def color_graham(val):
                color = "white"
                if val == "Barata": color = "#d4edda" 
                elif val == "Cara": color = "#f8d7da" 
                return f'background-color: {color}; color: black;'

            styler_graham = df_graham.style.format({
                "Preço Atual": "R$ {:.2f}",
                "Preço Justo (Graham)": "R$ {:.2f}",
                "Ratio (P/V)": "{:.2f}",
                "LPA": "R$ {:.2f}"
            }).map(color_graham, subset=['Status'])
            
            st.dataframe(styler_graham, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhuma ação atende aos critérios.")

    show_affiliate_banners()

with tab4:
    st.subheader("💰 Lucro Real vs Preço (Earnings Yield)")
    st.markdown("""
    Esta lista mostra as empresas (das 200 mais líquidas) onde o **Lucro por Ação (LPA)** é mais alto em relação ao **Preço da Ação**.
    
    *   **Conceito:** Uma "grande distância" do lucro para o preço indica que a empresa gera muito caixa para cada real investido na ação (Earnings Yield alto).
    *   **Interpretação:** Quanto maior o *Earnings Yield*, teoricamente "mais barata" a empresa está em relação ao seu lucro atual.
    """)

    if not st.session_state.app_liberado:
        show_lock_screen("tab4")
    else:
        if not df_earnings_gap.empty:
            
            # Formatação Visual
            def color_yield(val):
                # Se Yield > 15% (excelente retorno sobre preço)
                if val > 15: color = "#d4edda" 
                # Se Yield > 10%
                elif val > 10: color = "#e2e3e5"
                else: color = "white"
                return f'background-color: {color}; color: black;'

            styler_gap = df_earnings_gap.style.format({
                "Preço": "R$ {:.2f}",
                "LPA (Lucro/Ação)": "R$ {:.2f}",
                "P/L": "{:.2f}",
                "Earnings Yield (%)": "{:.2f}%"
            }).map(color_yield, subset=['Earnings Yield (%)'])
            
            st.dataframe(styler_gap, use_container_width=True, hide_index=True)
        else:
            st.info("Não há dados suficientes para este cálculo.")

    show_affiliate_banners()
