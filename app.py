import streamlit as st
import streamlit.components.v1 as components 

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Monitor B3 | Ranking & Scanners",
    layout="wide",
    page_icon="🇧🇷"
)

# --- IMPORTS GERAIS ---
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

# Tenta importar fundamentus
try:
    import fundamentus
except ImportError:
    st.error("Biblioteca 'fundamentus' não encontrada. Adicione ao requirements.txt")
    st.stop()

# --- CSS Global ---
st.markdown("""
    <style>
    /* Cabeçalhos à direita */
    [data-testid="stDataFrame"] table tr th { text-align: right !important; }
    /* Células à esquerda */
    [data-testid="stDataFrame"] table tr td { text-align: left !important; }
    /* Ajuste visual das abas */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 5px 5px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÃO DO CONTADOR DE VISITANTES ---
def update_visitor_counter():
    file_path = "visitor_counter.json"
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        if hasattr(st, "query_params"):
            current_params = st.query_params
            visitor_id = current_params.get("visitor_id", None)
        else:
            current_params = st.experimental_get_query_params()
            visitor_id = current_params.get("visitor_id", [None])[0]
    except:
        visitor_id = None

    if not visitor_id:
        visitor_id = str(uuid.uuid4())
        if hasattr(st, "query_params"):
            try: st.query_params["visitor_id"] = visitor_id
            except: pass
        
    data = {"total_visits": 0, "daily_visits": {}}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f: data = json.load(f)
        except: pass 

    if today not in data["daily_visits"]: data["daily_visits"][today] = []

    if visitor_id not in data["daily_visits"][today]:
        data["daily_visits"][today].append(visitor_id)
        data["total_visits"] += 1
        try:
            with open(file_path, "w") as f: json.dump(data, f)
        except: pass
            
    return data["total_visits"]

try: total_visitantes = update_visitor_counter()
except: total_visitantes = 0 

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📊 Estatísticas")
    st.metric(label="Visitantes Únicos", value=total_visitantes)
    st.divider()
    st.caption("Desenvolvido com Streamlit")

# --- Estado ---
if 'expander_open' not in st.session_state: st.session_state.expander_open = True
if 'tv_symbol' not in st.session_state: st.session_state.tv_symbol = "BMFBOVESPA:LREN3"

def close_expander(): st.session_state.expander_open = False

# --- Auxiliares ---
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

def format_short_number(val):
    if pd.isna(val) or val == 0: return ""
    abs_val = abs(val)
    if abs_val >= 1e9: return f"{val/1e9:.1f}B"
    elif abs_val >= 1e6: return f"{val/1e6:.0f}M"
    elif abs_val >= 1e3: return f"{val/1e3:.0f}K"
    return f"{val:.0f}"

def get_current_data():
    now = datetime.datetime.now()
    return now.strftime("%B"), now.year

# --- DADOS FUNDAMENTUS (ABA 1) ---
@st.cache_data(ttl=3600*6)
def get_ranking_data():
    try:
        df = fundamentus.get_resultado()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'papel'}, inplace=True)
        cols = ['pl', 'roe', 'dy', 'evebit', 'cotacao', 'liq2m', 'mrgliq', 'divbpatr', 'c5y']
        for col in cols:
            if col in df.columns: df[col] = df[col].apply(clean_fundamentus_col)
            else: df[col] = 0.0
        return df
    except: return pd.DataFrame()

def apply_best_filters(df):
    if df.empty: return df
    filtro = (
        (df['roe'] > 0.05) & (df['pl'] < 15) & (df['pl'] > 0) & 
        (df['evebit'] > 0) & (df['evebit'] < 10) &
        (df['dy'] > 0.04) & (df['mrgliq'] > 0.05) & (df['liq2m'] > 200000)
    )
    df_filtered = df[filtro].copy()
    df_filtered['dy'] = df_filtered['dy'] * 100
    df_filtered['mrgliq'] = df_filtered['mrgliq'] * 100
    df_filtered['roe'] = df_filtered['roe'] * 100
    df_filtered.rename(columns={
        'papel': 'Ativo', 'cotacao': 'Preço', 'pl': 'P/L', 
        'evebit': 'EV/EBIT', 'dy': 'DY', 'roe': 'ROE', 'mrgliq': 'Margem Líq.'
    }, inplace=True)
    return df_filtered.sort_values(by=['P/L', 'Margem Líq.'], ascending=[True, False]).reset_index(drop=True)

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
        lucro_queda_str = "N/D"
        val_queda = 0.0
        try:
            stock = yf.Ticker(ticker + ".SA")
            fin = stock.financials
            if not fin.empty:
                for name in ['Net Income', 'Net Income Common', 'Lucro Liquido']:
                    if name in fin.index:
                        inc_row = fin.loc[name]
                        if len(inc_row) >= 2:
                            curr, prev = inc_row.iloc[0], inc_row.iloc[1]
                            if curr < prev:
                                pct = ((curr - prev) / abs(prev)) * 100
                                val_queda = pct 
                                lucro_queda_str = f"{pct:.1f}%"
                            else: lucro_queda_str = "Subiu/Estável"
                        break
        except: lucro_queda_str = "Erro"

        if val_queda < 0 or ticker in lista_rj:
            risk_data.append({
                'Ativo': ticker, 'Preço': row['cotacao'], 
                'Alavancagem (Dív/Patr)': row['divbpatr'], 
                'Queda Lucro (Ano)': lucro_queda_str, 'Situação': status
            })
    return pd.DataFrame(risk_data)

# --- DADOS GRÁFICO (ABA 1 - CORRIGIDA) ---
@st.cache_data(ttl=3600*24)
def get_chart_data(ticker):
    try:
        stock = yf.Ticker(ticker + ".SA")
        financials = stock.financials.T
        quarterly = stock.quarterly_financials.T
        hist = stock.history(period="5y")
        
        if not financials.empty: 
            financials.index = pd.to_datetime(financials.index).tz_localize(None)
            financials = financials.sort_index()
        if not quarterly.empty: 
            quarterly.index = pd.to_datetime(quarterly.index).tz_localize(None)
            quarterly = quarterly.sort_index()
        if not hist.empty: 
            hist.index = pd.to_datetime(hist.index).tz_localize(None)

        def find_col(df, candidates):
            for cand in candidates:
                for col in df.columns:
                    if cand.lower() in col.lower(): return col
            return None

        rev_col = find_col(financials, ['Total Revenue', 'Revenue', 'Receita'])
        inc_col = find_col(financials, ['Net Income', 'Lucro'])
        if not rev_col or not inc_col: return None

        data_rows = []
        last_3_years = financials.tail(3)
        for date, row in last_3_years.iterrows():
            price = 0.0
            if not hist.empty:
                # Tenta pegar preço próximo da data de publicação do balanço ou fim do ano
                mask = hist.index <= date
                if mask.any(): price = hist.loc[mask, 'Close'].iloc[-1]
            data_rows.append({'Periodo': str(date.year), 'Receita': row[rev_col], 'Lucro': row[inc_col], 'Cotação': price})
            
        ttm_rev, ttm_inc, has_ttm = 0, 0, False
        if not quarterly.empty:
            q_limit = min(4, len(quarterly))
            last_q = quarterly.tail(q_limit)
            rev_q = find_col(quarterly, ['Total Revenue', 'Revenue'])
            inc_q = find_col(quarterly, ['Net Income', 'Lucro'])
            if rev_q and inc_q:
                ttm_rev = last_q[rev_q].sum()
                ttm_inc = last_q[inc_q].sum()
                has_ttm = True
        
        if has_ttm:
            curr_price = hist['Close'].iloc[-1] if not hist.empty else 0.0
            data_rows.append({'Periodo': 'TTM (12m)', 'Receita': ttm_rev, 'Lucro': ttm_inc, 'Cotação': curr_price})
        
        df_final = pd.DataFrame(data_rows)
        # Cria textos abreviados para o gráfico
        df_final['Receita_Texto'] = df_final['Receita'].apply(format_short_number)
        df_final['Lucro_Texto'] = df_final['Lucro'].apply(format_short_number)
        return df_final
    except: return None

# --- OUTROS (DIVIDENDOS, NOTICIAS) ---
@st.cache_data(ttl=3600*6)
def get_latest_dividends(ticker_list):
    divs_data = []
    for ticker in ticker_list[:10]:
        try:
            d = yf.Ticker(ticker + ".SA").dividends
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
                news_items.append({'title': entry.title, 'link': entry.link, 'date_str': "Recente", 'source': source})
        except: continue
    return news_items[:6]

# --- SCANNER BOLLINGER (ABA 2 - CORRIGIDO TOP 100) ---
@st.cache_data(ttl=600)
def scan_bollinger_weekly(df_top_liq):
    # 1. Obter TOP 100 Ações mais líquidas do DataFrame Fundamentalista
    if df_top_liq.empty:
        return pd.DataFrame()
    
    # Ordena por liquidez e pega os 100 primeiros símbolos
    top_100_tickers = df_top_liq.sort_values(by='liq2m', ascending=False).head(100)['papel'].tolist()
    
    # Adiciona sufixo .SA
    tickers_sa = [t + ".SA" for t in top_100_tickers]
    
    candidates = []
    
    try:
        # 2. Baixar dados semanais com auto_adjust=False (Preço Nominal)
        # Ajustado para period='1y' para ser mais rápido, já que só precisamos das últimas 20 semanas
        data = yf.download(tickers_sa, period="1y", interval="1wk", group_by='ticker', auto_adjust=False, progress=False, threads=True)
        
        for t in tickers_sa:
            try:
                # Tratamento do MultiIndex do yfinance
                if len(tickers_sa) > 1:
                    df_t = data[t].copy() if t in data else pd.DataFrame()
                else:
                    df_t = data.copy()
                
                # Limpeza básica
                if df_t.empty: continue
                # Garante que Close e Low não são nulos (pode haver semanas feriado/futuro)
                df_t.dropna(subset=['Close', 'Low'], inplace=True)
                
                # Precisa de 20 semanas para a banda
                if len(df_t) < 20: continue 

                # Cálculo Bandas Bollinger (20, 2)
                df_t['SMA20'] = df_t['Close'].rolling(window=20).mean()
                df_t['STD20'] = df_t['Close'].rolling(window=20).std()
                df_t['Lower'] = df_t['SMA20'] - (2.0 * df_t['STD20'])
                
                # Remove período de aquecimento (NaNs)
                df_t.dropna(subset=['Lower'], inplace=True)
                
                if df_t.empty: continue

                # Pega a vela ATUAL (Última da série)
                curr = df_t.iloc[-1]
                
                # === LÓGICA DE TOQUE NA BANDA INFERIOR ===
                # Preço Mínimo (Low) <= Banda Inferior (Lower)
                if curr['Low'] <= curr['Lower']:
                    clean_ticker = t.replace(".SA", "")
                    
                    status_txt = "Furou (Fech < Banda)" if curr['Close'] < curr['Lower'] else "Tocou (Pavio)"
                    
                    candidates.append({
                        'Ativo': clean_ticker,
                        'Preço': curr['Close'],
                        'Mínima Sem.': curr['Low'],
                        'Banda Inf': curr['Lower'],
                        'Status': status_txt,
                        'TV_Symbol': f"BMFBOVESPA:{clean_ticker}"
                    })
            except: continue
        
        return pd.DataFrame(candidates)
    except: return pd.DataFrame()

# --- SCANNER ROC EMA (ABA 3) ---
@st.cache_data(ttl=3600*4)
def scan_roc_weekly(df_top_liq):
    if df_top_liq.empty: return pd.DataFrame()
    top_tickers = df_top_liq.sort_values(by='liq2m', ascending=False).head(90)['papel'].tolist()
    tickers_sa = [t + ".SA" for t in top_tickers]
    candidates = []
    try:
        data = yf.download(tickers_sa, period="5y", interval="1wk", group_by='ticker', progress=False, threads=True)
        for t in tickers_sa:
            clean_ticker = t.replace(".SA", "")
            try:
                df_t = data[t].copy() if t in data else pd.DataFrame()
                if df_t.empty: continue
                df_t.dropna(subset=['Close'], inplace=True)
                if len(df_t) < 310: continue
                
                df_t['EMA17'] = df_t['Close'].ewm(span=17).mean()
                df_t['EMA34'] = df_t['Close'].ewm(span=34).mean()
                df_t['EMA72'] = df_t['Close'].ewm(span=72).mean()
                df_t['EMA305'] = df_t['Close'].ewm(span=305).mean()
                
                curr = df_t.iloc[-1]
                
                # Lógica simplificada Caiu Comprou
                is_uptrend = (curr['Close'] > curr['EMA72']) and (curr['Close'] > curr['EMA305'])
                is_pullback = (curr['Close'] < curr['EMA17'])
                
                prob = ""
                if is_uptrend and is_pullback:
                    if curr['Close'] > curr['EMA34']: prob = "Alta (Caiu Comprou)"
                    else: prob = "Média (Pullback Profundo)"
                
                if prob:
                    roc17 = ((curr['Close'] - curr['EMA17']) / curr['EMA17']) * 100
                    candidates.append({'Ativo': clean_ticker, 'Preço': curr['Close'], 'Probabilidade': prob, 'ROC17 %': roc17, 'TV_Symbol': f"BMFBOVESPA:{clean_ticker}"})
            except: continue
        return pd.DataFrame(candidates)
    except: return pd.DataFrame()

# --- WIDGET CHART TRADINGVIEW ---
def show_chart_widget(symbol_tv, interval="D"):
    html_code = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%", "height": 500, "symbol": "{symbol_tv}", "interval": "{interval}", 
        "timezone": "America/Sao_Paulo", "theme": "light", "style": "1", "locale": "br", 
        "toolbar_bg": "#f1f3f6", "enable_publishing": false, "allow_symbol_change": true, 
        "studies": ["MASimple@tv-basicstudies", "BB@tv-basicstudies"], 
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(html_code, height=500)

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================

components.html("""<div style="display: flex; justify-content: center; width: 100%;"><script src="https://pl28325401.effectivegatecpm.com/1a/83/79/1a8379a4a8ddb94a327a5797257a9f02.js"></script></div>""", height=90)

st.title("🇧🇷 Ranking de Ações Baratas e Rentáveis - B3")
mes_txt, ano_int = get_current_data()
st.markdown(f"**Referência:** {mes_txt}/{ano_int}")
st.warning("⚠️ **AVISO IMPORTANTE:** Utilize os dados apenas para estudo.")

# CARREGAMENTO
with st.spinner('Processando dados do mercado...'):
    df_raw = get_ranking_data()
    df_best = apply_best_filters(df_raw)
    df_warning = get_risk_table(df_raw)
    # Passamos df_raw para o scanner selecionar as Top 100
    df_scan_bb = scan_bollinger_weekly(df_raw) 
    df_scan_roc = scan_roc_weekly(df_raw)

tab1, tab2, tab3 = st.tabs(["🏆 Ranking Fundamentalista", "📉 Setup BB Semanal (Lower Band)", "🚀 Setup ROC (Caiu Comprou)"])

# === ABA 1: FUNDAMENTALISTA (VISUAL CORRIGIDO) ===
with tab1:
    st.markdown("""<div style="text-align: justify; margin-bottom: 20px;">Este <b>Screener Fundamentalista</b> filtra automaticamente as melhores oportunidades.</div>""", unsafe_allow_html=True)

    if not df_best.empty:
        st.subheader("🏆 Melhores Ações (Oportunidades)")
        st.caption("Filtro: P/L Baixo, Alta Rentabilidade e Dividendos.")
        
        cols_view = ['Ativo', 'Preço', 'EV/EBIT', 'P/L', 'ROE', 'DY', 'Margem Líq.']
        styler = df_best[cols_view].style.map(lambda x: 'background-color: #f2f2f2; color: black;', subset=['Preço', 'P/L', 'DY']).format({
            "Preço": "R$ {:.2f}", "EV/EBIT": "{:.2f}", "P/L": "{:.2f}", "ROE": "{:.2f}", "DY": "{:.2f}", "Margem Líq.": "{:.2f}"
        })
        st.dataframe(styler, use_container_width=True, hide_index=True)

    st.divider()
    col_ad1, col_ad2 = st.columns(2)
    with col_ad1:
        st.markdown("""<div style="background-color: #fffbe6; padding: 15px; border-radius: 10px; color: #333;"><h4>✈️ Nomad: Taxa Zero</h4><p>Código: <b>Y39FP3XF8I</b></p></div>""", unsafe_allow_html=True)
    with col_ad2:
        st.markdown("""<div style="background-color: #eaf6ff; padding: 15px; border-radius: 10px; color: #333;"><h4>🤝 Mercado Pago: R$ 30 OFF</h4><p>Desconto no 1º uso.</p></div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("⚠️ Atenção! Empresas em Risco")
    if not df_warning.empty:
        st.dataframe(df_warning.style.map(lambda v: 'color: red; font-weight: bold;' if isinstance(v,str) and '-' in v else '', subset=['Queda Lucro (Ano)']).format({"Preço": "R$ {:.2f}", "Alavancagem (Dív/Patr)": "{:.2f}"}), use_container_width=True, hide_index=True)
    
    st.divider()
    st.subheader("📈 Análise Visual (Proporcional)")
    opts = df_best['Ativo'].tolist()
    idx_default = 0
    if 'LREN3' in opts: idx_default = opts.index('LREN3')
    
    with st.expander("🔎 Selecionar Ação", expanded=st.session_state.expander_open):
        sel = st.selectbox("Ativo:", opts, index=idx_default, on_change=close_expander)
        
    if sel:
        d_chart = get_chart_data(sel)
        if d_chart is not None:
            # Lógica Plotly com eixos duplos e valores abreviados
            fig = go.Figure()
            
            # Barras: Receita (Eixo Esquerdo)
            fig.add_trace(go.Bar(
                x=d_chart['Periodo'], 
                y=d_chart['Receita'], 
                name="Receita", 
                marker_color='silver',
                text=d_chart['Receita_Texto'], # Exibe 10B
                textposition='auto'
            ))
            
            # Linha: Lucro (Eixo Esquerdo - Compartilhado para mostrar Margem)
            fig.add_trace(go.Scatter(
                x=d_chart['Periodo'], 
                y=d_chart['Lucro'], 
                name="Lucro Líq.", 
                line=dict(color='green', width=3),
                mode='lines+markers+text',
                text=d_chart['Lucro_Texto'], # Exibe 500M
                textposition='top center'
            ))
            
            # Linha: Cotação (Eixo DIREITO)
            fig.add_trace(go.Scatter(
                x=d_chart['Periodo'], 
                y=d_chart['Cotação'], 
                name="Cotação", 
                line=dict(color='blue', width=3, dash='dot'),
                yaxis='y2',
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title=f"{sel}: Receita vs Lucro (Esq) e Preço (Dir)",
                yaxis=dict(title="Financeiro (R$)", showgrid=False),
                yaxis2=dict(title="Cotação (R$)", overlaying='y', side='right', showgrid=False),
                legend=dict(orientation="h", y=1.1),
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Dados indisponíveis para {sel}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📰 Notícias")
        news = get_market_news()
        for n in news: st.markdown(f"**[{n['title']}]({n['link']})**")
    with c2:
        st.subheader("💰 Dividendos")
        df_divs = get_latest_dividends(df_best['Ativo'].tolist())
        if not df_divs.empty: st.dataframe(df_divs.style.format({"Valor": "R$ {:.4f}", "Data": "{:%d/%m/%Y}"}), hide_index=True)

# === ABA 2: BOLLINGER SEMANAL (CORRIGIDA TOP 100) ===
with tab2:
    st.subheader("📉 Bandas de Bollinger Semanal (Mínima tocou Banda Inf.)")
    st.markdown("**Universo:** Top 100 Ações mais líquidas da B3.")
    st.warning("⚠️ **Filtro:** Ação listada se a MÍNIMA (Low) da semana atual for menor ou igual à Banda Inferior.")
    
    col_list, col_chart = st.columns([1, 2])
    with col_list:
        if not df_scan_bb.empty:
            st.write(f"**{len(df_scan_bb)} Oportunidades:**")
            
            def color_st(v): return f"background-color: {'#ffcccc' if 'Furou' in v else '#fffacd'}; color: black"
            
            evt = st.dataframe(
                df_scan_bb[['Ativo', 'Preço', 'Mínima Sem.', 'Banda Inf', 'Status']].style.format({
                    "Preço": "{:.2f}", "Mínima Sem.": "{:.2f}", "Banda Inf": "{:.2f}"
                }).map(color_st, subset=['Status']),
                use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
            )
            if len(evt.selection.rows) > 0:
                st.session_state.tv_symbol = df_scan_bb.iloc[evt.selection.rows[0]]['TV_Symbol']
        else:
            st.info("Nenhuma das Top 100 ações está tocando a banda inferior nesta semana.")
            
    with col_chart:
        nm = st.session_state.tv_symbol.split(":")[-1]
        st.markdown(f"#### {nm} (Semanal)")
        show_chart_widget(st.session_state.tv_symbol, interval="W")

# === ABA 3: ROC ===
with tab3:
    st.subheader("🚀 Setup ROC")
    c1, c2 = st.columns([1, 2])
    with c1:
        if not df_scan_roc.empty:
            evt_roc = st.dataframe(df_scan_roc[['Ativo', 'Preço', 'Probabilidade']].style.format({"Preço": "R$ {:.2f}"}), use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
            if len(evt_roc.selection.rows) > 0:
                st.session_state.tv_symbol = df_scan_roc.iloc[evt_roc.selection.rows[0]]['TV_Symbol']
        else: st.info("Sem ativos hoje.")
    with c2:
        nm = st.session_state.tv_symbol.split(":")[-1]
        st.markdown(f"#### {nm} (Diário)")
        show_chart_widget(st.session_state.tv_symbol, interval="D")
