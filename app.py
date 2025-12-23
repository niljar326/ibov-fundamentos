import streamlit as st
import streamlit.components.v1 as components 

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Melhores Ações Ibovespa 2025 | Ranking Fundamentalista e Dividendos",
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
    except: visitor_id = None

    if not visitor_id:
        visitor_id = str(uuid.uuid4())
        if hasattr(st, "query_params"): st.query_params["visitor_id"] = visitor_id
        
    data = {"total_visits": 0, "daily_visits": {}}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f: data = json.load(f)
        except: pass 

    if today not in data["daily_visits"]: data["daily_visits"][today] = []
    if visitor_id not in data["daily_visits"][today]:
        data["daily_visits"][today].append(visitor_id)
        data["total_visits"] += 1
        with open(file_path, "w") as f: json.dump(data, f)
    return data["total_visits"]

try: total_visitantes = update_visitor_counter()
except: total_visitantes = 0 

with st.sidebar:
    st.header("📊 Estatísticas")
    st.metric(label="Visitantes Únicos", value=total_visitantes)
    st.divider()
    st.caption("Desenvolvido com Streamlit")

# --- Estado ---
if 'expander_open' not in st.session_state: st.session_state.expander_open = True
if 'selected_ticker_tab2' not in st.session_state: st.session_state.selected_ticker_tab2 = "BMFBOVESPA:LREN3"

def close_expander(): st.session_state.expander_open = False

# --- Auxiliares ---
def clean_fundamentus_col(x):
    if pd.isna(x) or x == '': return 0.0
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        x = x.strip().replace('%', '').replace('.', '').replace(',', '.')
        try: return float(x) / 100 if x.endswith('%') else float(x)
        except: return 0.0
    return 0.0

def format_short_number(val):
    if pd.isna(val) or val == 0: return ""
    if abs(val) >= 1e9: return f"{val/1e9:.1f}B"
    elif abs(val) >= 1e6: return f"{val/1e6:.0f}M"
    return f"{val:.0f}"

def get_current_data():
    now = datetime.datetime.now()
    return now.strftime("%B"), now.year

# --- DADOS FUNDAMENTALISTAS ---
@st.cache_data(ttl=3600*6)
def get_ranking_data():
    try:
        df = fundamentus.get_resultado()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'papel'}, inplace=True)
        cols = ['pl', 'roe', 'dy', 'evebit', 'cotacao', 'liq2m', 'mrgliq', 'divbpatr']
        for col in cols:
            if col in df.columns: df[col] = df[col].apply(clean_fundamentus_col)
            else: df[col] = 0.0
        return df
    except: return pd.DataFrame()

def apply_best_filters(df):
    if df.empty: return df
    filtro = ((df['roe'] > 0.05) & (df['pl'] < 15) & (df['pl'] > 0) & (df['dy'] > 0.04) & (df['liq2m'] > 200000))
    df_filtered = df[filtro].copy()
    df_filtered[['dy', 'mrgliq', 'roe']] = df_filtered[['dy', 'mrgliq', 'roe']] * 100
    df_filtered.rename(columns={'papel': 'Ativo', 'cotacao': 'Preço', 'pl': 'P/L', 'evebit': 'EV/EBIT', 'dy': 'DY', 'roe': 'ROE', 'mrgliq': 'Margem Líq.'}, inplace=True)
    return df_filtered.sort_values(by=['P/L', 'Margem Líq.'], ascending=[True, False]).reset_index(drop=True)

# --- LÓGICA DA TABELA DE RISCO ---
@st.cache_data(ttl=3600*12)
def get_risk_table(df_original):
    if df_original.empty: return pd.DataFrame()
    lista_rj = ['OIBR3', 'OIBR4', 'AMER3', 'GOLL4', 'AZUL4', 'RCSL3', 'RCSL4']
    mask = (df_original['divbpatr'] > 3.0) | (df_original['papel'].isin(lista_rj))
    df_risk = df_original[mask].copy()
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
                inc_row = None
                possible_names = ['Net Income', 'Net Income Common', 'Net Income Continuous']
                for name in possible_names:
                    if name in fin.index:
                        inc_row = fin.loc[name]
                        break
                if inc_row is not None and len(inc_row) >= 2:
                    curr, prev = inc_row.iloc[0], inc_row.iloc[1]
                    if curr < prev:
                        val_queda = ((curr - prev) / abs(prev)) * 100
                        lucro_queda_str = f"{val_queda:.1f}%"
                    else: lucro_queda_str = "Subiu"
        except: pass
        if val_queda < 0 or ticker in lista_rj:
            risk_data.append({'Ativo': ticker, 'Preço': row['cotacao'], 'Alavancagem (Dív/Patr)': row['divbpatr'], 'Queda Lucro (Ano)': lucro_queda_str, 'Situação': status})
    return pd.DataFrame(risk_data)

# --- CHART DATA (ABA 1) ---
@st.cache_data(ttl=3600*24)
def get_chart_data(ticker):
    try:
        stock = yf.Ticker(ticker + ".SA")
        financials = stock.financials.T
        hist = stock.history(period="5y")
        if financials.empty: return None
        data_rows = []
        # Simplificado para visual
        return pd.DataFrame({'Periodo': hist.index, 'Cotação': hist['Close'], 'Receita': 0, 'Lucro': 0, 'Receita_Texto': ''})
    except: return None

# --- DIVIDENDOS ---
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

# --- NOTÍCIAS ---
@st.cache_data(ttl=1800)
def get_market_news():
    feeds = {'Money Times': 'https://www.moneytimes.com.br/feed/', 'InfoMoney': 'https://www.infomoney.com.br/feed/', 'E-Investidor': 'https://einvestidor.estadao.com.br/feed/'}
    news_items = []
    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                try: dt_obj = datetime.datetime.fromtimestamp(mktime(entry.published_parsed)) - timedelta(hours=3)
                except: dt_obj = datetime.datetime.now()
                news_items.append({'title': entry.title, 'link': entry.link, 'date_obj': dt_obj, 'date_str': dt_obj.strftime("%d/%m %H:%M"), 'source': source})
        except: continue
    news_items.sort(key=lambda x: x['date_obj'], reverse=True)
    return news_items[:6]

# --- SCANNER BOLLINGER (SÓ BRASIL - SEMANAL - ATUAL E ANTERIOR) ---
@st.cache_data(ttl=600)
def scan_bollinger_br_weekly_lower():
    tickers_br = [
        "LREN3.SA", "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "WEGE3.SA", "PRIO3.SA", 
        "MGLU3.SA", "HAPV3.SA", "RDOR3.SA", "SUZB3.SA", "JBSS3.SA", "RAIZ4.SA", "GGBR4.SA", "CSAN3.SA",
        "VBBR3.SA", "B3SA3.SA", "BBSE3.SA", "CMIG4.SA", "ITSA4.SA", "BHIA3.SA", "GOLL4.SA", "AZUL4.SA", 
        "CVCB3.SA", "USIM5.SA", "CSNA3.SA", "EMBR3.SA", "CPLE6.SA", "RADL3.SA", "EQTL3.SA", "TOTS3.SA", 
        "RENT3.SA", "TIMS3.SA", "SBSP3.SA", "ELET3.SA", "ABEV3.SA", "ASAI3.SA", "CRFB3.SA", "MULT3.SA",
        "CYRE3.SA", "EZTC3.SA", "MRVE3.SA", "PETZ3.SA", "SOMA3.SA", "ALPA4.SA"
    ]
    
    candidates = []
    
    try:
        # Baixa dados SEMANAIS ('1wk') dos últimos 2 anos
        data = yf.download(tickers_br, period="2y", interval="1wk", group_by='ticker', progress=False, threads=True)
        
        for t in tickers_br:
            try:
                df_t = data[t].copy() if t in data else pd.DataFrame()
                if df_t.empty: continue
                
                # Limpeza básica
                df_t.dropna(subset=['Close'], inplace=True)
                if len(df_t) < 22: continue

                # Cálculo Bandas (20, 2)
                df_t['SMA20'] = df_t['Close'].rolling(window=20).mean()
                df_t['STD20'] = df_t['Close'].rolling(window=20).std()
                df_t['Lower'] = df_t['SMA20'] - (2.0 * df_t['STD20'])
                
                # Última semana (Atual) e Penúltima (Anterior)
                curr = df_t.iloc[-1]
                prev = df_t.iloc[-2]
                
                cond_curr = (curr['Low'] <= curr['Lower'])
                cond_prev = (prev['Low'] <= prev['Lower'])
                
                # Se tocou na atual OU na anterior
                if cond_curr or cond_prev:
                    clean_ticker = t.replace(".SA", "")
                    
                    # Define qual referência usar para cálculo de distância
                    ref_row = curr if cond_curr else prev
                    dist = ((curr['Close'] - ref_row['Lower']) / ref_row['Lower']) * 100
                    
                    quando = "Esta Semana" if cond_curr else "Semana Passada"
                    if cond_curr and cond_prev: quando = "Ambas"

                    candidates.append({
                        'Ativo': clean_ticker,
                        'Quando?': quando,
                        'Preço Atual': curr['Close'],
                        'Mínima Sem.': curr['Low'],
                        'Banda Inf': curr['Lower'],
                        'Distância Fech %': dist,
                        'TV_Symbol': f"BMFBOVESPA:{clean_ticker}"
                    })
            except: continue
        return pd.DataFrame(candidates)
    except: return pd.DataFrame()

# --- WIDGET CHART TRADINGVIEW ---
def show_chart_widget(symbol_tv):
    html_code = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%", "height": 500, "symbol": "{symbol_tv}", "interval": "W", 
        "timezone": "America/Sao_Paulo", "theme": "light", "style": "1", "locale": "br",
        "toolbar_bg": "#f1f3f6", "enable_publishing": false, "allow_symbol_change": true,
        "studies": ["BB@tv-basicstudies"], "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(html_code, height=500)

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================
st.title("🇧🇷 Ranking de Ações Baratas e Rentáveis - B3")
mes_txt, ano_int = get_current_data()
st.markdown(f"**Referência:** {mes_txt}/{ano_int}")

# --- AVISO LEGAL ---
st.warning("⚠️ **Aviso Importante:** As informações aqui apresentadas têm caráter meramente informativo e **não constituem recomendação de compra ou venda** de ativos. Ações listadas nos filtros (Setup BB ou Ranking) devem ser analisadas aprofundadamente antes de qualquer decisão de investimento.")

# 1. Carregamento dos Dados
with st.spinner('Processando dados do mercado...'):
    df_raw = get_ranking_data()
    df_best = apply_best_filters(df_raw)
    df_warning = get_risk_table(df_raw)
    df_scan_bb = scan_bollinger_br_weekly_lower() 

# --- SISTEMA DE ABAS ---
tab1, tab2 = st.tabs(["🏆 Ranking Fundamentalista", "📉 Setup BB Semanal (Lower Band)"])

# === ABA 1: CONTEÚDO ORIGINAL ===
with tab1:
    st.markdown("""
    <div style="text-align: justify; margin-bottom: 20px;">
    Este <b>Screener Fundamentalista</b> filtra automaticamente as melhores oportunidades. 
    Abaixo, você também encontra uma lista de <b>Alerta</b> para empresas em situações delicadas.
    </div>
    """, unsafe_allow_html=True)

    if not df_best.empty:
        st.subheader("🏆 Melhores Ações (Oportunidades)")
        st.caption("Filtro: P/L Baixo, Alta Rentabilidade e Dividendos.")
        
        cols_view = ['Ativo', 'Preço', 'EV/EBIT', 'P/L', 'ROE', 'DY', 'Margem Líq.']
        even_cols_subset = ['Preço', 'P/L', 'DY']
        styler = df_best[cols_view].style.map(
            lambda x: 'background-color: #f2f2f2; color: black;', 
            subset=even_cols_subset
        ).format({
            "Preço": "R$ {:.2f}", "EV/EBIT": "{:.2f}", "P/L": "{:.2f}",
            "ROE": "{:.2f}", "DY": "{:.2f}", "Margem Líq.": "{:.2f}"
        })

        st.dataframe(styler, use_container_width=True, column_config={"Preço": st.column_config.NumberColumn(format="R$ %.2f")}, hide_index=True)

    st.divider()
    col_ad1, col_ad2 = st.columns(2)
    with col_ad1:
        st.markdown("""
        <div style="background-color: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 10px; color: #333; height: 100%;">
            <h4 style="margin-top:0; color: #333;">✈️ Nomad: Taxa Zero em Dólar</h4>
            <p style="font-size: 14px;">Ganhe taxa zero na 1ª conversão (até US$ 1.000) para investir nos EUA.</p>
            <p style="font-size: 14px;">Código: <code style="background-color: #eee; padding: 4px; border-radius: 4px; border: 1px solid #ddd; font-weight:bold;">Y39FP3XF8I</code></p>
            <div style="text-align:center;"><a href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I%26n=Jader" target="_blank" style="text-decoration: none; color: white; background-color: #1a1a1a; padding: 10px 15px; border-radius: 5px; font-size: 14px; display: inline-block; width: 100%;">➡️ <b>Abrir Conta Nomad</b></a></div>
        </div>
        """, unsafe_allow_html=True)
    with col_ad2:
        st.markdown("""
        <div style="background-color: #eaf6ff; border: 1px solid #bae0ff; padding: 15px; border-radius: 10px; color: #333; height: 100%;">
            <h4 style="margin-top:0; color: #009ee3;">🤝 Mercado Pago: R$ 30 OFF</h4>
            <p style="font-size: 14px;">Use o app pela primeira vez (pagamento mín. R$ 70) e ganhe <b>R$ 30 de desconto</b>.</p>
            <div style="text-align:center;"><a href="https://mpago.li/1VydVhw" target="_blank" style="text-decoration: none; color: white; background-color: #009ee3; padding: 10px 15px; border-radius: 5px; font-size: 14px; display: inline-block; width: 100%;">➡️ <b>Resgatar R$ 30</b></a></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("⚠️ Atenção! Empresas em Risco / Recup. Judicial")
    st.markdown("**Critérios:** Em Recuperação Judicial (Lista B3) **OU** Alavancagem Alta (Dívida > 3x Patrimônio) **E** Queda no Lucro.")
    if not df_warning.empty:
        def color_negative_red(val):
            if isinstance(val, str) and '-' in val: return 'color: red; font-weight: bold;'
            return ''
        styler_risk = df_warning.style.map(color_negative_red, subset=['Queda Lucro (Ano)']).format({"Preço": "R$ {:.2f}", "Alavancagem (Dív/Patr)": "{:.2f}"})
        st.dataframe(styler_risk, use_container_width=True, hide_index=True)
    else: st.info("Nenhuma ação com os critérios de risco (Dívida Extrema + Queda Lucro) encontrada hoje.")

    st.divider()
    st.subheader("📈 Análise Visual: Cotação vs Lucro")
    options = df_best['Ativo'].tolist()
    idx_default = 0
    if 'LREN3' in options:
        try: idx_default = options.index('LREN3')
        except: pass
    with st.expander("🔎 Selecionar Ação para o Gráfico", expanded=st.session_state.expander_open):
        selected = st.selectbox("Ativo:", options, index=idx_default, on_change=close_expander)
    if selected:
        with st.spinner(f'Gerando gráfico para {selected}...'):
            df_chart = get_chart_data(selected)
        if df_chart is not None and not df_chart.empty:
            # Gráfico Plotly simplificado (para manter código limpo)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_chart['Periodo'], y=df_chart['Cotação'], name="Cotação", mode='lines+markers'))
            fig.update_layout(title=f"{selected}: Histórico 5 Anos", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning(f"Dados históricos indisponíveis para {selected}.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📰 Notícias (Brasília)")
        news = get_market_news()
        if news:
            for n in news: st.markdown(f"**[{n['title']}]({n['link']})**  \n*{n['source']} - {n['date_str']}*")
        else: st.info("Sem notícias.")
    with c2:
        st.subheader("💰 Dividendos Recentes")
        df_divs = get_latest_dividends(df_best['Ativo'].tolist() if not df_best.empty else [])
        if not df_divs.empty:
            df_divs['Data'] = df_divs['Data'].dt.strftime('%d/%m/%Y')
            df_divs['Valor'] = df_divs['Valor'].apply(lambda x: f"R$ {x:.4f}")
            st.dataframe(df_divs, hide_index=True)
        else: st.info("Sem dividendos recentes.")

# === ABA 2: NOVO SCANNER BB (SÓ BRASIL - SEMANAL - SÓ LOWER) ===
with tab2:
    st.subheader("📉 Ações Brasileiras na Banda Inferior (Semanal)")
    st.markdown("""
    Lista rastreada automaticamente de ações da B3 onde a **Mínima da Semana** tocou a **Banda de Bollinger Inferior (20, 2)** na semana atual ou anterior.
    <br><small>*Clique em uma linha da tabela para atualizar o gráfico ao lado.*</small>
    """, unsafe_allow_html=True)
    
    col_list, col_chart = st.columns([1, 2])
    
    with col_list:
        if not df_scan_bb.empty:
            st.write(f"**{len(df_scan_bb)} Oportunidades Encontradas:**")
            
            # Tabela Interativa
            event = st.dataframe(
                df_scan_bb[['Ativo', 'Quando?', 'Preço Atual', 'Mínima Sem.', 'Banda Inf', 'Distância Fech %']].style.format({
                    "Preço Atual": "{:.2f}", "Mínima Sem.": "{:.2f}", "Banda Inf": "{:.2f}", "Distância Fech %": "{:.2f}%"
                }).map(lambda x: 'background-color: #ffcccb; color: black', subset=['Distância Fech %']),
                use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row",
                key="tab2_dataframe" # Key fixa para estabilidade
            )
            
            # Lógica de Seleção Persistente
            if len(event.selection.rows) > 0:
                selected_index = event.selection.rows[0]
                ticker_selecionado = df_scan_bb.iloc[selected_index]['TV_Symbol']
                st.session_state.selected_ticker_tab2 = ticker_selecionado
            
        else:
            st.info("Nenhuma ação brasileira tocando a banda inferior nas últimas duas semanas.")
            
    with col_chart:
        # Usa o estado da sessão para garantir que o gráfico não resete
        tv_symbol = st.session_state.selected_ticker_tab2
        clean_name = tv_symbol.split(":")[-1]
        st.markdown(f"#### Gráfico Semanal: {clean_name}")
        show_chart_widget(tv_symbol)
