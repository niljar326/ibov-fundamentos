import streamlit as st
import pandas as pd
import fundamentus
import plotly.graph_objects as go
import plotly.express as px
import feedparser
import yfinance as yf
import datetime
from datetime import timedelta
from time import mktime

# --- 1. Configuração da Página (SEO) ---
st.set_page_config(
    page_title="Melhores Ações Ibovespa 2025 | Ranking Fundamentalista e Dividendos",
    layout="wide",
    page_icon="🇧🇷"
)

# --- CSS Global ---
st.markdown("""
    <style>
    /* Cabeçalhos à direita */
    [data-testid="stDataFrame"] table tr th { text-align: right !important; }
    /* Células à esquerda */
    [data-testid="stDataFrame"] table tr td { text-align: left !important; }
    </style>
""", unsafe_allow_html=True)

# --- Estado ---
if 'expander_open' not in st.session_state: st.session_state.expander_open = True
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
    return f"{val:.0f}"

def get_current_data():
    now = datetime.datetime.now()
    return now.strftime("%B"), now.year

# --- Dados Principais ---
@st.cache_data(ttl=3600*6)
def get_ranking_data():
    try:
        df = fundamentus.get_resultado()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'papel'}, inplace=True)
        cols = ['pl', 'roe', 'dy', 'evebit', 'cotacao', 'liq2m', 'mrgliq', 'divbpatr', 'c5y']
        for col in cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_fundamentus_col)
            else:
                df[col] = 0.0
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
    df_filtered.rename(columns={'papel': 'Ativo', 'cotacao': 'Preço', 'pl': 'P/L', 'evebit': 'EV/EBIT', 'dy': 'DY', 'roe': 'ROE', 'mrgliq': 'Margem Líq.'}, inplace=True)
    return df_filtered.sort_values(by=['P/L', 'Margem Líq.'], ascending=[True, False]).reset_index(drop=True)

# --- Lógica de Risco ---
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
        status = "Recup. Judicial / Reestruturação" if ticker in lista_rj else "Alta Alavancagem"
        lucro_queda_str, val_queda = "N/D", 0.0
        try:
            stock = yf.Ticker(ticker + ".SA")
            fin = stock.financials
            if not fin.empty:
                inc_row = None
                for name in ['Net Income', 'Net Income Common', 'Net Income Continuous']:
                    if name in fin.index:
                        inc_row = fin.loc[name]
                        break
                if inc_row is not None and len(inc_row) >= 2:
                    if inc_row.iloc[0] < inc_row.iloc[1]:
                        pct = ((inc_row.iloc[0] - inc_row.iloc[1]) / abs(inc_row.iloc[1])) * 100
                        val_queda = pct
                        lucro_queda_str = f"{pct:.1f}%"
                    else: lucro_queda_str = "Subiu/Estável"
        except: pass
        if val_queda < 0 or ticker in lista_rj:
            risk_data.append({'Ativo': ticker, 'Preço': row['cotacao'], 'Alavancagem (Dív/Patr)': row['divbpatr'], 'Queda Lucro (Ano)': lucro_queda_str, 'Situação': status})
    return pd.DataFrame(risk_data)

# --- LÓGICA DO GRÁFICO ANIMADO DINÂMICO (Rank por Trimestre) ---
@st.cache_data(ttl=3600*24)
def get_animated_ey_data_dynamic(ticker_list):
    # Aumentamos o pool para 30 ações para permitir que "novas" entrem no Top 10
    pool_tickers = ticker_list[:30] 
    raw_data = []

    # 1. Coleta Dados (Pool Expandido)
    for t in pool_tickers:
        try:
            stock = yf.Ticker(t + ".SA")
            q_fin = stock.quarterly_financials.T
            hist = stock.history(period="5y")
            
            if q_fin.empty or hist.empty: continue

            eps_col = None
            possible_cols = ['Basic EPS', 'Diluted EPS', 'Earnings Per Share']
            for c in q_fin.columns:
                if c in possible_cols:
                    eps_col = c
                    break
            
            if not eps_col: continue 

            for date, row in q_fin.iterrows():
                eps_val = row[eps_col]
                if pd.isna(eps_val): continue

                date_clean = pd.to_datetime(date).tz_localize(None)
                mask = hist.index.tz_localize(None) <= date_clean
                if not mask.any(): continue
                
                price_at_date = hist.loc[mask, 'Close'].iloc[-1]
                
                if price_at_date > 0:
                    ey_val = (eps_val / price_at_date) * 100
                    raw_data.append({
                        'Ativo': t,
                        'Data_Real': date_clean,
                        'EY': ey_val
                    })
        except: continue

    if not raw_data: return pd.DataFrame()
    
    df_raw = pd.DataFrame(raw_data)
    
    # 2. Esqueleto e Preenchimento (Forward Fill)
    all_dates = sorted(df_raw['Data_Real'].unique())
    master_data = []
    for t in pool_tickers:
        for d in all_dates:
            master_data.append({'Ativo': t, 'Data_Real': d})
            
    df_master = pd.DataFrame(master_data)
    df_merged = pd.merge(df_master, df_raw, on=['Ativo', 'Data_Real'], how='left')
    df_merged = df_merged.sort_values(by=['Ativo', 'Data_Real'])
    df_merged['EY'] = df_merged.groupby('Ativo')['EY'].ffill().fillna(0)
    
    # 3. Formatação de Data
    df_merged['Trimestre'] = df_merged['Data_Real'].apply(
        lambda x: f"{x.year}-Q{(x.month-1)//3 + 1}"
    )

    # 4. LÓGICA DE CORTE: TOP 10 POR TRIMESTRE
    # Aqui é onde a mágica acontece: Para cada trimestre, pegamos só os top 10 DAQUELE MOMENTO.
    frames_list = []
    unique_quarters = sorted(df_merged['Trimestre'].unique())

    for q in unique_quarters:
        # Pega dados daquele trimestre
        df_q = df_merged[df_merged['Trimestre'] == q].copy()
        
        # Ordena e pega Top 10
        df_q_top = df_q.sort_values(by='EY', ascending=False).head(10)
        
        frames_list.append(df_q_top)

    # Reconstrói o DataFrame final apenas com os vencedores de cada rodada
    df_final = pd.concat(frames_list)
    
    return df_final.sort_values(by=['Data_Real', 'EY'], ascending=[True, True])

# --- Lógica do Gráfico de Linha (Individual) ---
@st.cache_data(ttl=3600*24)
def get_chart_data(ticker):
    try:
        stock = yf.Ticker(ticker + ".SA")
        financials, quarterly, hist = stock.financials.T, stock.quarterly_financials.T, stock.history(period="5y")
        
        for d in [financials, quarterly, hist]:
            if not d.empty: d.index = pd.to_datetime(d.index).tz_localize(None)

        def find_col(df, candidates):
            for cand in candidates:
                for col in df.columns:
                    if cand.lower() in col.lower(): return col
            return None

        rev_col = find_col(financials, ['Total Revenue', 'Operating Revenue', 'Revenue'])
        inc_col = find_col(financials, ['Net Income', 'Net Income Common', 'Lucro Liquido'])
        
        if not rev_col or not inc_col: return None

        data_rows = []
        for date, row in financials.tail(3).iterrows():
            price = 0.0
            if not hist.empty:
                mask = hist.index <= date
                if mask.any(): price = hist.loc[mask, 'Close'].iloc[-1]
            data_rows.append({'Periodo': str(date.year), 'Receita': row[rev_col], 'Lucro': row[inc_col], 'Cotação': price})
            
        if not quarterly.empty:
            q_rev = find_col(quarterly, ['Total Revenue', 'Revenue'])
            q_inc = find_col(quarterly, ['Net Income', 'Lucro'])
            if q_rev and q_inc:
                data_rows.append({'Periodo': 'Últimos 12m', 'Receita': quarterly[q_rev].head(4).sum(), 'Lucro': quarterly[q_inc].head(4).sum(), 'Cotação': hist['Close'].iloc[-1] if not hist.empty else 0})
        
        df_f = pd.DataFrame(data_rows)
        df_f['Receita_Texto'] = df_f['Receita'].apply(format_short_number)
        return df_f
    except: return None

# --- Dividendos ---
@st.cache_data(ttl=3600*6)
def get_latest_dividends(ticker_list):
    divs_data = []
    for ticker in ticker_list[:10]:
        try:
            stk = yf.Ticker(ticker + ".SA")
            d = stk.dividends
            if not d.empty: divs_data.append({'Ativo': ticker, 'Valor': d.iloc[-1], 'Data': d.index[-1]})
        except: continue
    if divs_data:
        df = pd.DataFrame(divs_data)
        df['Data'] = pd.to_datetime(df['Data']).dt.tz_localize(None)
        return df.sort_values('Data', ascending=False).head(5)
    return pd.DataFrame()

# --- Notícias ---
@st.cache_data(ttl=1800)
def get_market_news():
    feeds = {'Money Times': 'https://www.moneytimes.com.br/feed/', 'InfoMoney': 'https://www.infomoney.com.br/feed/', 'E-Investidor': 'https://einvestidor.estadao.com.br/feed/'}
    news = []
    for src, url in feeds.items():
        try:
            f = feedparser.parse(url)
            for e in f.entries[:3]:
                try: dt = datetime.datetime.fromtimestamp(mktime(e.published_parsed)) - timedelta(hours=3)
                except: dt = datetime.datetime.now()
                news.append({'title': e.title, 'link': e.link, 'date_obj': dt, 'date_str': dt.strftime("%d/%m %H:%M"), 'source': src})
        except: continue
    return sorted(news, key=lambda x: x['date_obj'], reverse=True)[:6]

# --- Interface ---
st.title("🇧🇷 Ranking de Ações Baratas e Rentáveis - B3")
mes_txt, ano_int = get_current_data()
st.markdown(f"**Referência:** {mes_txt}/{ano_int}")

st.markdown("""<div style="text-align: justify; margin-bottom: 20px;">
Este <b>Screener Fundamentalista</b> filtra as melhores oportunidades. Veja abaixo o Ranking, Alertas de Risco e a Evolução da Rentabilidade.
</div>""", unsafe_allow_html=True)

with st.spinner('Processando dados do mercado...'):
    df_raw = get_ranking_data()
    df_best = apply_best_filters(df_raw)
    df_warning = get_risk_table(df_raw)

# 1. Melhores Ações
if not df_best.empty:
    st.subheader("🏆 Melhores Ações (Oportunidades)")
    cols_view = ['Ativo', 'Preço', 'EV/EBIT', 'P/L', 'ROE', 'DY', 'Margem Líq.']
    even_cols = ['Preço', 'P/L', 'DY']
    styler = df_best[cols_view].style.map(lambda x: 'background-color: #f2f2f2; color: black;', subset=even_cols)\
        .format({"Preço": "R$ {:.2f}", "EV/EBIT": "{:.2f}", "P/L": "{:.2f}", "ROE": "{:.2f}", "DY": "{:.2f}", "Margem Líq.": "{:.2f}"})
    st.dataframe(styler, use_container_width=True, hide_index=True)

# 2. Tabela de Risco
st.divider()
st.subheader("⚠️ Atenção! Empresas em Risco / Recup. Judicial")
if not df_warning.empty:
    def color_red(v): return 'color: red; font-weight: bold;' if isinstance(v, str) and '-' in v else ''
    st.dataframe(df_warning.style.map(color_red, subset=['Queda Lucro (Ano)']).format({"Preço": "R$ {:.2f}", "Alavancagem (Dív/Patr)": "{:.2f}"}), use_container_width=True, hide_index=True)
else: st.info("Nenhuma ação crítica encontrada.")

# 3. GRÁFICO ANIMADO DINÂMICO
st.divider()
st.subheader("📺 Corrida de Rentabilidade (Earnings Yield)")
st.markdown("Visualização dinâmica: As ações **entram e saem do Top 10** a cada trimestre conforme seu desempenho histórico.")

if not df_best.empty:
    # Aumentamos o pool de busca para Top 30 para permitir rotatividade no gráfico
    with st.spinner("Analisando histórico de 30 empresas para montar a corrida (pode levar 20s)..."):
        top_30_tickers = df_best['Ativo'].head(30).tolist()
        df_anim = get_animated_ey_data_dynamic(top_30_tickers)

    if not df_anim.empty:
        max_ey = df_anim['EY'].max()
        range_x_fixed = [0, max_ey * 1.1]

        fig_anim = px.bar(
            df_anim, 
            x="EY", 
            y="Ativo", 
            animation_frame="Trimestre", 
            orientation='h',
            text="EY",
            range_x=range_x_fixed,
            color="Ativo",
            title="Top 10 Earnings Yield (%) por Trimestre"
        )
        
        fig_anim.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_anim.update_layout(
            xaxis_title="Earnings Yield (LPA/Preço) %",
            yaxis_title="",
            showlegend=False,
            height=600, # Aumentei um pouco altura
            yaxis={'categoryorder':'total ascending'} # Isso ajuda a ordenar visualmente
        )
        
        st.plotly_chart(fig_anim, use_container_width=True)
        st.caption("*Nota: O gráfico seleciona dinamicamente as 10 melhores daquele trimestre dentre as Top 30 atuais.*")
    else:
        st.warning("Dados históricos insuficientes para gerar a animação agora.")

# 4. Gráfico Individual
st.divider()
st.subheader("📈 Análise Detalhada: Cotação vs Lucro")
options = df_best['Ativo'].tolist()
idx_def = options.index('LREN3') if 'LREN3' in options else 0
with st.expander("🔎 Selecionar Ação", expanded=st.session_state.expander_open):
    sel = st.selectbox("Ativo:", options, index=idx_def, on_change=close_expander)

if sel:
    with st.spinner(f'Gerando gráfico para {sel}...'):
        df_c = get_chart_data(sel)
    if df_c is not None:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_c['Periodo'], y=df_c['Receita'], name="Receita", marker=dict(color='#A9A9A9'), text=df_c['Receita_Texto'], textposition='outside', yaxis='y1'))
        fig.add_trace(go.Scatter(x=df_c['Periodo'], y=df_c['Lucro'], name="Lucro", mode='lines+markers', line=dict(color='green', width=3), yaxis='y2'))
        fig.add_trace(go.Scatter(x=df_c['Periodo'], y=df_c['Cotação'], name="Cotação", mode='lines+markers', line=dict(color='blue', width=3), yaxis='y3'))
        fig.update_layout(title=f"{sel}: Receita vs Lucro vs Preço", xaxis=dict(title="Período"), yaxis=dict(title="Receita", showgrid=False), yaxis2=dict(title="Lucro", overlaying="y", side="right", showgrid=False), yaxis3=dict(title="Cotação", overlaying="y", side="right", position=0.95, showgrid=False), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
    else: st.warning("Sem dados.")

# 5. Notícias/Divs
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("📰 Notícias (Brasília)")
    for n in get_market_news(): st.markdown(f"**[{n['title']}]({n['link']})**\n*{n['source']} - {n['date_str']}*")
with c2:
    st.subheader("💰 Dividendos Recentes")
    st.dataframe(get_latest_dividends(df_best['Ativo'].tolist()), hide_index=True)
