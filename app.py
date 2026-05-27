/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  TrendingUp, 
  Sparkles, 
  TrendingDown, 
  BadgeAlert, 
  HelpCircle, 
  Clock, 
  Newspaper, 
  Coins, 
  ArrowUpRight, 
  ExternalLink,
  DollarSign,
  Briefcase,
  ChevronRight,
  Info
} from 'lucide-react';

import { STOCK_DATABASE, MARKET_NEWS, LATEST_DIVIDENDS } from './data/stocks';
import LockScreen from './components/LockScreen';
import StockChart from './components/StockChart';
import TradingViewWidget from './components/TradingViewWidget';

type TabType = 'ranking' | 'magic' | 'graham' | 'eps' | 'asymmetry';

export default function App() {
  // Navigation & User settings
  const [activeTab, setActiveTab] = useState<TabType>('ranking');
  const [userName, setUserName] = useState(() => localStorage.getItem('b3_user_name') || '');
  const [appLiberado, setAppLiberado] = useState(() => localStorage.getItem('b3_app_liberado') === 'true');
  const [utcTime, setUtcTime] = useState(new Date());

  // Dropdown states for interactive chart selects
  const [selectedChartTicker, setSelectedChartTicker] = useState('PETR4');
  const [selectedAsymmetryTicker, setSelectedAsymmetryTicker] = useState('BBSE3');

  // Real-time UTC clock ticker
  useEffect(() => {
    const interval = setInterval(() => {
      setUtcTime(new Date());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Sync persists
  useEffect(() => {
    localStorage.setItem('b3_user_name', userName);
  }, [userName]);

  const handleUnlock = () => {
    setAppLiberado(true);
    localStorage.setItem('b3_app_liberado', 'true');
  };

  const handleLockReset = () => {
    setAppLiberado(false);
    localStorage.setItem('b3_app_liberado', 'false');
  };

  // 1. DATA ENGINES - FUNDAMENTALIST FILTER
  const dfBest = STOCK_DATABASE.filter(stock => 
    stock.roe > 0.05 &&
    stock.pl > 0 && stock.pl < 15 &&
    stock.evebit > 0 && stock.evebit < 10 &&
    stock.dy > 0.04 &&
    stock.mrgliq > 0.05 &&
    stock.liq2m > 200000
  ).sort((a, b) => {
    if (a.pl !== b.pl) return a.pl - b.pl; // lower P/L first
    return b.mrgliq - a.mrgliq;            // higher Margem Líquida secondary
  });

  // Risk stocks list (Alavancagem ou RJ)
  const listRJ = ['OIBR3', 'OIBR4', 'AMER3', 'GOLL4', 'AZUL4'];
  const dfWarning = STOCK_DATABASE.filter(stock => 
    stock.divbpatr > 3.0 || listRJ.includes(stock.papel)
  ).sort((a, b) => b.divbpatr - a.divbpatr);

  // 2. DATA ENGINES - MAGIC FORMULA
  // Rank universes by highest EY and highest ROIC
  const candidatesMagic = STOCK_DATABASE.filter(stock => stock.liq2m > 100000 && stock.evebit > 0);
  const eySorted = [...candidatesMagic].sort((a, b) => a.evebit - b.evebit); // EV/EBIT lower is better (which means EY 1/[EV/EBIT] is higher)
  const roicSorted = [...candidatesMagic].sort((a, b) => b.roic - a.roic);   // ROIC higher is better

  const dfMagic = candidatesMagic.map(stock => {
    const rankEy = eySorted.findIndex(s => s.papel === stock.papel) + 1;
    const rankRoic = roicSorted.findIndex(s => s.papel === stock.papel) + 1;
    const scoreMagico = rankEy + rankRoic;
    return {
      ...stock,
      rankEy,
      rankRoic,
      scoreMagico,
      earningYield: 1 / stock.evebit
    };
  }).sort((a, b) => a.scoreMagico - b.scoreMagico).slice(0, 40);

  // 3. DATA ENGINES - GRAHAM VALUATION
  const dfGraham = STOCK_DATABASE.filter(stock => stock.pl > 0 && stock.pvp > 0 && stock.liq2m > 100000)
    .map(stock => {
      const valorIntrinseco = Math.sqrt(22.5 * stock.lpa * stock.vpa);
      const ratio = valorIntrinseco / stock.cotacao;
      return {
        ...stock,
        valorIntrinseco,
        ratio,
        status: ratio > 1.0 ? 'Barata' : 'Cara'
      };
    }).sort((a, b) => b.ratio - a.ratio);

  // 4. DATA ENGINES - EPS DILUÍDO (> 1.0)
  const dfEps = STOCK_DATABASE.filter(stock => stock.liq2m > 100000 && stock.epsTrimestral > 1.0)
    .sort((a, b) => b.epsTrimestral - a.epsTrimestral);

  // 5. DATA ENGINES - ASYMMETRY (Lucro acima do preço)
  const asymmetryStocks = STOCK_DATABASE.filter(stock => {
    const h = stock.historico;
    if (!h || h.length < 2) return false;
    const prices = h.map(x => x.cotacao);
    const profits = h.map(x => x.lucro);

    const pMin = Math.min(...prices);
    const pMax = Math.max(...prices);
    const lMin = Math.min(...profits);
    const lMax = Math.max(...profits);

    if (pMax > pMin && lMax > lMin) {
      const currentP = prices[prices.length - 1];
      const currentL = profits[profits.length - 1];
      const pPos = (currentP - pMin) / (pMax - pMin);
      const lPos = (currentL - lMin) / (lMax - lMin);
      return lPos > pPos; // relative progress in profit ends above relative progress in stock price
    }
    return false;
  });

  const activeChartData = STOCK_DATABASE.find(s => s.papel === selectedChartTicker)?.historico || [];
  const activeAsymmetryChartData = STOCK_DATABASE.find(s => s.papel === selectedAsymmetryTicker)?.historico || [];

  // Current month reference
  const refMonth = utcTime.toLocaleDateString('pt-BR', { month: 'long', year: 'numeric' });

  // Formatting helper
  const fmtPct = (val: number) => `${(val * 100).toFixed(1)}%`;
  const fmtCurrency = (val: number) => `R$ ${val.toLocaleString('pt-BR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans flex flex-col transition-all selection:bg-blue-100 selection:text-blue-900" id="b3-dashboard-root">
      
      {/* Top Banner Advertisement simulate */}
      <div className="bg-gradient-to-r from-blue-900 to-indigo-950 text-white text-center py-2.5 px-4 text-xs font-medium flex items-center justify-center gap-2 relative overflow-hidden" id="sponsored-top-banner">
        <span className="bg-amber-400 text-slate-900 font-bold px-1.5 py-0.5 rounded text-[10px] tracking-wider uppercase">AD</span>
        <span className="truncate">Ganhe benefícios e apoie nossa comunidade acessando os parceiros abaixo ou na barra lateral!</span>
        <div className="absolute top-0 right-0 bottom-0 left-0 bg-[radial-gradient(circle,rgba(255,255,255,0.15)_1px,transparent_1px)] bg-[size:16px_16px] pointer-events-none opacity-40" />
      </div>

      <div className="flex-1 w-full max-w-7xl mx-auto px-4 md:px-8 py-8 flex flex-col lg:flex-row gap-8">
        
        {/* SIDEBAR ON LEFT */}
        <aside className="w-full lg:w-72 shrink-0 flex flex-col gap-6" id="dashboard-sidebar">
          
          {/* Main Logo Card */}
          <div className="bg-white border border-slate-200/80 rounded-3xl p-6 shadow-sm flex flex-col gap-3 relative overflow-hidden">
            <div className="h-12 w-12 rounded-2xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white font-sans font-bold text-xl shadow-lg shadow-blue-500/20">
              B3
            </div>
            <div>
              <h1 className="font-sans font-extrabold tracking-tight text-slate-900 text-lg leading-tight">
                Ranking Ibovespa
              </h1>
              <p className="text-xs text-slate-400 font-mono mt-0.5">Versão Ultra-Polida 2025</p>
            </div>
            <div className="absolute top-4 right-4 text-slate-100 font-bold font-mono text-5xl pointer-events-none select-none">
              BR
            </div>
          </div>

          {/* User Welcome personalization */}
          <div className="bg-white border border-slate-200/80 rounded-3xl p-6 shadow-sm flex flex-col gap-3">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Identificação</span>
            <div className="flex flex-col gap-2">
              <label htmlFor="sidebar-name-input" className="text-xs text-slate-500 font-medium">Seu nome para relatórios:</label>
              <input
                id="sidebar-name-input"
                type="text"
                autoComplete="off"
                placeholder="Ex: Nilton, Maria, Lucas..."
                value={userName}
                onChange={(e) => setUserName(e.target.value)}
                className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10 outline-none transition-all placeholder:text-slate-400 text-slate-800 bg-slate-50/50"
              />
            </div>
            {userName && (
              <p className="text-[11px] text-slate-500 italic">
                Bem-vindo(a), <strong className="text-slate-800">{userName}</strong>!
              </p>
            )}
          </div>

          {/* Real-time Clock Widget */}
          <div className="bg-white border border-slate-200/80 rounded-3xl p-5 shadow-sm flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-blue-500 shrink-0" />
              <span className="text-xs font-semibold text-slate-700">Relatório B3 Online</span>
            </div>
            <span className="font-mono text-[11px] font-bold text-slate-500 tracking-wider">
              {utcTime.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })} BRT
            </span>
          </div>

          {/* Giga+ Fibra WhatsApp Affiliate banner */}
          <a
            href="https://wa.me/552220410353?text=Use%20o%20codigo%20DVT329%20e%20ganhe%2020%25%20nas%20duas%20primeiras%20mensalidades"
            target="_blank"
            rel="noopener noreferrer"
            className="group block bg-gradient-to-br from-emerald-500 to-green-600 rounded-3xl p-6 text-white shadow-md shadow-emerald-500/10 hover:shadow-lg transition-all relative overflow-hidden"
            id="whatsapp-affiliate-banner"
          >
            {/* Ambient pattern */}
            <div className="absolute inset-0 bg-[radial-gradient(circle,rgba(255,255,255,0.12)_1px,transparent_1px)] bg-[size:16px_16px] pointer-events-none opacity-50" />
            
            <div className="flex items-center gap-3 mb-4">
              <div className="h-10 w-10 bg-white/20 backdrop-blur-md rounded-2xl flex items-center justify-center text-white shrink-0">
                <svg className="w-5 h-5 fill-current" viewBox="0 0 24 24">
                  <path d="M.012 24l1.635-5.97A11.91 11.91 0 0 1 0 12.008C0 5.4 5.4 0 12.004 0c6.602 0 11.984 5.4 11.984 12.008 0 6.603-5.382 12.004-11.984 12.004-2.102 0-4.14-.548-5.955-1.583L0 24zm6.59-4.814l.385.228a9.92 9.92 0 0 0 5.033 1.365c5.496 0 9.97-4.47 9.97-9.972 0-5.503-4.474-9.972-9.97-9.972-5.507 0-9.978 4.47-9.978 9.972 0 2.215.736 4.316 2.13 6.024l.25.309-1.258 4.605 4.71-1.232c-.1-.005.14.07-.272-.327zm10.741-6.19c-.312-.158-1.85-.913-2.136-1.018-.287-.105-.495-.158-.703.158-.207.315-.8.1-.976.315-.177.21-.355.237-.667.08-.312-.158-1.32-.487-2.514-1.55-.93-.83-1.558-1.855-1.74-2.17-.183-.316-.02-.487.137-.644.14-.141.312-.368.468-.553.156-.184.208-.316.312-.526.104-.21.052-.395-.026-.553-.078-.158-.703-1.697-.963-2.329-.253-.614-.51-.53-.703-.54l-.597-.013c-.208 0-.547.078-.833.394-.287.316-1.094 1.07-1.094 2.607 0 1.539 1.12 3.026 1.276 3.237.156.21 2.2 3.36 5.33 4.715 1.7.737 2.378.842 3.226.71.6-.092 1.85-.754 2.11-1.474.26-.719.26-1.334.182-1.465-.078-.131-.286-.21-.598-.368z"/>
                </svg>
              </div>
              <div className="flex flex-col">
                <span className="font-bold text-xs tracking-wide">Código: DVT329</span>
                <span className="text-[10px] text-emerald-100">Giga+ Fibra Telecom</span>
              </div>
            </div>
            
            <h4 className="font-bold text-sm leading-tight mb-1">
              Internet Estável com 20% OFF!
            </h4>
            <p className="text-[11px] text-emerald-55 leading-relaxed mb-4">
              Assine as duas primeiras mensalidades de fibra de alta performance com desconto exclusivo. Clique para falar com o atendente direto.
            </p>

            <div className="bg-white/10 dark:bg-black/15 px-3 py-2 rounded-xl text-xs font-semibold flex items-center justify-between hover:bg-white/20 transition-all font-sans">
              <span>Resgatar Cupom no WhatsApp</span>
              <ArrowUpRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
            </div>
          </a>

          {/* Quick Stats Panel / About */}
          <div className="bg-white border border-slate-200/80 rounded-3xl p-6 shadow-sm flex flex-col gap-4">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Universo de Cobertura</span>
            <div className="flex flex-col gap-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-500">Ações Monitoradas:</span>
                <strong className="text-slate-800">{STOCK_DATABASE.length} empresas</strong>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-500">Liquidez diária base:</span>
                <strong className="text-slate-800">&gt; R$ 100k</strong>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-500">Métricas analíticas:</span>
                <strong className="text-slate-800"> Graham & Greenblatt</strong>
              </div>
            </div>

            {appLiberado && (
              <div className="pt-2 border-t border-slate-100">
                <button
                  onClick={handleLockReset}
                  className="w-full text-center py-2 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-xl text-xs font-semibold transition-all"
                  title="Bloquear o dashboard para testes de anúncios"
                >
                  🔒 Testar Bloqueio de Anúncio
                </button>
              </div>
            )}
          </div>

        </aside>

        {/* MAIN DASHBOARD BLOCK */}
        <main className="flex-1 flex flex-col gap-8" id="dashboard-main-content">
          
          {/* Main Title & Month Selector */}
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-blue-600 animate-pulse" />
                <span className="text-xs font-bold text-slate-500 uppercase tracking-wider font-mono">B3 Ibovespa Inteligente</span>
              </div>
              <h2 className="font-sans font-black tracking-tight text-slate-900 text-3xl mt-1 leading-tight">
                Análise de Ações Baratas e Rentáveis
              </h2>
              {userName ? (
                <p className="text-slate-500 text-sm mt-1">
                  Relatório tático de investimentos customizado para <strong className="text-slate-800 font-semibold">{userName}</strong> • Referência de <strong>{refMonth}</strong>
                </p>
              ) : (
                <p className="text-slate-500 text-sm mt-1">
                  Relatório tático de investimentos • Referência de <strong className="text-slate-800 capitalize font-semibold">{refMonth}</strong>
                </p>
              )}
            </div>

            <div className="inline-flex self-start md:self-auto bg-blue-50 border border-blue-100 px-4 py-2 rounded-2xl items-center gap-2.5 shadow-sm">
              <span className="text-lg">🇧🇷</span>
              <div className="text-left font-sans">
                <p className="text-[10px] uppercase tracking-wider font-bold text-blue-700 leading-none">Bolsa Paulista</p>
                <p className="text-xs font-black text-blue-950 mt-0.5">Ibovespa em 2025</p>
              </div>
            </div>
          </div>

          {/* Premium Tab Selection Navigation (Using stylings representing Streamlit radio tab mimicking buttons) */}
          <div className="bg-white border border-slate-200/80 p-1.5 rounded-2xl flex flex-wrap gap-1 leading-none shadow-sm" id="premium-tab-bar">
            {[
              { id: 'ranking', label: '🏆 Ranking Fundamentalista', desc: 'Filtros Sérios' },
              { id: 'magic', label: '✨ Fórmula Mágica', desc: 'Joel Greenblatt' },
              { id: 'graham', label: '💎 Graham Valuation', desc: 'Preço Justo' },
              { id: 'eps', label: '📈 EPS Diluído', desc: 'Lucro Trimestral' },
              { id: 'asymmetry', label: '📉 Assimetria Lucro/Preço', desc: 'Robô de Sinais' }
            ].map(tab => (
              <button
                key={tab.id}
                id={`tab-trigger-button-${tab.id}`}
                onClick={() => setActiveTab(tab.id as TabType)}
                className={`flex-1 min-w-[140px] px-3 py-2.5 rounded-xl text-center transition-all flex flex-col items-center justify-center gap-0.5 text-xs font-bold leading-none ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-600/15'
                    : 'bg-white text-slate-600 hover:bg-slate-50 border border-transparent'
                }`}
              >
                <span>{tab.label}</span>
                <span className={`text-[9px] font-medium leading-none ${activeTab === tab.id ? 'text-blue-100' : 'text-slate-400'}`}>
                  {tab.desc}
                </span>
              </button>
            ))}
          </div>

          {/* Tab content area mapped securely */}
          <div className="min-h-[400px] flex flex-col gap-8" id="dashboard-tab-views">

            {/* IF APP IS COMPLETELY LOCKED */}
            {!appLiberado ? (
              <LockScreen onUnlock={handleUnlock} />
            ) : (
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeTab}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.25 }}
                  className="flex flex-col gap-8"
                >
                  
                  {/* TAB 1: RANKING FUNDAMENTALISTA */}
                  {activeTab === 'ranking' && (
                    <div className="flex flex-col gap-8" id="tab-view-ranking">
                      
                      {/* Introductory card */}
                      <div className="bg-white border border-slate-200/80 rounded-3xl p-6 shadow-sm flex flex-col gap-2">
                        <div className="flex items-center gap-2 text-blue-600 font-bold text-xs uppercase font-mono">
                          <TrendingUp className="w-4 h-4" />
                          <span>Critérios Fundamentalistas Aplicados</span>
                        </div>
                        <h3 className="font-bold text-slate-800 text-lg">Ranking de Oportunidades Saudáveis</h3>
                        <p className="text-slate-500 text-sm leading-relaxed">
                          Filtramos o mercado brasileiro aplicando restrições de alto nível: <strong>ROE &gt; 5%</strong>, <strong>0 &lt; P/L &lt; 15</strong>, <strong>0 &lt; EV/EBIT &lt; 10</strong>, dividendos com <strong>Dividend Yield &gt; 4%</strong>, margem líquida positiva superior a <strong>5%</strong> e liquidez diária expressiva acima de <strong>R$ 200 mil</strong>.
                        </p>
                      </div>

                      {/* Best Stocks Table */}
                      <div className="bg-white border border-slate-200/80 rounded-3xl shadow-sm overflow-hidden flex flex-col" id="best-stocks-table-card">
                        <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
                          <h4 className="font-bold text-slate-900 text-sm flex items-center gap-2">
                            🏆 Melhores Ações Fundamentalistas da B3 ({dfBest.length})
                          </h4>
                          <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 uppercase">Filtros Ativos</span>
                        </div>
                        <div className="overflow-x-auto">
                          <table className="w-full text-left border-collapse text-xs">
                            <thead>
                              <tr className="bg-slate-50/70 border-b border-slate-200 text-slate-400 uppercase tracking-wider font-mono font-bold text-[10px]">
                                <th className="px-6 py-3">Ativo</th>
                                <th className="px-6 py-3">Nome Comercial</th>
                                <th className="px-6 py-3 text-right">Preço Atual</th>
                                <th className="px-6 py-3 text-right">0 &lt; P/L &lt; 15</th>
                                <th className="px-6 py-3 text-right">EV/EBIT</th>
                                <th className="px-6 py-3 text-right">ROE</th>
                                <th className="px-6 py-3 text-right">Div. Yield (DY)</th>
                                <th className="px-6 py-3 text-right">Margem Líq (%)</th>
                              </tr>
                            </thead>
                            <tbody>
                              {dfBest.map((stock, i) => (
                                <tr key={stock.papel} className="border-b border-slate-100 hover:bg-slate-50/80 transition-colors font-medium text-slate-700">
                                  <td className="px-6 py-3.5 font-bold text-slate-950 font-mono text-xs">{stock.papel}</td>
                                  <td className="px-6 py-3.5 text-slate-400 font-sans">{stock.empresa}</td>
                                  <td className="px-6 py-3.5 text-right font-mono text-slate-900 font-semibold">{fmtCurrency(stock.cotacao)}</td>
                                  <td className="px-6 py-3.5 text-right font-mono text-blue-600 font-bold bg-blue-50/30">{stock.pl.toFixed(2)}</td>
                                  <td className="px-6 py-3.5 text-right font-mono text-amber-700">{stock.evebit.toFixed(2)}</td>
                                  <td className="px-6 py-3.5 text-right font-mono text-slate-900">{fmtPct(stock.roe)}</td>
                                  <td className="px-6 py-3.5 text-right font-mono text-emerald-600 font-semibold">{fmtPct(stock.dy)}</td>
                                  <td className="px-6 py-3.5 text-right font-mono text-slate-900">{fmtPct(stock.mrgliq)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>

                      {/* Risks alert list */}
                      <div className="bg-white border border-red-200/60 rounded-3xl shadow-sm overflow-hidden flex flex-col" id="warning-stocks-table-card">
                        <div className="px-6 py-5 border-b border-red-100 bg-red-50/30 flex items-center justify-between">
                          <h4 className="font-bold text-red-900 text-sm flex items-center gap-2">
                            <BadgeAlert className="w-4 h-4 text-red-600" />
                            ⚠️ Empresas em Situação de Risco ou Alta Alavancagem ({dfWarning.length})
                          </h4>
                          <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-red-100 text-red-800 uppercase">Atenção</span>
                        </div>
                        <div className="overflow-x-auto">
                          <table className="w-full text-left border-collapse text-xs">
                            <thead>
                              <tr className="bg-slate-50/70 border-b border-slate-200 text-slate-400 uppercase tracking-wider font-mono font-bold text-[10px]">
                                <th className="px-6 py-3">Ativo</th>
                                <th className="px-6 py-3">Nome Comercial</th>
                                <th className="px-6 py-3 text-right">Cotação</th>
                                <th className="px-6 py-3 text-right">Alavancagem (Div/Patr)</th>
                                <th className="px-6 py-3 text-right">Var. Resultados</th>
                                <th className="px-6 py-3">Situação Observada</th>
                              </tr>
                            </thead>
                            <tbody>
                              {dfWarning.map((stock) => (
                                <tr key={stock.papel} className="border-b border-slate-100 hover:bg-red-50/10 transition-colors font-medium text-slate-700">
                                  <td className="px-6 py-3.5 font-bold text-slate-950 font-mono">{stock.papel}</td>
                                  <td className="px-6 py-3.5 text-slate-400">{stock.empresa}</td>
                                  <td className="px-6 py-3.5 text-right font-mono">{fmtCurrency(stock.cotacao)}</td>
                                  <td className="px-6 py-3.5 text-right font-mono text-red-600 font-bold">{stock.divbpatr.toFixed(2)}</td>
                                  <td className="px-6 py-3.5 text-right font-mono text-red-500 font-medium">{stock.quedaLucro}</td>
                                  <td className="px-6 py-3.5">
                                    <span className={`inline-flex px-2 py-0.5 rounded text-[10px] font-bold ${
                                      stock.situacao === 'Recup. Judicial' ? 'bg-red-100 text-red-800 border border-red-200' : 'bg-amber-100 text-amber-800 border border-amber-200'
                                    }`}>
                                      {stock.situacao}
                                    </span>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>

                      {/* interactive select and chart */}
                      <div className="flex flex-col gap-4">
                        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                          <div>
                            <h3 className="font-bold text-slate-800 text-base">📈 Gráfico Cotação vs Lucro/Receita</h3>
                            <p className="text-xs text-slate-400">Selecione qualquer ativo monitorado para renderizar o gráfico histórico</p>
                          </div>
                          
                          <div className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-2xl border border-slate-200 shadow-sm w-full sm:w-auto">
                            <span className="text-xs font-bold text-slate-400 shrink-0 uppercase tracking-wider font-mono">Papel:</span>
                            <select
                              id="historical-chart-select"
                              value={selectedChartTicker}
                              onChange={(e) => setSelectedChartTicker(e.target.value)}
                              className="w-full bg-transparent font-mono font-bold text-slate-800 text-xs focus:outline-none outline-none border-none py-1 cursor-pointer"
                            >
                              {STOCK_DATABASE.map(s => (
                                <option key={s.papel} value={s.papel}>
                                  {s.papel} - {s.empresa}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <StockChart data={activeChartData} ticker={selectedChartTicker} />
                      </div>

                      {/* News and Dividends Grid bottom */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-4">
                        
                        {/* News Frame */}
                        <div className="bg-white border border-slate-200/80 rounded-3xl p-6 shadow-sm flex flex-col gap-4">
                          <h4 className="font-bold text-slate-900 text-sm flex items-center gap-2">
                            <Newspaper className="w-4 h-4 text-blue-500" />
                            📰 Notícias Recentes do Mercado Financeiro
                          </h4>
                          <div className="flex flex-col gap-3">
                            {MARKET_NEWS.map((news, i) => (
                              <a
                                key={i}
                                href={news.link}
                                target="_blank"
                                rel="noreferrer"
                                className="group flex flex-col gap-1 p-3 rounded-xl border border-slate-100 hover:border-blue-200 hover:bg-blue-50/10 transition-all"
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <span className="font-bold font-mono text-[9px] text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded uppercase">
                                    {news.source}
                                  </span>
                                  <ExternalLink className="w-3 h-3 text-slate-300 group-hover:text-blue-500 transition-colors" />
                                </div>
                                <h5 className="font-bold text-slate-800 text-xs leading-snug group-hover:text-blue-900 transition-colors">
                                  {news.title}
                                </h5>
                              </a>
                            ))}
                          </div>
                        </div>

                        {/* Recent Dividends */}
                        <div className="bg-white border border-slate-200/80 rounded-3xl p-6 shadow-sm flex flex-col gap-4">
                          <h4 className="font-bold text-slate-900 text-sm flex items-center gap-2">
                            <Coins className="w-4 h-4 text-emerald-500" />
                            💰 Últimos Dividendos Anunciados (B3)
                          </h4>
                          <div className="flex flex-col gap-3">
                            {LATEST_DIVIDENDS.map((div, i) => (
                              <div
                                key={i}
                                className="flex items-center justify-between p-3.5 rounded-xl border border-slate-100 bg-slate-50/30"
                              >
                                <div className="flex items-center gap-2">
                                  <div className="h-8 w-8 rounded-lg bg-emerald-50 text-emerald-600 font-mono font-bold text-xs flex items-center justify-center">
                                    {div.ativo}
                                  </div>
                                  <div className="flex flex-col">
                                    <span className="font-bold text-slate-800 text-xs">Ação Preferencial/Ordinária</span>
                                    <span className="text-[10px] text-slate-400">Data ex: {new Date(div.data).toLocaleDateString('pt-BR')}</span>
                                  </div>
                                </div>
                                <span className="font-mono text-emerald-600 font-black text-xs">
                                  R$ {div.valor.toFixed(4)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>

                      </div>

                    </div>
                  )}

                  {/* TAB 2: FÓRMULA MÁGICA */}
                  {activeTab === 'magic' && (
                    <div className="flex flex-col gap-8" id="tab-view-magic">
                      
                      {/* Explainer */}
                      <div className="bg-white border border-slate-200/80 rounded-3xl p-6 shadow-sm flex flex-col gap-4">
                        <div className="flex items-center gap-2 text-indigo-600 font-bold text-xs uppercase font-mono">
                          <Sparkles className="w-4 h-4" />
                          <span>Joel Greenblatt Magic Formula</span>
                        </div>
                        <h3 className="font-bold text-slate-800 text-lg">Metodologia Tática: Fórmula Mágica</h3>
                        <p className="text-slate-500 text-sm leading-relaxed">
                          Criada por Joel Greenblatt, esta metodologia rankeia ações com liquidez significativa e foca em encontrar empresas baratas com alta qualidade de retorno.
                        </p>
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-2">
                          <div className="bg-slate-50 border border-slate-150 p-4 rounded-2xl flex flex-col gap-1">
                            <span className="text-[10px] text-slate-400 font-semibold uppercase">1. Filtro base</span>
                            <span className="text-xs font-bold text-slate-800">Liquidez diária &gt; R$ 100k e EV/EBIT positivo</span>
                          </div>
                          <div className="bg-slate-50 border border-slate-150 p-4 rounded-2xl flex flex-col gap-1">
                            <span className="text-[10px] text-slate-400 font-semibold uppercase">2. Rank EY</span>
                            <span className="text-xs font-bold text-slate-800">Earning Yield (EBIT / Valor da Empresa) - Maior para Menor</span>
                          </div>
                          <div className="bg-slate-50 border border-slate-150 p-4 rounded-2xl flex flex-col gap-1">
                            <span className="text-[10px] text-slate-400 font-semibold uppercase">3. Rank ROIC</span>
                            <span className="text-xs font-bold text-slate-800">Retorno sobre capital investido - Maior para Menor</span>
                          </div>
                          <div className="bg-slate-50 border border-slate-150 p-4 rounded-2xl flex flex-col gap-1">
                            <span className="text-[10px] text-slate-400 font-semibold uppercase">4. Score Mágico</span>
                            <span className="text-xs font-bold text-slate-800">Rank EY + Rank ROIC. Menor pontuação lidera o topo!</span>
                          </div>
                        </div>
                      </div>

                      {/* Magic Table */}
                      <div className="bg-white border border-slate-200/80 rounded-3xl shadow-sm overflow-hidden flex flex-col" id="magic-table-card">
                        <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
                          <h4 className="font-bold text-slate-900 text-sm">
                            ✨ Lista Top 40 Fórmula Mágica Ibovespa
                          </h4>
                          <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-amber-100 text-amber-800 uppercase">Classificado por Score Mágico</span>
                        </div>
                        <div className="overflow-x-auto">
                          <table className="w-full text-left border-collapse text-xs">
                            <thead>
                              <tr className="bg-slate-50/70 border-b border-slate-200 text-slate-400 uppercase tracking-wider font-mono font-bold text-[10px]">
                                <th className="px-6 py-3">Ativo</th>
                                <th className="px-6 py-3 text-right">Preço</th>
                                <th className="px-6 py-3 text-right">Score Mágico (DY + ROIC)</th>
                                <th className="px-6 py-3 text-right">Rank EY</th>
                                <th className="px-6 py-3 text-right">Rank ROIC</th>
                                <th className="px-6 py-3 text-right">Earning Yield (1 / EV/EBIT)</th>
                                <th className="px-6 py-3 text-right">Ev/Ebit</th>
                                <th className="px-6 py-3 text-right">ROIC (%)</th>
                              </tr>
                            </thead>
                            <tbody>
                              {dfMagic.map((stock, i) => {
                                // Background highlight gradient
                                let scoreColor = 'bg-slate-50 text-slate-800';
                                if (stock.scoreMagico < 10) scoreColor = 'bg-emerald-50 text-emerald-800 border-l-4 border-emerald-500';
                                else if (stock.scoreMagico < 25) scoreColor = 'bg-blue-50 text-blue-800';
                                else if (stock.scoreMagico >= 35) scoreColor = 'bg-red-50 text-red-800';

                                return (
                                  <tr key={stock.papel} className="border-b border-slate-100 hover:bg-slate-50/50 transition-colors font-medium text-slate-700">
                                    <td className="px-6 py-3.5 font-bold text-slate-900 font-mono flex items-center gap-1.5">
                                      <span className="text-slate-300 font-mono text-[10px] w-4">{i + 1}</span>
                                      {stock.papel}
                                    </td>
                                    <td className="px-6 py-3.5 text-right font-mono">{fmtCurrency(stock.cotacao)}</td>
                                    <td className={`px-6 py-3.5 text-right font-mono font-black ${scoreColor}`}>
                                      {stock.scoreMagico} pts
                                    </td>
                                    <td className="px-6 py-3.5 text-right font-mono text-slate-500">{stock.rankEy}º</td>
                                    <td className="px-6 py-3.5 text-right font-mono text-slate-500">{stock.rankRoic}º</td>
                                    <td className="px-6 py-3.5 text-right font-mono text-indigo-600 font-semibold">{fmtPct(stock.earningYield)}</td>
                                    <td className="px-6 py-3.5 text-right font-mono">{stock.evebit.toFixed(2)}</td>
                                    <td className="px-6 py-3.5 text-right font-mono text-slate-900">{fmtPct(stock.roic)}</td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>

                    </div>
                  )}

                  {/* TAB 3: GRAHAM */}
                  {activeTab === 'graham' && (
                    <div className="flex flex-col gap-8" id="tab-view-graham">
                      
                      {/* Description */}
                      <div className="bg-white border border-slate-200/80 rounded-3xl p-6 shadow-sm flex flex-col gap-3">
                        <div className="flex items-center gap-2 text-amber-600 font-bold text-xs uppercase font-mono">
                          <DollarSign className="w-4 h-4" />
                          <span>Benjamin Graham Valuation Model</span>
                        </div>
                        <h3 className="font-bold text-slate-800 text-lg">Modelo Graham Clássico de Preço Justo</h3>
                        <p className="text-slate-500 text-sm leading-relaxed">
                          A famosa equação de Graham de Valor Intrínseco é calculada como: <code className="bg-slate-100 px-2 py-0.5 rounded font-mono font-bold text-slate-800 text-xs">V.I = Raiz(22.5 * LPA * VPA)</code>.
                        </p>
                        <ul className="text-xs text-slate-500 list-disc pl-5 mt-1 leading-relaxed flex flex-col gap-1">
                          <li><strong>LPA (Lucro por ação):</strong> Quanto lucro a empresa gerou para cada ação emitida.</li>
                          <li><strong>VPA (Valor Patrimonial por ação):</strong> Qual o valor contábil líquido associado àquela ação.</li>
                          <li><strong>Upside/Downside:</strong> Se for superior a 1.0, o preço justo está acima da cotação, indicando margem de segurança física para compras <strong>(Barata)</strong>.</li>
                        </ul>
                      </div>

                      {/* Graham Table list */}
                      <div className="bg-white border border-slate-200/80 rounded-3xl shadow-sm overflow-hidden flex flex-col" id="graham-table-card">
                        <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
                          <h4 className="font-bold text-slate-900 text-sm">
                            💎 Avaliação de Margem de Segurança Graham
                          </h4>
                          <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-blue-100 text-blue-800 uppercase">Ordenados por Maior Margem</span>
                        </div>
                        <div className="overflow-x-auto">
                          <table className="w-full text-left border-collapse text-xs">
                            <thead>
                              <tr className="bg-slate-50/70 border-b border-slate-200 text-slate-400 uppercase tracking-wider font-mono font-bold text-[10px]">
                                <th className="px-6 py-3">Ativo</th>
                                <th className="px-6 py-3 text-right">Preço Atual</th>
                                <th className="px-6 py-3 text-right">Preço Justo (Graham)</th>
                                <th className="px-6 py-3 text-right">Upside Margem (V.I / Preço)</th>
                                <th className="px-6 py-3">Status</th>
                                <th className="px-6 py-3 text-right">LPA</th>
                                <th className="px-6 py-3 text-right">VPA</th>
                              </tr>
                            </thead>
                            <tbody>
                              {dfGraham.map((stock) => {
                                const isBarata = stock.status === 'Barata';
                                return (
                                  <tr key={stock.papel} className="border-b border-slate-100 hover:bg-slate-50/50 transition-colors font-medium text-slate-700">
                                    <td className="px-6 py-3.5 font-bold text-slate-950 font-mono">{stock.papel}</td>
                                    <td className="px-6 py-3.5 text-right font-mono">{fmtCurrency(stock.cotacao)}</td>
                                    <td className="px-6 py-3.5 text-right font-mono font-bold text-slate-900">{fmtCurrency(stock.valorIntrinseco)}</td>
                                    <td className="px-6 py-3.5 text-right font-mono text-purple-700 font-bold bg-purple-50/25">
                                      {stock.ratio.toFixed(2)}x
                                    </td>
                                    <td className="px-6 py-3.5">
                                      <span className={`inline-flex px-2 py-0.5 rounded text-[10px] font-bold ${
                                        isBarata ? 'bg-emerald-100 text-emerald-800' : 'bg-red-100 text-red-800'
                                      }`}>
                                        {isBarata ? '✓ Barata (Desconto)' : 'Valor Esticado'}
                                      </span>
                                    </td>
                                    <td className="px-6 py-3.5 text-right font-mono text-slate-500">{fmtCurrency(stock.lpa)}</td>
                                    <td className="px-6 py-3.5 text-right font-mono text-slate-500">{fmtCurrency(stock.vpa)}</td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>

                    </div>
                  )}

                  {/* TAB 4: EPS DILUÍDO */}
                  {activeTab === 'eps' && (
                    <div className="flex flex-col gap-8" id="tab-view-eps">
                      
                      {/* Explainer Expander */}
                      <div className="bg-white border border-slate-200/80 rounded-3xl p-6 shadow-sm flex flex-col gap-3">
                        <div className="flex items-center gap-1.5 text-blue-600">
                          <Info className="w-4 h-4" />
                          <h4 className="font-bold text-slate-800 text-sm">O que é EPS Diluído (LPA Trimestral)?</h4>
                        </div>
                        <p className="text-slate-600 text-xs leading-relaxed">
                          É uma métrica financeira que mede o valor do lucro líquido de uma empresa atribuído a cada ação corporativa em circulação, deduzida toda a diluição possível decorrente de concessões de bônus, conversões de debêntures ou opções de novas ações. Valores elevados no trimestre mais recente indicam forte capacidade de geração de lucro.
                        </p>
                      </div>

                      {/* Side by side stats and TV widget */}
                      <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
                        
                        {/* Highlights list on side */}
                        <div className="lg:col-span-2 bg-white border border-slate-200/80 rounded-3xl shadow-sm overflow-hidden flex flex-col">
                          <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
                            <h4 className="font-bold text-slate-900 text-xs">
                              📈 EPS Líquido Trimestral &gt; R$ 1,00
                            </h4>
                          </div>
                          <div className="p-4 flex-1 flex flex-col gap-3">
                            {dfEps.map((stock) => (
                              <div
                                key={stock.papel}
                                className="flex items-center justify-between p-3 rounded-2xl border border-slate-100 hover:border-slate-200 hover:bg-slate-50 bg-slate-50/10 transition-all font-sans"
                              >
                                <div className="flex items-center gap-2.5">
                                  <div className="h-8 w-8 rounded-xl bg-blue-50 text-blue-600 font-mono font-bold text-xs flex items-center justify-center">
                                    {stock.papel}
                                  </div>
                                  <div>
                                    <h5 className="font-bold text-slate-800 text-xs">{stock.empresa}</h5>
                                    <span className="text-[10px] text-slate-400 font-mono">Ref: {stock.dataRef}</span>
                                  </div>
                                </div>
                                <div className="text-right">
                                  <span className="font-mono text-emerald-600 font-bold text-xs block">
                                    {fmtCurrency(stock.epsTrimestral)}
                                  </span>
                                  <span className="text-[9px] text-slate-400">Preço: {fmtCurrency(stock.cotacao)}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Chart embed space */}
                        <div className="lg:col-span-3 flex flex-col gap-4">
                          <TradingViewWidget symbol="PETR4" />
                        </div>

                      </div>

                    </div>
                  )}

                  {/* TAB 5: ASSIMETRIA */}
                  {activeTab === 'asymmetry' && (
                    <div className="flex flex-col gap-8" id="tab-view-asymmetry">
                      
                      {/* Explainer */}
                      <div className="bg-white border border-slate-200/80 rounded-3xl p-6 shadow-sm flex flex-col gap-2">
                        <div className="flex items-center gap-2 text-green-600 font-bold text-xs uppercase font-mono">
                          <TrendingUp className="w-4 h-4" />
                          <span>Filtro de Assimetria Positiva</span>
                        </div>
                        <h3 className="font-bold text-slate-800 text-lg">📈 Assimetria Visual: Lucros acima do Preço</h3>
                        <p className="text-slate-500 text-sm leading-relaxed">
                          Este robô computacional varre as empresas mais líquidas em busca de um padrão assimétrico de ouro: <strong>empresas onde a direção dos lucros líquidos subiu com vigor acumulado, porém a cotação do papel na bolsa permaneceu atrasada ou em tendência oposta</strong>.
                        </p>
                      </div>

                      {/* Asymmetry select and render */}
                      <div className="flex flex-col gap-4">
                        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                          <div>
                            <h4 className="font-bold text-slate-800 text-base">Selecione Ativo com Assimetria Detectada:</h4>
                            <p className="text-xs text-slate-400">Encontradas {asymmetryStocks.length} ações qualificadas no último lote de cálculos.</p>
                          </div>

                          <div className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-2xl border border-slate-200 shadow-sm w-full sm:w-auto">
                            <span className="text-xs font-bold text-slate-400 shrink-0 uppercase tracking-wider font-mono">Ativo Seguro:</span>
                            <select
                              id="asymmetry-chart-select"
                              value={selectedAsymmetryTicker}
                              onChange={(e) => setSelectedAsymmetryTicker(e.target.value)}
                              className="w-full bg-transparent font-mono font-bold text-slate-800 text-xs focus:outline-none outline-none border-none py-1 cursor-pointer"
                            >
                              {asymmetryStocks.map(s => (
                                <option key={s.papel} value={s.papel}>
                                  {s.papel} - {s.empresa}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        {/* Historical Graph */}
                        <StockChart data={activeAsymmetryChartData} ticker={selectedAsymmetryTicker} />

                        <div className="bg-emerald-50 border border-emerald-150 p-4 rounded-3xl flex items-center gap-3">
                          <span className="text-lg">✅</span>
                          <span className="text-xs text-emerald-800 font-sans leading-relaxed">
                            <strong>Sinal Confirmado:</strong> A linha pontilhada verde (Lucro Líquido nos últimos 12m) termina visualmente acima da linha azul contínua (Cotação de Fechamento do ponto) garantindo uma grande assimetria positiva de segurança!
                          </span>
                        </div>
                      </div>

                    </div>
                  )}

                  {/* Foot ads affiliate simulated footer exactly like the streamlit code */}
                  <div className="mt-8 pt-8 border-t border-slate-200">
                    <h4 className="font-bold text-slate-600 text-xs uppercase tracking-wider text-center mb-6">Nossos Membros Patrocinadores Oficiais</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 leading-relaxed">
                      
                      {/* Affiliate 1 */}
                      <a
                        href="https://nomad.onelink.me/wIQT/Invest?code=Y39FP3XF8I&n=Jader"
                        target="_blank"
                        rel="noreferrer"
                        className="bg-white border border-slate-200/85 hover:border-amber-300 p-5 rounded-3xl flex flex-col justify-between gap-4 hover:shadow-md transition-all group"
                      >
                        <div className="flex flex-col gap-1.5 leading-snug">
                          <div className="flex items-center gap-2">
                            <span className="text-lg">✈️</span>
                            <h4 className="font-bold text-sm text-slate-800 group-hover:text-amber-700 transition-colors">Nomad: Conta em Dólar Gratuita</h4>
                          </div>
                          <p className="text-xs text-slate-500 leading-normal">
                            Abra sua conta bancária e de investimentos global nos EUA sem taxas mensais de manutenção. Use o convênio e garanta taxa cambial zero na primeira conversão de saldo!
                          </p>
                        </div>
                        <div className="text-xs font-semibold text-amber-700 bg-amber-50 rounded-xl px-4 py-2 text-center group-hover:bg-amber-100 transition-all font-mono">
                          Abrir Conta Internacional Nomad →
                        </div>
                      </a>

                      {/* Affiliate 2 */}
                      <a
                        href="https://mpago.li/1VydVhw"
                        target="_blank"
                        rel="noreferrer"
                        className="bg-white border border-slate-200/85 hover:border-sky-300 p-5 rounded-3xl flex flex-col justify-between gap-4 hover:shadow-md transition-all group"
                      >
                        <div className="flex flex-col gap-1.5 leading-snug">
                          <div className="flex items-center gap-2">
                            <span className="text-lg">🤝</span>
                            <h4 className="font-bold text-sm text-slate-800 group-hover:text-blue-700 transition-colors">Mercado Pago: R$ 30,00 Grátis de Desconto</h4>
                          </div>
                          <p className="text-xs text-slate-500 leading-normal">
                            Crie sua conta digital líder em rendimentos e maquininhas Point no Mercado Pago para receber pagamentos e conquiste um bônus de boas-vindas especial de trinta reais!
                          </p>
                        </div>
                        <div className="text-xs font-semibold text-blue-700 bg-sky-50 rounded-xl px-4 py-2 text-center group-hover:bg-sky-100 transition-all font-mono font-bold">
                          Resgatar Bônus Mercado Pago R$ 30 →
                        </div>
                      </a>

                    </div>
                  </div>

                </motion.div>
              </AnimatePresence>
            )}

          </div>

        </main>
      </div>

      {/* Main Footer under entire frame */}
      <footer className="w-full bg-slate-900 text-slate-400 py-10 border-t border-slate-800 text-xs">
        <div className="max-w-7xl mx-auto px-6 md:px-8 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex flex-col gap-1 text-center md:text-left">
            <h5 className="text-slate-200 font-bold font-sans">Ibovespa Fundamentalista © 2026</h5>
            <p className="text-slate-500">Desenvolvido com maestria em React, TypeScript e Tailwind CSS para emulação perfeita do app.py.</p>
          </div>
          <div className="flex items-center gap-6">
            <span className="hover:text-white transition-colors cursor-pointer">Termos de Uso</span>
            <span className="hover:text-white transition-colors cursor-pointer">Políticas Gerais</span>
            <span className="hover:text-slate-300 cursor-default bg-slate-800 px-2.5 py-1 rounded text-[10px] font-mono">B3 IB_V_2025</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
