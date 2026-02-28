import React, { useState, useEffect } from 'react';
import { SlidersHorizontal, AlertTriangle, RefreshCw } from 'lucide-react';

const ScenarioPanel = ({ data }) => {
    const [marketDrop, setMarketDrop] = useState(0);
    const [assetDrop, setAssetDrop] = useState(0);
    const [results, setResults] = useState(null);
    const [isCalculating, setIsCalculating] = useState(false);

    // Clear results when sliders change (stale indicator)
    useEffect(() => { setResults(null); }, [marketDrop, assetDrop]);

    const handleRecalculate = () => {
        if (!data) return;

        setIsCalculating(true);

        // Slight delay so the spinner is visible
        setTimeout(() => {
            const beta = data.beta ?? 1.0;
            const baseVar = data.monte_carlo_var_95 ?? 0;    // negative decimal, e.g. -0.018
            const portVol = data.portfolio_volatility ?? 0;
            const portReturn = data.portfolio_expected_return ?? 0;

            // Market shock effect on portfolio via Beta: ΔP = β × market_drop
            const mktShockDecimal = -(parseInt(marketDrop) / 100);
            const assetShockDecimal = -(parseInt(assetDrop) / 100);

            // Stressed expected daily return (scaled from annual shock to daily)
            // Assume the shock is the sharpest 1-day move
            const stressedReturn = portReturn
                + beta * mktShockDecimal
                + assetShockDecimal * 0.3; // partial single-asset contribution

            // Stressed VaR: base VaR + market shock component + vol adjustment
            const stressedVar = baseVar
                + beta * mktShockDecimal
                + portVol * Math.sqrt(Math.abs(mktShockDecimal + assetShockDecimal)) * -0.5;

            // Portfolio P&L impact as % change
            const portfolioImpact = (beta * mktShockDecimal + assetShockDecimal * 0.25) * 100;

            // Risk level classification
            const totalShock = parseInt(marketDrop) + parseInt(assetDrop);
            const riskLevel = totalShock < 15 ? 'low' : totalShock < 35 ? 'moderate' : 'extreme';

            setResults({ stressedVar, stressedReturn, portfolioImpact, riskLevel });
            setIsCalculating(false);
        }, 400);
    };

    const hasStress = parseInt(marketDrop) > 10 || parseInt(assetDrop) > 15;
    const hasData = !!data;
    const assetLabel = data?.asset_names?.[0] ?? 'Top Asset';

    const riskColors = {
        low: 'text-yellow-400 border-yellow-600/40 bg-yellow-950/30',
        moderate: 'text-orange-400 border-orange-600/40 bg-orange-950/30',
        extreme: 'text-rose-400 border-rose-600/40 bg-rose-950/30',
    };

    return (
        <div className="glass-panel rounded-xl p-5 h-full flex flex-col border-l-2 border-l-cyan-500/30 relative">
            {/* Header */}
            <div className="flex justify-between items-center mb-5 z-10">
                <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                    <SlidersHorizontal size={16} className="text-cyan-400" />
                    Stress Scenarios
                </h2>
                <span className="flex h-2 w-2 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.8)]"></span>
                </span>
            </div>

            <div className="space-y-4 flex-grow z-10">
                {/* Scenario 1: Market Shock */}
                <div className="bg-slate-900/40 p-4 rounded-lg border border-slate-700/30 shadow-inner">
                    <div className="flex justify-between items-end mb-3">
                        <label className="text-xs font-semibold text-slate-300">Market Index Shock</label>
                        <span className="text-sm font-mono text-rose-400 font-bold">-{marketDrop}%</span>
                    </div>
                    <input
                        type="range"
                        min="0" max="30"
                        value={marketDrop}
                        onChange={(e) => setMarketDrop(e.target.value)}
                        className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500 transition-all focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 mt-1.5 font-mono px-1">
                        <span>0%</span><span>-15%</span><span>-30%</span>
                    </div>
                </div>

                {/* Scenario 2: Single Asset Shock */}
                <div className="bg-slate-900/40 p-4 rounded-lg border border-slate-700/30 shadow-inner">
                    <div className="flex justify-between items-end mb-3">
                        <label className="text-xs font-semibold text-slate-300">
                            Asset Shock: <span className="text-cyan-400 font-mono">{assetLabel}</span>
                        </label>
                        <span className="text-sm font-mono text-rose-400 font-bold">-{assetDrop}%</span>
                    </div>
                    <input
                        type="range"
                        min="0" max="40"
                        value={assetDrop}
                        onChange={(e) => setAssetDrop(e.target.value)}
                        className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-rose-500 transition-all focus:outline-none focus:ring-2 focus:ring-rose-500/50"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 mt-1.5 font-mono px-1">
                        <span>0%</span><span>-20%</span><span>-40%</span>
                    </div>
                </div>

                {/* Results Block */}
                {results && (
                    <div className={`p-3 rounded-lg border text-xs font-mono space-y-2 ${riskColors[results.riskLevel]}`}>
                        <p className="text-[10px] uppercase tracking-widest font-bold opacity-70 mb-1">Stressed Estimates</p>
                        <div className="flex justify-between">
                            <span className="text-slate-400">Portfolio Impact</span>
                            <span className="font-bold">{results.portfolioImpact.toFixed(2)}%</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-400">Stressed VaR (95%)</span>
                            <span className="font-bold">{(results.stressedVar * 100).toFixed(2)}%</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-400">Scenario Risk</span>
                            <span className="font-bold uppercase">{results.riskLevel}</span>
                        </div>
                    </div>
                )}

                {/* Tail Risk Warning */}
                <div className={`overflow-hidden transition-all duration-500 ${hasStress && results ? 'max-h-24 opacity-100' : 'max-h-0 opacity-0'}`}>
                    <div className="flex items-start gap-3 bg-gradient-to-r from-rose-950/60 to-slate-900/40 border border-rose-900/40 p-3 rounded-lg text-xs">
                        <AlertTriangle className="w-4 h-4 flex-shrink-0 text-rose-400 mt-0.5" />
                        <div>
                            <p className="font-semibold text-rose-300 mb-0.5">Tail Risk Elevated</p>
                            <p className="text-slate-400">VaR breach likely under this stress scenario.</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Recalculate Button */}
            <div className="mt-4 z-10">
                {!hasData && (
                    <p className="text-[10px] text-slate-500 text-center mb-2 font-mono">Upload portfolio to enable stress testing</p>
                )}
                <button
                    onClick={handleRecalculate}
                    disabled={!hasData || isCalculating}
                    className={`w-full py-3 text-white text-sm font-semibold rounded-lg smooth-transition border flex items-center justify-center gap-2
                        ${hasData
                            ? 'bg-cyan-600/90 hover:bg-cyan-500 shadow-[0_0_15px_rgba(8,145,178,0.3)] hover:shadow-[0_0_25px_rgba(8,145,178,0.6)] border-cyan-500/50 hover:border-cyan-400 cursor-pointer'
                            : 'bg-slate-700/40 border-slate-700/30 text-slate-500 cursor-not-allowed'
                        }`}
                >
                    <RefreshCw size={14} className={isCalculating ? 'animate-spin' : ''} />
                    {isCalculating ? 'Calculating...' : 'Recalculate Bounds'}
                </button>
            </div>

            {/* Background glow */}
            <div
                className="absolute inset-0 pointer-events-none rounded-xl transition-opacity duration-700 mix-blend-screen"
                style={{ background: `radial-gradient(circle at bottom right, rgba(225, 29, 72, ${(parseInt(marketDrop) + parseInt(assetDrop)) * 0.003}) 0%, transparent 70%)` }}
            />
        </div>
    );
};

export default ScenarioPanel;
