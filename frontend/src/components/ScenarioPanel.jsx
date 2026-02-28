import React, { useState } from 'react';
import { SlidersHorizontal, AlertTriangle } from 'lucide-react';

const ScenarioPanel = () => {
    const [marketDrop, setMarketDrop] = useState(0);
    const [assetDrop, setAssetDrop] = useState(0);

    return (
        <div className="glass-panel rounded-xl p-5 h-full flex flex-col border-l-2 border-l-cyan-500/50">
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                    <SlidersHorizontal size={16} className="text-cyan-400" />
                    Stress Scenarios
                </h2>
                <span className="flex h-2 w-2 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
                </span>
            </div>

            <div className="space-y-6 flex-grow">
                {/* Scenario 1 */}
                <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700/50">
                    <div className="flex justify-between items-end mb-3">
                        <label className="text-xs font-semibold text-slate-300">NIFTY 50 Shock</label>
                        <span className="text-sm font-mono text-rose-400 font-bold">-{marketDrop}%</span>
                    </div>
                    <input
                        type="range"
                        min="0" max="30"
                        value={marketDrop}
                        onChange={(e) => setMarketDrop(e.target.value)}
                        className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
                        <span>0%</span>
                        <span>-15%</span>
                        <span>-30%</span>
                    </div>
                </div>

                {/* Scenario 2 */}
                <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700/50">
                    <div className="flex justify-between items-end mb-3">
                        <label className="text-xs font-semibold text-slate-300">Asset Shock: RELIANCE</label>
                        <span className="text-sm font-mono text-rose-400 font-bold">-{assetDrop}%</span>
                    </div>
                    <input
                        type="range"
                        min="0" max="40"
                        value={assetDrop}
                        onChange={(e) => setAssetDrop(e.target.value)}
                        className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-500"
                    />
                </div>
            </div>

            {/* Recompute Button & Warning */}
            <div className="mt-6">
                {(marketDrop > 10 || assetDrop > 15) ? (
                    <div className="mb-3 flex items-start gap-2 bg-rose-950/40 border border-rose-900/50 p-3 rounded-md text-xs text-rose-300">
                        <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                        <p>Significant tail scenario detected. Portfolio VaR limit may be breached.</p>
                    </div>
                ) : null}

                <button className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-semibold rounded-lg transition-all shadow-[0_0_15px_rgba(8,145,178,0.4)] hover:shadow-[0_0_25px_rgba(8,145,178,0.6)]">
                    Recalculate Bounds
                </button>
            </div>
        </div>
    );
};

export default ScenarioPanel;
