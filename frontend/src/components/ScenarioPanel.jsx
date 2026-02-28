import React, { useState } from 'react';
import { SlidersHorizontal, AlertTriangle } from 'lucide-react';

const ScenarioPanel = () => {
    const [marketDrop, setMarketDrop] = useState(0);
    const [assetDrop, setAssetDrop] = useState(0);

    return (
        <div className="glass-panel rounded-xl p-5 h-full flex flex-col border-l-2 border-l-cyan-500/30 relative">
            <div className="flex justify-between items-center mb-6 z-10">
                <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                    <SlidersHorizontal size={16} className="text-cyan-400" />
                    Stress Scenarios
                </h2>
                <span className="flex h-2 w-2 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.8)]"></span>
                </span>
            </div>

            <div className="space-y-6 flex-grow z-10">
                {/* Scenario 1 */}
                <div className="bg-slate-900/40 p-4 rounded-lg border border-slate-700/30 shadow-inner">
                    <div className="flex justify-between items-end mb-3">
                        <label className="text-xs font-semibold text-slate-300">NIFTY 50 Shock</label>
                        <span className="text-sm font-mono text-rose-400 font-bold">-{marketDrop}%</span>
                    </div>
                    <input
                        type="range"
                        min="0" max="30"
                        value={marketDrop}
                        onChange={(e) => setMarketDrop(e.target.value)}
                        className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500 hover:accent-cyan-400 transition-all focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 mt-2 font-mono px-1">
                        <span>0%</span>
                        <span>-15%</span>
                        <span>-30%</span>
                    </div>
                </div>

                {/* Scenario 2 */}
                <div className="bg-slate-900/40 p-4 rounded-lg border border-slate-700/30 shadow-inner">
                    <div className="flex justify-between items-end mb-3">
                        <label className="text-xs font-semibold text-slate-300">Asset Shock: RELIANCE</label>
                        <span className="text-sm font-mono text-rose-400 font-bold">-{assetDrop}%</span>
                    </div>
                    <input
                        type="range"
                        min="0" max="40"
                        value={assetDrop}
                        onChange={(e) => setAssetDrop(e.target.value)}
                        className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-rose-500 hover:accent-rose-400 transition-all focus:outline-none focus:ring-2 focus:ring-rose-500/50"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 mt-2 font-mono px-1">
                        <span>0%</span>
                        <span>-20%</span>
                        <span>-40%</span>
                    </div>
                </div>
            </div>

            {/* Recompute Button & Warning */}
            <div className="mt-6 z-10">
                <div className={`overflow-hidden transition-all duration-500 ease-in-out ${marketDrop > 10 || assetDrop > 15 ? 'max-h-24 opacity-100 mb-4' : 'max-h-0 opacity-0 mb-0'}`}>
                    <div className="flex items-start gap-3 bg-gradient-to-r from-rose-950/60 to-slate-900/40 border border-rose-900/40 p-3 rounded-lg text-xs shadow-[inset_0_1px_0_rgba(255,255,255,0.05),0_0_15px_rgba(225,29,72,0.15)] relative overflow-hidden">
                        {/* Soft background glow */}
                        <div className="absolute top-0 left-0 w-full h-full bg-rose-500/5 blur-xl pointer-events-none"></div>
                        <AlertTriangle className="w-5 h-5 flex-shrink-0 text-rose-400 mt-0.5 relative z-10" />
                        <div className="relative z-10">
                            <p className="font-semibold text-rose-300 mb-0.5 tracking-wide">Tail Risk Elevated</p>
                            <p className="text-slate-400">Significant stress detected. VaR limit breach likely under this scenario.</p>
                        </div>
                    </div>
                </div>

                <button className="w-full py-3 bg-cyan-600/90 hover:bg-cyan-500 text-white text-sm font-semibold rounded-lg smooth-transition shadow-[0_0_15px_rgba(8,145,178,0.3)] hover:shadow-[0_0_25px_rgba(8,145,178,0.6)] border border-cyan-500/50 hover:border-cyan-400">
                    Recalculate Bounds
                </button>
            </div>

            {/* Background subtle gradient for entire panel based on slider values */}
            <div
                className="absolute inset-0 pointer-events-none rounded-xl transition-opacity duration-700 ease-in-out mix-blend-screen"
                style={{
                    background: `radial-gradient(circle at bottom right, rgba(225, 29, 72, ${(parseInt(marketDrop) + parseInt(assetDrop)) * 0.003}) 0%, transparent 70%)`
                }}
            />
        </div>
    );
};

export default ScenarioPanel;
