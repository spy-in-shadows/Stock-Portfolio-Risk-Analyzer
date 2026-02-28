import React, { useState, useEffect } from 'react';
import { UploadCloud, CheckCircle2, FileSpreadsheet } from 'lucide-react';

const UploadPortfolio = () => {
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        // Simulate loading state for 1.5 seconds to show the skeleton shimmer
        const timer = setTimeout(() => setIsLoading(false), 1500);
        return () => clearTimeout(timer);
    }, []);

    if (isLoading) {
        return (
            <div className="glass-panel rounded-xl p-5 h-full flex flex-col">
                <div className="w-32 h-4 skeleton-shimmer rounded mb-6"></div>
                <div className="w-full h-32 skeleton-shimmer rounded-lg mb-4"></div>
                <div className="w-full h-12 skeleton-shimmer rounded-lg"></div>
            </div>
        );
    }

    return (
        <div className="glass-panel rounded-xl p-5 h-full flex flex-col relative overflow-hidden group">
            <h2 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wider relative z-10">Portfolio Data</h2>

            <div className="border-2 border-dashed border-slate-700/50 bg-slate-900/40 rounded-lg p-6 flex flex-col items-center justify-center flex-grow transition-all duration-300 hover:border-cyan-500/50 hover:bg-slate-800/60 cursor-pointer relative z-10 hover:shadow-[0_0_30px_rgba(8,145,178,0.15)]">
                <div className="relative">
                    <div className="absolute inset-0 bg-cyan-500/20 blur-xl rounded-full scale-150 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                    <div className="p-3 bg-slate-800/80 rounded-full mb-3 group-hover:bg-cyan-900/50 transition-colors border border-slate-700 group-hover:border-cyan-700 relative z-10">
                        <UploadCloud className="w-6 h-6 text-slate-400 group-hover:text-cyan-400 transition-colors" />
                    </div>
                </div>
                <p className="text-sm font-medium text-slate-200 text-center">Drag & drop CSV</p>
                <div className="flex items-center gap-1 mt-2 text-xs text-slate-500 bg-slate-950/50 px-2 py-1 rounded-md border border-slate-800">
                    <FileSpreadsheet size={12} className="text-emerald-500/70" />
                    <span>Ticker, Quantity format</span>
                </div>
            </div>

            <div className="mt-4 bg-gradient-to-r from-emerald-950/40 to-slate-900/40 border border-emerald-900/30 rounded-lg p-3 flex items-start gap-3 relative z-10 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                <div className="relative mt-0.5">
                    <span className="absolute inset-0 bg-emerald-500 blur-sm opacity-40 rounded-full"></span>
                    <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0 relative z-10" />
                </div>
                <div>
                    <p className="text-xs font-semibold text-emerald-300 tracking-wide">Current Portfolio Active</p>
                    <p className="text-[10px] text-slate-400 mt-0.5 font-mono">15 ASSETS • LOADED 2 MINS AGO</p>
                </div>
            </div>
        </div>
    );
};

export default UploadPortfolio;
