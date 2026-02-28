import React from 'react';
import { UploadCloud, CheckCircle2 } from 'lucide-react';

const UploadPortfolio = () => {
    return (
        <div className="glass-panel rounded-xl p-5 h-full flex flex-col">
            <h2 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wider">Portfolio Data</h2>

            <div className="border-2 border-dashed border-slate-700 bg-slate-900/50 rounded-lg p-6 flex flex-col items-center justify-center flex-grow transition-colors hover:border-cyan-500 hover:bg-slate-800/50 cursor-pointer group">
                <div className="p-3 bg-slate-800 rounded-full mb-3 group-hover:bg-cyan-900/40 transition-colors">
                    <UploadCloud className="w-6 h-6 text-slate-400 group-hover:text-cyan-400" />
                </div>
                <p className="text-sm font-medium text-slate-200 text-center">Drag & drop CSV</p>
                <p className="text-xs text-slate-500 text-center mt-1">Ticker, Quantity format expected</p>
            </div>

            <div className="mt-4 bg-emerald-950/30 border border-emerald-900/50 rounded-lg p-3 flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-500 flex-shrink-0 mt-0.5" />
                <div>
                    <p className="text-xs font-medium text-emerald-400">Current Portfolio Active</p>
                    <p className="text-[10px] text-slate-400 mt-0.5 font-mono">15 ASSETS • LOADED 2 MINS AGO</p>
                </div>
            </div>
        </div>
    );
};

export default UploadPortfolio;
