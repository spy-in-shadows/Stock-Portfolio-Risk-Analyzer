import React from 'react';
import { Network } from 'lucide-react';

const CorrelationHeatmap = () => {
    // Dummy data representing an asset correlation matrix
    const assets = ['RELIANCE', 'INFY', 'HDFCBANK', 'TCS', 'ITC'];
    const matrix = [
        [1.00, 0.45, 0.62, 0.38, 0.22],
        [0.45, 1.00, 0.18, 0.82, 0.11],
        [0.62, 0.18, 1.00, 0.29, 0.44],
        [0.38, 0.82, 0.29, 1.00, 0.05],
        [0.22, 0.11, 0.44, 0.05, 1.00],
    ];

    const getColor = (value) => {
        if (value === 1) return 'bg-cyan-900/50 text-cyan-400';
        if (value > 0.7) return 'bg-rose-900/40 text-rose-300 font-bold'; // High correlation warning
        if (value > 0.4) return 'bg-amber-900/30 text-amber-300';
        return 'bg-slate-800/50 text-slate-400';
    };

    return (
        <div className="glass-panel rounded-xl p-5 h-full flex flex-col">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                    <Network size={16} className="text-indigo-400" />
                    Correlation Matrix
                </h2>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30">
                    Clustered Risk Detected
                </span>
            </div>

            <div className="overflow-x-auto flex-grow outline-none mt-2">
                <table className="w-full text-xs font-mono text-center border-collapse">
                    <thead>
                        <tr>
                            <th className="p-2 text-slate-500 font-normal"></th>
                            {assets.map(a => (
                                <th key={a} className="p-2 text-slate-400 font-medium truncate">{a.substring(0, 4)}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {matrix.map((row, i) => (
                            <tr key={i}>
                                <td className="p-2 text-slate-400 font-medium text-left truncate">{assets[i].substring(0, 4)}</td>
                                {row.map((val, j) => (
                                    <td key={`${i}-${j}`} className="p-1">
                                        <div className={`w-full h-8 flex items-center justify-center rounded transition-colors cursor-crosshair hover:ring-1 hover:ring-white/50 ${getColor(val)}`}>
                                            {val.toFixed(2)}
                                        </div>
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default CorrelationHeatmap;
