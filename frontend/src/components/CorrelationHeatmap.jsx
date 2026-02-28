import React, { useState } from 'react';
import { Network } from 'lucide-react';

const CorrelationHeatmap = () => {
    // Updated assets including NIFTY 50, BANK NIFTY, and GOLD
    const assets = ['NIFTY 50', 'BANK NIFTY', 'GOLD', 'RELIANCE', 'INFY'];

    // Nifty 50 and Bank Nifty highly positive (e.g. 0.88), Nifty/Bank Nifty and Gold inverse (-0.45 / -0.38)
    const matrix = [
        [1.00, 0.88, -0.45, 0.75, 0.60],
        [0.88, 1.00, -0.38, 0.65, 0.45],
        [-0.45, -0.38, 1.00, -0.20, -0.15],
        [0.75, 0.65, -0.20, 1.00, 0.55],
        [0.60, 0.45, -0.15, 0.55, 1.00],
    ];

    const [hoveredCell, setHoveredCell] = useState(null);

    // Smooth gradient from red (high inverse) to neutral to green (high positive)
    const getHeatmapColor = (value) => {
        // value is between -1 and 1
        // Let's interpolate: 
        // -1: rgba(255, 0, 0, 0.8) (#FF0000 in rgb)
        // 0: neutral
        // 1: rgba(0, 192, 64, 0.8) (#00C040 in rgb)
        if (value > 0) {
            // Lerp from slate to #00C040
            const alpha = value * 0.8;
            return `rgba(0, 192, 64, ${alpha})`;
        } else {
            // Lerp from slate to #FF0000
            const alpha = Math.abs(value) * 0.8;
            return `rgba(255, 0, 0, ${alpha})`;
        }
    };

    return (
        <div className="glass-panel rounded-xl p-5 h-full flex flex-col relative">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                    <Network size={16} className="text-indigo-400" />
                    Correlation Matrix
                </h2>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30">
                    Inverse Trend Detected
                </span>
            </div>

            <div className="overflow-x-auto flex-grow outline-none mt-2 relative">
                <table className="w-full text-xs font-mono text-center border-collapse">
                    <thead>
                        <tr>
                            <th className="p-2 text-slate-500 font-normal"></th>
                            {assets.map(a => (
                                <th key={a} className="p-2 text-slate-400 font-medium truncate" title={a}>
                                    {a.substring(0, 5)}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {matrix.map((row, i) => (
                            <tr key={i}>
                                <td className="p-2 text-slate-400 font-medium text-left truncate" title={assets[i]}>
                                    {assets[i].substring(0, 10)}
                                </td>
                                {row.map((val, j) => (
                                    <td key={`${i}-${j}`} className="p-1 relative">
                                        <div
                                            className="w-full h-8 flex items-center justify-center rounded transition-all duration-300 cursor-crosshair hover:ring-1 hover:ring-white/50 hover:scale-110 relative z-10"
                                            style={{ backgroundColor: getHeatmapColor(val) }}
                                            onMouseEnter={() => setHoveredCell({ i, j, val })}
                                            onMouseLeave={() => setHoveredCell(null)}
                                        >
                                            <span className={Math.abs(val) > 0.5 ? 'text-white font-bold' : 'text-slate-300'}>
                                                {val > 0 ? '+' : ''}{val.toFixed(2)}
                                            </span>
                                        </div>
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Floating Tooltip */}
            {hoveredCell && (
                <div className="absolute z-50 pointer-events-none bg-slate-900 border border-slate-700/50 p-2 rounded shadow-2xl backdrop-blur-md"
                    style={{ bottom: '10px', right: '10px' }}>
                    <p className="text-[10px] text-slate-400 mb-1 tracking-wider uppercase">Linear Dependency</p>
                    <p className="text-sm font-bold text-slate-200">
                        {assets[hoveredCell.i]} <span className="text-slate-500 font-normal mx-1">vs</span> {assets[hoveredCell.j]}
                    </p>
                    <p className={`text-xs mt-1 font-mono ${hoveredCell.val > 0.5 ? 'text-[#00C040]' : hoveredCell.val < -0.3 ? 'text-[#FF0000]' : 'text-slate-400'}`}>
                        Correlation: {hoveredCell.val.toFixed(2)}
                        <span className="ml-2 text-slate-500">
                            ({hoveredCell.val > 0.5 ? 'Strong Positive' : hoveredCell.val < -0.3 ? 'Inverse/Hedge' : 'Weak'})
                        </span>
                    </p>
                </div>
            )}

            {/* Gradient Legend */}
            <div className="mt-4 flex items-center justify-between text-[10px] font-mono text-slate-500 border-t border-slate-800/50 pt-3">
                <span>Inverse (-1)</span>
                <div className="flex-grow mx-4 h-1.5 rounded-full bg-gradient-to-r from-[#FF0000] via-slate-800 to-[#00C040]"></div>
                <span>Positive (+1)</span>
            </div>
        </div>
    );
};

export default CorrelationHeatmap;
