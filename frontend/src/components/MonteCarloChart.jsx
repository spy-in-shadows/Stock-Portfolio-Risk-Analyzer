import React from 'react';
import { ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { ActivitySquare } from 'lucide-react';

const MonteCarloChart = () => {
    // Generate dummy geometric brownian motion paths for visualization
    const data = Array.from({ length: 30 }).map((_, i) => {
        const expected = 100 * Math.exp(0.1 * (i / 252));
        const var95 = 100 * Math.exp(0.1 * (i / 252) - 1.65 * 0.2 * Math.sqrt(i / 252));
        const upper95 = 100 * Math.exp(0.1 * (i / 252) + 1.65 * 0.2 * Math.sqrt(i / 252));

        return {
            day: i,
            path1: 100 * Math.exp(((0.1 - 0.5 * 0.2 * 0.2) * (i / 252)) + (0.2 * Math.sqrt(i / 252) * (Math.random() * 2 - 1))),
            path2: 100 * Math.exp(((0.1 - 0.5 * 0.2 * 0.2) * (i / 252)) + (0.2 * Math.sqrt(i / 252) * (Math.random() * 2 - 1.2))), // slightly bearish
            path3: 100 * Math.exp(((0.1 - 0.5 * 0.2 * 0.2) * (i / 252)) + (0.2 * Math.sqrt(i / 252) * (Math.random() * 2 - 0.8))), // slightly bullish
            median: expected,
            var95: var95,
            confidenceBand: [var95, upper95]
        };
    });

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="glass-panel p-3 text-xs border border-slate-700/50 rounded-lg shadow-2xl backdrop-blur-md">
                    <p className="text-slate-400 mb-2 font-semibold uppercase tracking-wider border-b border-slate-700 pb-1">Day T+{label}</p>
                    <div className="space-y-1">
                        {payload.map((p, idx) => {
                            if (p.name === 'confidenceBand') return null; // Don't show band in list
                            return (
                                <div key={idx} className="flex justify-between items-center gap-4 font-mono">
                                    <span style={{ color: p.color || p.stroke }}>{p.name}:</span>
                                    <span className="text-slate-200">
                                        ₹{(p.value * 245000).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="glass-panel rounded-xl p-5 h-full flex flex-col min-h-[320px] relative">
            <div className="flex justify-between items-center mb-6 relative z-10">
                <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                    <ActivitySquare size={16} className="text-emerald-400" />
                    Monte Carlo Simulation
                </h2>
                <div className="text-xs text-slate-500 font-mono bg-slate-800/50 px-2 py-1 rounded">10,000 PATHS • 30 DAYS</div>
            </div>

            <div className="flex-grow w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                        <defs>
                            <linearGradient id="bandGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.15} />
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} strokeOpacity={0.4} />
                        <XAxis dataKey="day" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} domain={['auto', 'auto']} tickFormatter={(v) => `${(v - 100).toFixed(0)}%`} />
                        <Tooltip
                            content={<CustomTooltip />}
                            cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1, strokeDasharray: '4 4' }}
                        />
                        <ReferenceLine y={100} stroke="#334155" strokeDasharray="3 3" />

                        {/* Shaded Confidence Band */}
                        <Area type="monotone" dataKey="confidenceBand" stroke="none" fill="url(#bandGradient)" name="95% Range" />

                        <Line type="monotone" dataKey="path1" stroke="#3b82f6" strokeWidth={1} strokeOpacity={0.4} dot={false} name="Path 1" />
                        <Line type="monotone" dataKey="path2" stroke="#ef4444" strokeWidth={1} strokeOpacity={0.4} dot={false} name="Stress Path" />
                        <Line type="monotone" dataKey="path3" stroke="#10b981" strokeWidth={1} strokeOpacity={0.4} dot={false} name="Bull Path" />
                        <Line type="monotone" dataKey="median" stroke="#e2e8f0" strokeWidth={2} dot={false} name="Expected (E[x])" />
                        <Line type="monotone" dataKey="var95" stroke="#780606" strokeWidth={2} strokeDasharray="4 4" dot={false} name="Historical 5% VaR" />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default MonteCarloChart;
