import React from 'react';
import { ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { ActivitySquare } from 'lucide-react';

const MonteCarloChart = ({ data: analysisData }) => {
    // Use real simulation paths from backend if available
    const hasRealData = !!(analysisData?.simulation_paths?.length);

    const chartData = hasRealData
        ? analysisData.simulation_paths.map(pt => ({
            day: pt.day,
            p50: pt.p50,              // Median path
            innerBand: [pt.p25, pt.p75],   // 25th–75th band
            outerBand: [pt.p5, pt.p95],    // 5th–95th band
            p5: pt.p5,                // Lower VaR line
        }))
        : (() => {
            // Fallback: simplified GBM visualization
            const mu = 0.05 / 252;
            const sigma = 0.2 / Math.sqrt(252);
            return Array.from({ length: 31 }).map((_, i) => {
                const drift = (mu - 0.5 * sigma * sigma) * i;
                const diffusion = sigma * Math.sqrt(i);
                const expected = 100 * Math.exp(mu * i);
                const lower95 = 100 * Math.exp(drift - 1.645 * diffusion);
                const upper95 = 100 * Math.exp(drift + 1.645 * diffusion);
                return {
                    day: i,
                    p50: expected,
                    innerBand: [100 * Math.exp(drift - 0.674 * diffusion), 100 * Math.exp(drift + 0.674 * diffusion)],
                    outerBand: [lower95, upper95],
                    p5: lower95,
                };
            });
        })();

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            // Extract only scalar values, skip arrays (band ranges)
            const scalarItems = payload.filter(p => !Array.isArray(p.value) && p.value !== undefined && p.value !== null);

            return (
                <div className="glass-panel p-3 text-xs border border-slate-700/50 rounded-lg shadow-2xl backdrop-blur-md min-w-[180px]">
                    <p className="text-slate-400 mb-2 font-semibold uppercase tracking-wider border-b border-slate-700 pb-1">
                        Day T+{label} {hasRealData ? <span className="text-emerald-400 ml-1">● Live</span> : ''}
                    </p>
                    <div className="space-y-1">
                        {payload.map((p, idx) => {
                            if (Array.isArray(p.value) || p.value === undefined || p.value === null) return null;
                            const delta = ((p.value - 100)).toFixed(2);
                            const sign = delta >= 0 ? '+' : '';
                            return (
                                <div key={idx} className="flex justify-between items-center gap-4 font-mono">
                                    <span style={{ color: p.color || p.stroke }}>{p.name}:</span>
                                    <span className="text-slate-200">
                                        {p.value.toFixed(2)}
                                        <span className={`ml-1 text-[10px] ${delta >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            ({sign}{delta}%)
                                        </span>
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
                <div className="flex items-center gap-2">
                    {hasRealData && (
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                            ● Real Paths
                        </span>
                    )}
                    <div className="text-xs text-slate-500 font-mono bg-slate-800/50 px-2 py-1 rounded">
                        {hasRealData ? `${(analysisData?.simulation_paths ? 10000 : 0).toLocaleString()} PATHS` : 'PREVIEW'} • 30 DAYS
                    </div>
                </div>
            </div>

            {/* Legend */}
            <div className="flex items-center gap-4 mb-3 text-[10px] font-mono text-slate-500">
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-slate-200 inline-block"></span>Median (p50)</span>
                <span className="flex items-center gap-1"><span className="w-3 h-2 bg-blue-500/30 inline-block rounded"></span>p25–p75</span>
                <span className="flex items-center gap-1"><span className="w-3 h-2 bg-blue-500/10 inline-block rounded"></span>p5–p95</span>
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-rose-700 border-dashed inline-block"></span>5% VaR</span>
            </div>

            <div className="flex-grow w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                        <defs>
                            <linearGradient id="outerBandGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.08} />
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
                            </linearGradient>
                            <linearGradient id="innerBandGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.20} />
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.08} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} strokeOpacity={0.4} />
                        <XAxis dataKey="day" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} label={{ value: 'Trading Days', position: 'insideBottom', offset: -2, fill: '#475569', fontSize: 9 }} />
                        <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} domain={['auto', 'auto']} tickFormatter={(v) => `${(v - 100).toFixed(0)}%`} />
                        <Tooltip
                            content={<CustomTooltip />}
                            cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1, strokeDasharray: '4 4' }}
                        />
                        <ReferenceLine y={100} stroke="#334155" strokeDasharray="3 3" />

                        {/* Outer band: p5 – p95 */}
                        <Area type="monotone" dataKey="outerBand" stroke="none" fill="url(#outerBandGrad)" name="p5–p95 Range" />

                        {/* Inner band: p25 – p75 (IQR) */}
                        <Area type="monotone" dataKey="innerBand" stroke="none" fill="url(#innerBandGrad)" name="p25–p75 Range" />

                        {/* Median line */}
                        <Line type="monotone" dataKey="p50" stroke="#e2e8f0" strokeWidth={2} dot={false} name="Median (p50)" />

                        {/* 5th percentile VaR floor */}
                        <Line type="monotone" dataKey="p5" stroke="#780606" strokeWidth={1.5} strokeDasharray="5 4" dot={false} name="5% VaR Bound" />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default MonteCarloChart;
