import React from 'react';
import { TrendingUp, TrendingDown, Minus, BarChart3, Target, Activity, ArrowUpRight, ArrowDownRight } from 'lucide-react';

const StatusBadge = ({ status }) => {
    if (!status) return <span className="text-xs text-slate-500 font-mono">—</span>;

    const config = {
        'Outperforming': { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', icon: <TrendingUp size={12} /> },
        'At Par': { color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30', icon: <Minus size={12} /> },
        'Underperforming': { color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/30', icon: <TrendingDown size={12} /> },
    };

    const c = config[status] || config['At Par'];
    return (
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider ${c.color} ${c.bg} border ${c.border}`}>
            {c.icon} {status}
        </span>
    );
};

const MetricRow = ({ label, value, suffix = '', colorize = false, invert = false }) => {
    if (value === null || value === undefined) {
        return (
            <div className="flex justify-between items-center py-2 border-b border-slate-800/40 last:border-0">
                <span className="text-xs text-slate-500">{label}</span>
                <span className="text-xs text-slate-600 font-mono">N/A</span>
            </div>
        );
    }

    let textColor = 'text-slate-200';
    if (colorize) {
        const isPositive = invert ? value < 0 : value > 0;
        textColor = isPositive ? 'text-emerald-400' : value === 0 ? 'text-slate-400' : 'text-rose-400';
    }

    const arrow = colorize && value !== 0 ? (
        (invert ? value < 0 : value > 0)
            ? <ArrowUpRight size={12} className="text-emerald-400" />
            : <ArrowDownRight size={12} className="text-rose-400" />
    ) : null;

    return (
        <div className="flex justify-between items-center py-2 border-b border-slate-800/40 last:border-0">
            <span className="text-xs text-slate-400">{label}</span>
            <span className={`text-xs font-mono font-semibold flex items-center gap-1 ${textColor}`}>
                {arrow}
                {typeof value === 'number' ? (value > 0 && colorize ? '+' : '') : ''}
                {typeof value === 'number' ? value.toFixed(4) : value}{suffix}
            </span>
        </div>
    );
};

const BenchmarkComparison = ({ data }) => {
    const comparison = data?.comparison;

    const hasData = comparison && (
        comparison.portfolio_cumulative_return !== null ||
        comparison.benchmark_cumulative_return !== null ||
        comparison.blended_benchmark_cumulative_return !== null
    );

    return (
        <div className="glass-panel rounded-xl p-5 h-full flex flex-col relative overflow-hidden">
            {/* Header */}
            <div className="flex justify-between items-center mb-5 relative z-10">
                <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                    <BarChart3 size={16} className="text-violet-400" />
                    Benchmark Comparison
                </h2>
                {data?.start_date && data?.end_date && (
                    <span className="text-[10px] font-mono text-slate-500 bg-slate-800/50 px-2 py-0.5 rounded border border-slate-700/30">
                        {data.start_date} → {data.end_date}
                    </span>
                )}
            </div>

            {!hasData ? (
                <div className="flex-grow flex flex-col items-center justify-center gap-3 opacity-40">
                    <BarChart3 size={32} className="text-slate-600" />
                    <p className="text-xs text-slate-500 text-center font-mono uppercase tracking-wider">
                        Upload portfolio to compare
                    </p>
                </div>
            ) : (
                <div className="flex flex-col gap-5 flex-grow relative z-10">

                    {/* ── Cumulative Returns Bar ──────────────────────── */}
                    <div className="space-y-3">
                        <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Cumulative Returns</p>
                        {[
                            { label: 'Portfolio', value: comparison.portfolio_cumulative_return, color: 'bg-cyan-500' },
                            { label: 'Benchmark', value: comparison.benchmark_cumulative_return, color: 'bg-blue-500' },
                            { label: 'Blended', value: comparison.blended_benchmark_cumulative_return, color: 'bg-violet-500' },
                        ].map(item => {
                            if (item.value === null || item.value === undefined) return null;
                            const pct = item.value * 100;
                            const maxAbs = Math.max(
                                Math.abs((comparison.portfolio_cumulative_return ?? 0) * 100),
                                Math.abs((comparison.benchmark_cumulative_return ?? 0) * 100),
                                Math.abs((comparison.blended_benchmark_cumulative_return ?? 0) * 100),
                                1
                            );
                            const barWidth = Math.min(Math.abs(pct) / maxAbs * 100, 100);

                            return (
                                <div key={item.label} className="space-y-1">
                                    <div className="flex justify-between items-center">
                                        <span className="text-[10px] text-slate-400 font-medium">{item.label}</span>
                                        <span className={`text-xs font-mono font-bold ${pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            {pct >= 0 ? '+' : ''}{pct.toFixed(2)}%
                                        </span>
                                    </div>
                                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full ${item.color} transition-all duration-700 ease-out`}
                                            style={{ width: `${barWidth}%`, opacity: 0.8 }}
                                        />
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* ── Performance Classification ─────────────────── */}
                    <div className="space-y-3">
                        <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Performance Status</p>
                        <div className="grid grid-cols-2 gap-3">
                            <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-800/50">
                                <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-2">vs Single Benchmark</p>
                                <StatusBadge status={comparison.performance_vs_single} />
                            </div>
                            <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-800/50">
                                <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-2">vs Blended Benchmark</p>
                                <StatusBadge status={comparison.performance_vs_blended} />
                            </div>
                        </div>
                    </div>

                    {/* ── Detailed Metrics ────────────────────────────── */}
                    <div className="space-y-1 bg-slate-900/30 rounded-lg p-3 border border-slate-800/30">
                        <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-2">Risk-Adjusted Metrics</p>
                        <MetricRow label="Relative Gap (Single)" value={comparison.relative_gap_single} colorize />
                        <MetricRow label="Relative Gap (Blended)" value={comparison.relative_gap_blended} colorize />
                        <MetricRow label="Sharpe (Portfolio)" value={comparison.sharpe_portfolio} colorize />
                        <MetricRow label="Sharpe (Benchmark)" value={comparison.sharpe_benchmark} colorize />
                        <MetricRow label="Relative Sharpe" value={comparison.relative_sharpe_vs_single} colorize />
                        <MetricRow label="Tracking Error" value={comparison.tracking_error_blended} />
                        <MetricRow label="Information Ratio" value={comparison.information_ratio_blended} colorize />
                    </div>
                </div>
            )}

            {/* Accent strip */}
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-violet-500 to-blue-500 opacity-80" />
        </div>
    );
};

export default BenchmarkComparison;
