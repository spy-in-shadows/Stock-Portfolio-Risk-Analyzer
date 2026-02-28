import React, { useEffect, useRef } from 'react';
import { Zap, ShieldAlert, TrendingDown, Target, Activity } from 'lucide-react';
import gsap from 'gsap';

const RiskMetrics = () => {
    const cardsRef = useRef([]);

    useEffect(() => {
        // Hover micro-animations
        cardsRef.current.forEach(card => {
            if (!card) return;
            card.addEventListener('mouseenter', () => {
                gsap.to(card, { y: -4, scale: 1.02, duration: 0.3, ease: 'power2.out' });
            });
            card.addEventListener('mouseleave', () => {
                gsap.to(card, { y: 0, scale: 1, duration: 0.3, ease: 'power2.out' });
            });
        });
    }, []);

    const metrics = [
        { label: 'Total Exposure', value: '₹24.5M', sub: 'Gross notional', icon: <Target size={16} />, color: 'text-indigo-400', border: 'border-indigo-500/30', bg: 'bg-indigo-500/10' },
        { label: '95% Daily VaR', value: '₹420K', sub: '1.71% of port', icon: <ShieldAlert size={16} />, color: 'text-rose-400', border: 'border-rose-500/30', bg: 'bg-rose-500/10' },
        { label: 'Portfolio Vol (Ann)', value: '18.4%', sub: 'Benchmark: 14.2%', icon: <Activity size={16} />, color: 'text-amber-400', border: 'border-amber-500/30', bg: 'bg-amber-500/10' },
        { label: 'Sharpe Ratio', value: '1.82', sub: 'Risk-free: 6.0%', icon: <TrendingDown size={16} />, color: 'text-emerald-400', border: 'border-emerald-500/30', bg: 'bg-emerald-500/10' },
        { label: 'Beta (vs NIFTY)', value: '1.14', sub: 'Market proxy', icon: <Zap size={16} />, color: 'text-cyan-400', border: 'border-cyan-500/30', bg: 'bg-cyan-500/10' },
    ];

    return (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {metrics.map((m, idx) => (
                <div
                    key={m.label}
                    ref={el => cardsRef.current[idx] = el}
                    className={`glass-panel rounded-xl p-4 border-t-2 ${m.border} flex flex-col cursor-crosshair`}
                >
                    <div className="flex justify-between items-start mb-2">
                        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">{m.label}</span>
                        <div className={`p-1.5 rounded-md ${m.bg} ${m.color}`}>
                            {m.icon}
                        </div>
                    </div>
                    <div className="mt-auto">
                        <h3 className="text-2xl font-bold text-slate-100 tracking-tight">{m.value}</h3>
                        <p className="text-[11px] text-slate-500 mt-1 font-mono">{m.sub}</p>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default RiskMetrics;
