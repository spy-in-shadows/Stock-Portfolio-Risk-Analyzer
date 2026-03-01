import React, { useEffect, useRef, useState, useMemo } from 'react';
import { Zap, ShieldAlert, Target, Activity, TrendingDown } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import gsap from 'gsap';

// Simple number counting component
const AnimatedNumber = ({ value, format = 'number' }) => {
    const elRef = useRef(null);
    const objRef = useRef({ val: 0 });
    const tweenRef = useRef(null);

    useEffect(() => {
        // Correctly parse signed numbers like '-0.06' or '1.23'
        const raw = String(value).replace(/[^0-9.\-]/g, '');  // keep minus sign
        let target = parseFloat(raw);
        if (isNaN(target)) target = 0;

        // Kill any in-progress tween before starting a new one
        if (tweenRef.current) tweenRef.current.kill();

        tweenRef.current = gsap.to(objRef.current, {
            val: target,
            duration: 1.2,
            ease: 'power3.out',
            onUpdate: () => {
                const current = objRef.current.val;
                if (!elRef.current) return;

                let formatted = '';
                if (format === 'currencyM') formatted = `₹${current.toFixed(1)}M`;
                else if (format === 'currencyK') formatted = `₹${Math.round(current)}K`;
                else if (format === 'percentage') {
                    // Use 2dp so small values like 0.06% don't round to 0.0%
                    formatted = `${current.toFixed(2)}%`;
                } else {
                    formatted = current.toFixed(2);
                }

                elRef.current.innerText = formatted;
            }
        });

        return () => { if (tweenRef.current) tweenRef.current.kill(); };
    }, [value, format]);

    return <span ref={elRef}>0</span>;
};

// Generate dummy sparkline data
const generateSparkline = (trend) => {
    return Array.from({ length: 15 }).map((_, i) => ({
        val: trend === 'up' ? i + Math.random() * 5 :
            trend === 'down' ? 15 - i + Math.random() * 5 :
                5 + Math.random() * 10
    }));
};

const RiskMetrics = ({ data }) => {
    const cardsRef = useRef([]);

    useEffect(() => {
        cardsRef.current.forEach(card => {
            if (!card) return;
            // Hover micro-animations (lift + bloom)
            card.addEventListener('mouseenter', () => {
                gsap.to(card, { y: -4, scale: 1.01, boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.5), inset 0 0 0 1px rgba(255,255,255,0.08)', duration: 0.3, ease: 'power2.out' });
            });
            card.addEventListener('mouseleave', () => {
                gsap.to(card, { y: 0, scale: 1, boxShadow: 'inset 0 0 0 1px rgba(255,255,255,0.04), 0 4px 20px -2px rgba(0, 0, 0, 0.4)', duration: 0.3, ease: 'power2.out' });
            });
        });
    }, []);

    const metrics = useMemo(() => {
        if (!data) {
            return [
                { label: 'Expt Return', value: '0.0', format: 'number', sub: 'Historical mean', icon: <Target size={16} />, color: 'text-indigo-400', bg: 'bg-indigo-500/10', gradient: 'from-indigo-500 to-blue-500', trend: 'flat', primary: true },
                { label: 'Monte Carlo VaR', value: '0.0', format: 'percentage', sub: '95% Confidence', icon: <ShieldAlert size={16} />, color: 'text-rose-400', bg: 'bg-rose-500/10', gradient: 'from-rose-500 to-red-500', trend: 'flat', primary: true },
                { label: 'Portfolio Vol', value: '0.0', format: 'percentage', sub: 'Daily std dev', icon: <Activity size={16} />, color: 'text-amber-400', bg: 'bg-amber-500/10', gradient: 'from-amber-500 to-orange-500', trend: 'flat', primary: false },
                { label: 'Historical VaR', value: '0.0', format: 'percentage', sub: '95% · Daily', icon: <TrendingDown size={16} />, color: 'text-emerald-400', bg: 'bg-emerald-500/10', gradient: 'from-emerald-500 to-teal-500', trend: 'flat', primary: false },
                { label: 'Portfolio Beta', value: '0.0', format: 'number', sub: 'versus Benchmark', icon: <Zap size={16} />, color: 'text-cyan-400', bg: 'bg-cyan-500/10', gradient: 'from-cyan-500 to-blue-500', trend: 'flat', primary: false },
            ];
        }

        return [
            {
                label: 'Expt Return',
                // Use 4dp so daily values like 0.065% don't round to 0.00%
                value: (data.portfolio_expected_return * 100).toFixed(4),
                format: 'percentage',
                sub: 'Daily · Historical',
                icon: <Target size={16} />,
                color: 'text-indigo-400', bg: 'bg-indigo-500/10', gradient: 'from-indigo-500 to-blue-500',
                trend: data.portfolio_expected_return > 0 ? 'up' : 'down', primary: true
            },
            {
                label: 'MC VaR (30D)',
                value: (Math.abs(data.monte_carlo_var95_30d ?? data.monte_carlo_var_95) * 100).toFixed(2),
                format: 'percentage',
                sub: '30-Day · 95% CI',
                icon: <ShieldAlert size={16} />,
                color: 'text-rose-400', bg: 'bg-rose-500/10', gradient: 'from-rose-500 to-red-500',
                trend: 'down', primary: true
            },
            {
                label: 'Portfolio Vol',
                value: (data.portfolio_volatility * 100).toFixed(2),
                format: 'percentage',
                sub: 'Daily Std Dev',
                icon: <Activity size={16} />,
                color: 'text-amber-400', bg: 'bg-amber-500/10', gradient: 'from-amber-500 to-orange-500',
                trend: 'flat', primary: false
            },
            {
                label: 'Historical VaR',
                value: (Math.abs(data.historical_var_95 || 0) * 100).toFixed(2),
                format: 'percentage',
                sub: '95% · Daily',
                icon: <TrendingDown size={16} />,
                color: 'text-emerald-400', bg: 'bg-emerald-500/10', gradient: 'from-emerald-500 to-teal-500',
                trend: 'down', primary: false
            },
            {
                label: 'Portfolio Beta',
                value: data.beta.toFixed(2),
                format: 'number',
                sub: 'versus Benchmark',
                icon: <Zap size={16} />,
                color: 'text-cyan-400', bg: 'bg-cyan-500/10', gradient: 'from-cyan-500 to-blue-500',
                trend: data.beta > 1 ? 'up' : 'down', primary: false
            },
        ];
    }, [data]);

    return (
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            {metrics.map((m, idx) => (
                <div
                    key={m.label}
                    ref={el => cardsRef.current[idx] = el}
                    className="glass-panel rounded-xl p-4 flex flex-col cursor-crosshair transform-gpu smooth-transition lg:col-span-1 col-span-1"
                >
                    {/* Gradient accent strip */}
                    <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${m.gradient} opacity-80`}></div>

                    <div className="flex justify-between items-start mb-2 relative z-10">
                        <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">{m.label}</span>
                        <div className={`p-1.5 rounded-md ${m.bg} ${m.color}`}>
                            {m.icon}
                        </div>
                    </div>

                    <div className="mt-auto relative z-10 flex flex-col">
                        <div className="flex items-end justify-between">
                            <h3 className={`${m.primary ? 'text-3xl' : 'text-2xl'} font-extrabold text-slate-100 tracking-tighter`}>
                                <AnimatedNumber value={m.value} format={m.format} />
                            </h3>
                            {/* Tiny Sparkline */}
                            <div className="w-16 h-8 opacity-60">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={generateSparkline(m.trend)}>
                                        <Line type="monotone" dataKey="val" stroke="currentColor" strokeWidth={1.5} dot={false} className={m.color} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                        <p className="text-[11px] text-slate-500 mt-1 font-mono">{m.sub}</p>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default RiskMetrics;
