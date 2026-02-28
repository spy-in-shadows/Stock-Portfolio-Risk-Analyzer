import React, { useEffect, useRef, useState, useMemo } from 'react';
import { Zap, ShieldAlert, TrendingDown, Target, Activity } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import gsap from 'gsap';

// Simple number counting component
const AnimatedNumber = ({ value, format = 'number' }) => {
    const elRef = useRef(null);
    const objRef = useRef({ val: 0 });

    useEffect(() => {
        // Parse numbers like '24.5' from '₹24.5M'
        let target = parseFloat(value.replace(/[^0-9.]/g, ''));
        if (isNaN(target)) target = 0;

        gsap.to(objRef.current, {
            val: target,
            duration: 1.5,
            ease: 'power3.out',
            onUpdate: () => {
                let current = objRef.current.val;
                if (!elRef.current) return;

                // Format back to string
                let formatted = '';
                if (format === 'currencyM') formatted = `₹${current.toFixed(1)}M`;
                else if (format === 'currencyK') formatted = `₹${Math.round(current)}K`;
                else if (format === 'percentage') formatted = `${current.toFixed(1)}%`;
                else formatted = current.toFixed(2);

                elRef.current.innerText = formatted;
            }
        });
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

const RiskMetrics = () => {
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

    const metrics = useMemo(() => [
        { label: 'Total Exposure', value: '₹24.5M', format: 'currencyM', sub: 'Gross notional', icon: <Target size={16} />, color: 'text-indigo-400', bg: 'bg-indigo-500/10', gradient: 'from-indigo-500 to-blue-500', trend: 'up', primary: true },
        { label: '95% Daily VaR', value: '₹420K', format: 'currencyK', sub: '1.71% of port', icon: <ShieldAlert size={16} />, color: 'text-rose-400', bg: 'bg-rose-500/10', gradient: 'from-rose-500 to-red-500', trend: 'down', primary: true },
        { label: 'Portfolio Vol (Ann)', value: '18.4%', format: 'percentage', sub: 'Benchmark: 14.2%', icon: <Activity size={16} />, color: 'text-amber-400', bg: 'bg-amber-500/10', gradient: 'from-amber-500 to-orange-500', trend: 'flat', primary: false },
        { label: 'Sharpe Ratio', value: '1.82', format: 'number', sub: 'Risk-free: 6.0%', icon: <TrendingDown size={16} />, color: 'text-emerald-400', bg: 'bg-emerald-500/10', gradient: 'from-emerald-500 to-teal-500', trend: 'up', primary: false },
        { label: 'Beta (vs NIFTY)', value: '1.14', format: 'number', sub: 'Market proxy', icon: <Zap size={16} />, color: 'text-cyan-400', bg: 'bg-cyan-500/10', gradient: 'from-cyan-500 to-blue-500', trend: 'flat', primary: false },
    ], []);

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
