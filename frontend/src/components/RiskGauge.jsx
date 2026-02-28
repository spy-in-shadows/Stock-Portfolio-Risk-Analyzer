import React, { useEffect, useRef, useState } from 'react';
import { RadialBarChart, RadialBar, ResponsiveContainer, PolarAngleAxis } from 'recharts';
import { ShieldCheck } from 'lucide-react';
import gsap from 'gsap';

const RiskGauge = () => {
    const [score, setScore] = useState(0);
    const targetScore = 78; // 0-100 where higher is safer/better risk profile

    useEffect(() => {
        // Animate gauge fill on mount
        const obj = { val: 0 };
        gsap.to(obj, {
            val: targetScore,
            duration: 2,
            ease: "power3.out",
            onUpdate: () => {
                setScore(Math.round(obj.val));
            }
        });
    }, []);

    const data = [
        { name: 'RiskScore', value: score, fill: score > 70 ? '#10b981' : score > 40 ? '#f59e0b' : '#ef4444' }
    ];

    let riskMessage;
    if (score >= 70) {
        riskMessage = "Your portfolio is well-protected. Your investments are balanced to minimize large sudden losses.";
    } else if (score >= 40) {
        riskMessage = "Your portfolio has moderate risk. You have a mix of safe and volatile investments.";
    } else {
        riskMessage = "Your portfolio is highly aggressive. Be prepared for larger swings in your investment value.";
    }

    return (
        <div className="glass-panel rounded-xl p-5 h-full flex flex-col items-center justify-center relative overflow-hidden group">
            {/* Background rotating glow */}
            <div className="absolute inset-0 opacity-20 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-emerald-500/40 via-transparent to-transparent group-hover:opacity-40 transition-opacity duration-1000"></div>

            <div className="flex justify-between items-center w-full mb-2 z-10">
                <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                    <ShieldCheck size={16} className="text-emerald-400" />
                    Portfolio Risk Score
                </h2>
            </div>

            <div className="relative w-full h-48 flex items-center justify-center z-10 mt-2">
                <ResponsiveContainer width="100%" height="100%">
                    <RadialBarChart
                        cx="50%"
                        cy="60%"
                        innerRadius="70%"
                        outerRadius="100%"
                        barSize={14}
                        data={data}
                        startAngle={180}
                        endAngle={0}
                    >
                        <PolarAngleAxis
                            type="number"
                            domain={[0, 100]}
                            angleAxisId={0}
                            tick={false}
                        />
                        <RadialBar
                            minAngle={15}
                            background={{ fill: '#1e293b' }}
                            clockWise
                            dataKey="value"
                            cornerRadius={10}
                        />
                    </RadialBarChart>
                </ResponsiveContainer>

                {/* Center Text */}
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-[20%] text-center">
                    <div className="text-4xl font-extrabold text-white tracking-tighter">
                        {score}
                        <span className="text-sm text-slate-500 font-normal ml-1">/100</span>
                    </div>
                    <div className={`text-[10px] font-mono mt-1 uppercase tracking-widest ${score > 70 ? 'text-emerald-400' : score > 40 ? 'text-amber-400' : 'text-rose-400'}`}>
                        {score > 70 ? 'Optimal Risk Boundary' : score > 40 ? 'Moderate Risk Profile' : 'High Risk Alert'}
                    </div>
                </div>
            </div>

            <p className="text-xs text-slate-300 text-center mt-2 z-10 leading-relaxed max-w-[90%]">
                {riskMessage}
            </p>

            {/* CIBIL-style Score Legend */}
            <div className="w-full flex justify-between items-center mt-auto pt-4 border-t border-slate-800/50 z-10">
                <div className="flex flex-col items-center">
                    <div className="text-[9px] text-slate-500 mb-1 tracking-wider font-semibold">HIGH RISK</div>
                    <div className="w-12 h-1.5 rounded-full bg-[#ef4444] mb-1"></div>
                    <div className="text-[10px] font-mono text-slate-400">0-40</div>
                </div>
                <div className="flex flex-col items-center">
                    <div className="text-[9px] text-slate-500 mb-1 tracking-wider font-semibold">MODERATE</div>
                    <div className="w-12 h-1.5 rounded-full bg-[#f59e0b] mb-1"></div>
                    <div className="text-[10px] font-mono text-slate-400">41-70</div>
                </div>
                <div className="flex flex-col items-center">
                    <div className="text-[9px] text-slate-500 mb-1 tracking-wider font-semibold">OPTIMAL</div>
                    <div className="w-12 h-1.5 rounded-full bg-[#10b981] mb-1"></div>
                    <div className="text-[10px] font-mono text-slate-400">71-100</div>
                </div>
            </div>
        </div>
    );
};

export default RiskGauge;
