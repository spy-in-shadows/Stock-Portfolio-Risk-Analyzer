import React, { useEffect, useRef } from 'react';
import gsap from 'gsap';
import RiskMetrics from '../components/RiskMetrics';
import UploadPortfolio from '../components/UploadPortfolio';
import CorrelationHeatmap from '../components/CorrelationHeatmap';
import MonteCarloChart from '../components/MonteCarloChart';
import ScenarioPanel from '../components/ScenarioPanel';
import RiskGauge from '../components/RiskGauge';

const Dashboard = () => {
    const containerRef = useRef(null);
    const [analysisData, setAnalysisData] = React.useState(null);

    const handleUploadSuccess = (data) => {
        setAnalysisData(data);
    };

    useEffect(() => {
        // Initial entrance animation
        const ctx = gsap.context(() => {
            // Stagger animate all top-level grid items that have the .gsap-entry class
            gsap.from('.gsap-entry', {
                y: 30,
                opacity: 0,
                duration: 0.8,
                stagger: 0.1,
                ease: 'power3.out',
                delay: 0.2
            });

            // Animate header separately
            gsap.from('.header-entry', {
                y: -20,
                opacity: 0,
                duration: 0.8,
                ease: 'power3.out'
            });
        }, containerRef);

        return () => ctx.revert();
    }, []);

    return (
        <div ref={containerRef} className="container mx-auto px-4 py-8 max-w-7xl">
            {/* Header */}
            <header className="header-entry mb-8 flex flex-col md:flex-row justify-between items-start md:items-end border-b border-slate-800 pb-4 gap-4 md:gap-0">
                <div>
                    <h1 className="text-3xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tight flex items-center gap-3">
                        Risk Analytics Terminal
                        <span className="relative flex h-3 w-3">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-3 w-3 bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.8)]"></span>
                        </span>
                    </h1>
                    <p className="text-slate-400 mt-1 text-sm">Portfolio Downside Risk and Tail Exposure Measurement</p>
                </div>
                <div className="text-xs font-mono text-slate-500 uppercase tracking-wider flex gap-4 bg-slate-900/50 p-2 rounded-lg border border-slate-800/50">
                    <span className="smooth-transition hover:text-slate-300 cursor-default">Environment: <span className="text-emerald-400">Production</span></span>
                    <span className="smooth-transition hover:text-slate-300 cursor-default">Status: <span className="text-emerald-400 font-bold">Live</span></span>
                </div>
            </header>

            {/* Main Grid Network */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

                {/* Left Column: Context & Scenarios */}
                <div className="lg:col-span-3 flex flex-col gap-8">
                    <div className="gsap-entry">
                        <UploadPortfolio onUploadSuccess={handleUploadSuccess} />
                    </div>
                    <div className="gsap-entry h-[22rem]">
                        <RiskGauge value={analysisData?.monte_carlo_var_95} />
                    </div>
                    <div className="gsap-entry flex-grow lg:min-h-[24rem]">
                        <ScenarioPanel data={analysisData} />
                    </div>
                </div>

                {/* Right Column: Main Data View */}
                <div className="lg:col-span-9 flex flex-col gap-8">
                    {/* Top: Risk Metrics */}
                    <div className="gsap-entry">
                        <RiskMetrics data={analysisData} />
                    </div>

                    {/* Middle/Bottom: Charts Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 flex-grow">
                        <div className="gsap-entry">
                            <MonteCarloChart data={analysisData} />
                        </div>
                        <div className="gsap-entry">
                            <CorrelationHeatmap matrix={analysisData?.correlation_matrix} assets={analysisData?.asset_names} />
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
};

export default Dashboard;
