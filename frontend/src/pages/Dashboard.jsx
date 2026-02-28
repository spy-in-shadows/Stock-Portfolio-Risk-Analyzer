import React, { useEffect, useRef } from 'react';
import gsap from 'gsap';
import RiskMetrics from '../components/RiskMetrics';
import UploadPortfolio from '../components/UploadPortfolio';
import CorrelationHeatmap from '../components/CorrelationHeatmap';
import MonteCarloChart from '../components/MonteCarloChart';
import ScenarioPanel from '../components/ScenarioPanel';

const Dashboard = () => {
    const containerRef = useRef(null);

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
            <header className="header-entry mb-8 flex justify-between items-end border-b border-slate-800 pb-4">
                <div>
                    <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tight">
                        Risk Analytics Terminal
                    </h1>
                    <p className="text-slate-400 mt-1 text-sm">Portfolio Downside Risk and Tail Exposure Measurement</p>
                </div>
                <div className="text-xs font-mono text-slate-500 uppercase tracking-wider flex gap-4">
                    <span>Env: <span className="text-emerald-400">Production</span></span>
                    <span>Status: <span className="text-emerald-400">Live</span></span>
                </div>
            </header>

            {/* Main Grid Network */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

                {/* Left Column: Upload & Scenario Panel */}
                <div className="lg:col-span-3 flex flex-col gap-6">
                    <div className="gsap-entry">
                        <UploadPortfolio />
                    </div>
                    <div className="gsap-entry flex-grow">
                        <ScenarioPanel />
                    </div>
                </div>

                {/* Right Column: Main Data View */}
                <div className="lg:col-span-9 flex flex-col gap-6">
                    {/* Top: Risk Metrics */}
                    <div className="gsap-entry">
                        <RiskMetrics />
                    </div>

          /* Middle/Bottom: Charts Grid */
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-grow">
                        <div className="gsap-entry">
                            <MonteCarloChart />
                        </div>
                        <div className="gsap-entry">
                            <CorrelationHeatmap />
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
};

export default Dashboard;
