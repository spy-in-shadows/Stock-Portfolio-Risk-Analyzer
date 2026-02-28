import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, CheckCircle2, FileSpreadsheet, Globe } from 'lucide-react';

const UploadPortfolio = ({ onUploadSuccess }) => {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [ticker, setTicker] = useState('^NSEI'); // Default to Nifty 50
    const fileInputRef = useRef(null);

    const handleFileChange = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        if (!ticker) {
            setError("Please enter a benchmark ticker first.");
            return;
        }

        setIsLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('benchmark_ticker', ticker);

        // Default parameters
        formData.append('confidence_level', '0.95');
        formData.append('simulations', '5000');

        try {
            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || data.error || 'Upload failed');
            }

            if (onUploadSuccess) {
                onUploadSuccess(data);
            }
        } catch (err) {
            setError(err.message);
            console.error('Upload error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    const triggerFileInput = () => {
        fileInputRef.current?.click();
    };

    return (
        <div className="glass-panel rounded-xl p-5 h-full flex flex-col relative overflow-hidden group">
            <h2 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wider relative z-10">Risk Analytics Configuration</h2>

            {/* Benchmark Selection */}
            <div className="mb-4 relative z-10">
                <label className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-1.5 block">Market Benchmark</label>
                <div className="relative">
                    <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
                        <Globe size={14} className="text-cyan-500" />
                    </div>
                    <input
                        type="text"
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value.toUpperCase())}
                        placeholder="e.g. ^NSEI, SPY, BTC-USD"
                        className="w-full bg-slate-900/60 border border-slate-700/50 rounded-lg py-2 pl-9 pr-4 text-xs font-mono text-slate-100 focus:outline-none focus:border-cyan-500/50 transition-all"
                    />
                </div>
            </div>

            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                accept=".csv"
            />

            <div
                onClick={triggerFileInput}
                className={`border-2 border-dashed ${error ? 'border-rose-500/50' : 'border-slate-700/50'} bg-slate-900/40 rounded-lg p-6 flex flex-col items-center justify-center flex-grow transition-all duration-300 hover:border-cyan-500/50 hover:bg-slate-800/60 cursor-pointer relative z-10 hover:shadow-[0_0_30px_rgba(8,145,178,0.15)]`}
            >
                {isLoading ? (
                    <div className="flex flex-col items-center gap-3">
                        <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
                        <p className="text-xs text-cyan-400 font-medium tracking-tight">Fetching Market Data...</p>
                    </div>
                ) : (
                    <>
                        <div className="relative">
                            <div className="absolute inset-0 bg-cyan-500/20 blur-xl rounded-full scale-150 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                            <div className="p-3 bg-slate-800/80 rounded-full mb-3 group-hover:bg-cyan-900/50 transition-colors border border-slate-700 group-hover:border-cyan-700 relative z-10">
                                <UploadCloud className="w-6 h-6 text-slate-400 group-hover:text-cyan-400 transition-colors" />
                            </div>
                        </div>
                        <p className="text-sm font-medium text-slate-200 text-center uppercase tracking-tighter">
                            {error ? 'Review Config' : 'Upload Assets CSV'}
                        </p>
                        {error && <p className="text-[10px] text-rose-400 mt-1 max-w-[200px] text-center">{error}</p>}
                    </>
                )}
            </div>

            <div className="mt-4 bg-slate-900/60 border border-slate-800/50 rounded-lg p-3 relative z-10">
                <div className="flex items-center gap-2 mb-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.6)]"></div>
                    <p className="text-[10px] uppercase font-bold text-slate-400 tracking-widest">Global Data Standard</p>
                </div>

                <div className="grid grid-cols-2 gap-1.5">
                    <div className="flex flex-col gap-1 p-2 bg-slate-800/40 rounded border border-slate-700/30 text-center">
                        <span className="text-[8px] text-slate-500 font-mono uppercase tracking-tighter">Col 1</span>
                        <span className="text-[10px] text-cyan-400 font-bold font-mono">DATE</span>
                    </div>
                    <div className="flex flex-col gap-1 p-2 bg-slate-800/40 rounded border border-slate-700/30 text-center">
                        <span className="text-[8px] text-slate-500 font-mono uppercase tracking-tighter">Cols 2...N</span>
                        <span className="text-[10px] text-indigo-400 font-bold font-mono">ASSET PRICES</span>
                    </div>
                </div>

                <p className="text-[9px] text-slate-500 mt-2 font-medium italic border-t border-slate-800/50 pt-2 text-center uppercase tracking-tighter">
                    Note: CSV should contain only asset prices. Benchmark is fetched via ticker.
                </p>
            </div>
        </div>
    );
};

export default UploadPortfolio;
