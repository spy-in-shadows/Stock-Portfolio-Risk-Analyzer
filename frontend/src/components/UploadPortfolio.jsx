import React, { useState, useRef } from 'react';
import { UploadCloud, Globe, Layers, Plus, X } from 'lucide-react';

const UploadPortfolio = ({ onUploadSuccess }) => {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [ticker, setTicker] = useState('^NSEI');
    const [blendedEnabled, setBlendedEnabled] = useState(true);
    const [blendedTickers, setBlendedTickers] = useState([
        { ticker: '^NSEI', weight: '70' },
        { ticker: '^NSEMDCP50', weight: '30' },
    ]);
    const fileInputRef = useRef(null);

    const addBlendedRow = () => {
        setBlendedTickers(prev => [...prev, { ticker: '', weight: '' }]);
    };

    const removeBlendedRow = (idx) => {
        setBlendedTickers(prev => prev.filter((_, i) => i !== idx));
    };

    const updateBlendedRow = (idx, field, value) => {
        setBlendedTickers(prev => prev.map((row, i) =>
            i === idx ? { ...row, [field]: field === 'ticker' ? value.toUpperCase() : value } : row
        ));
    };

    const handleFileChange = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        if (!ticker && !blendedEnabled) {
            setError("Please enter a benchmark ticker or enable blended benchmark.");
            return;
        }

        setIsLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);

        // Single benchmark
        if (ticker) {
            formData.append('benchmark_ticker', ticker);
        }

        // Blended benchmark
        if (blendedEnabled && blendedTickers.length >= 2) {
            const validRows = blendedTickers.filter(r => r.ticker && r.weight);
            if (validRows.length >= 2) {
                const blendedPayload = {
                    tickers: validRows.map(r => r.ticker),
                    weights: validRows.map(r => parseFloat(r.weight) / 100),
                };
                formData.append('blended_benchmark', JSON.stringify(blendedPayload));
            }
        }

        // Default parameters
        formData.append('confidence_level', '0.95');
        formData.append('simulations', '5000');

        // Debug: log what we're sending
        for (const [key, value] of formData.entries()) {
            console.log(`[Upload] ${key}:`, value instanceof File ? value.name : value);
        }

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

            {/* Single Benchmark */}
            <div className="mb-3 relative z-10">
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

            {/* Blended Benchmark Toggle */}
            <div className="mb-3 relative z-10">
                <button
                    type="button"
                    onClick={() => setBlendedEnabled(!blendedEnabled)}
                    className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg border text-[10px] font-bold uppercase tracking-widest transition-all ${blendedEnabled
                        ? 'bg-violet-900/30 border-violet-500/40 text-violet-300'
                        : 'bg-slate-900/40 border-slate-700/40 text-slate-500 hover:border-violet-500/30 hover:text-violet-400'
                        }`}
                >
                    <Layers size={12} />
                    Blended Benchmark {blendedEnabled ? '● ON' : '○ OFF'}
                </button>
            </div>

            {/* Blended Benchmark Inputs */}
            {blendedEnabled && (
                <div className="mb-3 relative z-10 bg-slate-900/40 rounded-lg p-3 border border-violet-800/30 space-y-2">
                    <div className="flex justify-between items-center mb-1">
                        <span className="text-[9px] text-violet-400 uppercase font-bold tracking-widest">Blended Components</span>
                        <span className="text-[9px] text-slate-500 font-mono">Weights must sum to 100%</span>
                    </div>
                    {blendedTickers.map((row, idx) => (
                        <div key={idx} className="flex gap-2 items-center">
                            <input
                                type="text"
                                value={row.ticker}
                                onChange={(e) => updateBlendedRow(idx, 'ticker', e.target.value)}
                                placeholder="Ticker"
                                className="flex-1 bg-slate-800/60 border border-slate-700/40 rounded py-1.5 px-2.5 text-[11px] font-mono text-slate-200 focus:outline-none focus:border-violet-500/50"
                            />
                            <div className="relative w-20">
                                <input
                                    type="number"
                                    value={row.weight}
                                    onChange={(e) => updateBlendedRow(idx, 'weight', e.target.value)}
                                    placeholder="Wt%"
                                    min="0"
                                    max="100"
                                    className="w-full bg-slate-800/60 border border-slate-700/40 rounded py-1.5 px-2.5 pr-6 text-[11px] font-mono text-slate-200 focus:outline-none focus:border-violet-500/50"
                                />
                                <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-slate-500">%</span>
                            </div>
                            {blendedTickers.length > 2 && (
                                <button
                                    onClick={() => removeBlendedRow(idx)}
                                    className="p-1 rounded hover:bg-rose-900/30 text-slate-600 hover:text-rose-400 transition-colors"
                                >
                                    <X size={12} />
                                </button>
                            )}
                        </div>
                    ))}
                    <button
                        onClick={addBlendedRow}
                        className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded border border-dashed border-slate-700/40 text-[10px] text-slate-500 hover:text-violet-400 hover:border-violet-500/40 transition-all font-bold uppercase tracking-wider"
                    >
                        <Plus size={10} /> Add Ticker
                    </button>
                </div>
            )}

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
                    <p className="text-[10px] uppercase font-bold text-slate-400 tracking-widest">Broker Export Standard</p>
                </div>

                <div className="grid grid-cols-3 gap-1.5">
                    <div className="flex flex-col gap-1 p-2 bg-slate-800/40 rounded border border-cyan-800/30 text-center">
                        <span className="text-[8px] text-slate-500 font-mono uppercase tracking-tighter">Required</span>
                        <span className="text-[10px] text-cyan-400 font-bold font-mono">TICKER</span>
                    </div>
                    <div className="flex flex-col gap-1 p-2 bg-slate-800/40 rounded border border-indigo-800/30 text-center">
                        <span className="text-[8px] text-slate-500 font-mono uppercase tracking-tighter">Required</span>
                        <span className="text-[10px] text-indigo-400 font-bold font-mono">WEIGHT %</span>
                    </div>
                    <div className="flex flex-col gap-1 p-2 bg-slate-800/40 rounded border border-slate-700/30 text-center">
                        <span className="text-[8px] text-slate-500 font-mono uppercase tracking-tighter">Optional</span>
                        <span className="text-[10px] text-slate-400 font-bold font-mono">ANY...</span>
                    </div>
                </div>

                <p className="text-[9px] text-slate-500 mt-2 font-medium italic border-t border-slate-800/50 pt-2 text-center uppercase tracking-tighter">
                    Prices fetched automatically · No price history needed
                </p>
            </div>
        </div>
    );
};

export default UploadPortfolio;
