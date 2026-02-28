import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, CheckCircle2, FileSpreadsheet } from 'lucide-react';

const UploadPortfolio = ({ onUploadSuccess }) => {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    const handleFileChange = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        setIsLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);
        // Default parameters
        formData.append('confidence_level', '0.95');
        formData.append('simulations', '5000'); // Reduced for faster feedback

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
            <h2 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wider relative z-10">Portfolio Data</h2>

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
                        <p className="text-xs text-cyan-400 font-medium tracking-tight">Processing Matrix...</p>
                    </div>
                ) : (
                    <>
                        <div className="relative">
                            <div className="absolute inset-0 bg-cyan-500/20 blur-xl rounded-full scale-150 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                            <div className="p-3 bg-slate-800/80 rounded-full mb-3 group-hover:bg-cyan-900/50 transition-colors border border-slate-700 group-hover:border-cyan-700 relative z-10">
                                <UploadCloud className="w-6 h-6 text-slate-400 group-hover:text-cyan-400 transition-colors" />
                            </div>
                        </div>
                        <p className="text-sm font-medium text-slate-200 text-center">
                            {error ? 'Try Again' : 'Upload Portfolio CSV'}
                        </p>
                        {error && <p className="text-[10px] text-rose-400 mt-1 max-w-[150px] text-center line-clamp-1">{error}</p>}
                    </>
                )}
            </div>

            <div className="mt-4 bg-gradient-to-r from-emerald-950/40 to-slate-900/40 border border-emerald-900/30 rounded-lg p-3 flex items-start gap-3 relative z-10 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                <div className="relative mt-0.5">
                    <span className="absolute inset-0 bg-emerald-500 blur-sm opacity-40 rounded-full"></span>
                    <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0 relative z-10" />
                </div>
                <div>
                    <p className="text-xs font-semibold text-emerald-300 tracking-wide">Standardized Format</p>
                    <p className="text-[10px] text-slate-400 mt-0.5 font-mono">DATE • ASSETS... • BENCHMARK</p>
                </div>
            </div>
        </div>
    );
};

export default UploadPortfolio;
