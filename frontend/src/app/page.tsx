"use client";

import React, { useState, useRef, DragEvent } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [caption, setCaption] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragActive, setIsDragActive] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0];
      if (selectedFile.type.startsWith("image/")) {
        setFile(selectedFile);
        setPreviewUrl(URL.createObjectURL(selectedFile));
        setCaption(null);
        setError(null);
      } else {
        setError("Please select a valid image file (PNG, JPG, JPEG).");
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setCaption(null);
      setError(null);
    }
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleGenerate = async () => {
    if (!file) return;
    
    setIsLoading(true);
    setError(null);
    setCaption(null);
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/caption";
      const response = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with ${response.status}`);
      }
      
      const data = await response.json();
      if (data.success) {
        setCaption(data.caption);
      } else {
        throw new Error("API returned success: false");
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to communicate with the captioning server. Please ensure the backend is running at http://localhost:8000.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setCaption(null);
    setError(null);
    setCopied(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleCopy = () => {
    if (caption) {
      navigator.clipboard.writeText(caption);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="relative min-h-screen bg-[#0B0F19] text-[#EDF2F7] overflow-hidden flex flex-col font-sans">
      {/* Decorative background blobs */}
      <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-500/10 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-violet-600/15 blur-[120px] pointer-events-none" />
      
      {/* Grid pattern overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f293708_1px,transparent_1px),linear-gradient(to_bottom,#1f293708_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none" />
      
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-gray-900 bg-[#0B0F19]/80 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-blue-500 to-violet-600 shadow-lg shadow-blue-500/25">
              <span className="font-extrabold text-white text-base">P</span>
            </div>
            <span className="text-xl font-bold tracking-tight bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">Pixel_Info</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="hidden sm:inline-block text-xs font-semibold px-3 py-1 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 shadow-inner">
              Active Model: ResNet50 + LSTM
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 mx-auto w-full max-w-4xl px-6 py-12 flex flex-col items-center justify-center relative z-10">
        <div className="text-center mb-12 max-w-2xl">
          <h2 className="text-4xl font-extrabold tracking-tight bg-gradient-to-b from-white to-gray-400 bg-clip-text text-transparent sm:text-5xl mb-4 leading-normal">
            AI Image Caption Generator
          </h2>
          <p className="text-lg text-gray-400 leading-relaxed">
            Upload any image and let our deep learning model generate a natural language caption using advanced Beam Search decoding.
          </p>
        </div>

        {/* Upload Zone */}
        {!file ? (
          <div 
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`w-full aspect-[16/10] max-w-2xl rounded-2xl border-2 border-dashed flex flex-col items-center justify-center p-8 transition-all cursor-pointer bg-gray-950/20 backdrop-blur-sm shadow-xl ${
              isDragActive 
                ? "border-blue-500 bg-blue-500/5 shadow-blue-500/5 scale-[1.01]" 
                : "border-gray-800 hover:border-gray-700 hover:bg-gray-900/30"
            }`}
            onClick={handleBrowseClick}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange}
              accept="image/*"
              className="hidden"
            />
            
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gray-950 border border-gray-800 mb-5 text-gray-400 shadow-md">
              <svg className="h-7 w-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
            </div>
            
            <p className="text-lg font-semibold text-gray-200 mb-1">
              Drag & drop image here
            </p>
            <p className="text-sm text-gray-500 mb-5">
              Supports PNG, JPG, or JPEG
            </p>
            <button className="px-5 py-2.5 rounded-xl bg-gray-950 border border-gray-800 text-sm font-bold hover:bg-gray-900 hover:text-white transition-colors shadow-sm">
              Browse Files
            </button>
          </div>
        ) : (
          /* Preview & Results Container */
          <div className="w-full max-w-2xl rounded-2xl border border-gray-800 bg-gray-950/40 backdrop-blur-sm p-6 shadow-2xl flex flex-col gap-6">
            <div className="relative w-full aspect-[16/10] rounded-xl overflow-hidden border border-gray-800 bg-black/40">
              {previewUrl && (
                <img 
                  src={previewUrl} 
                  alt="Selected preview" 
                  className="w-full h-full object-contain"
                />
              )}
              <button 
                onClick={handleClear}
                className="absolute top-3 right-3 p-2.5 rounded-xl bg-black/60 hover:bg-black/85 backdrop-blur-sm text-gray-400 hover:text-white transition-colors border border-white/10"
                title="Clear image"
              >
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="flex flex-col gap-4">
              {!caption && !isLoading && !error && (
                <button 
                  onClick={handleGenerate}
                  className="w-full py-4 px-4 rounded-xl bg-gradient-to-r from-blue-500 to-violet-600 text-sm font-bold text-white shadow-lg shadow-blue-500/20 hover:opacity-95 active:scale-[0.99] transition-all"
                >
                  Generate Caption
                </button>
              )}
              
              {isLoading && (
                <div className="flex flex-col items-center justify-center py-6 gap-4">
                  <div className="relative flex items-center justify-center">
                    <div className="h-10 w-10 animate-spin rounded-full border-4 border-solid border-blue-500 border-t-transparent" />
                    <div className="absolute h-10 w-10 rounded-full border-4 border-solid border-blue-500/10 pointer-events-none" />
                  </div>
                  <p className="text-sm font-semibold text-gray-400 animate-pulse">Running Beam Search Inference...</p>
                </div>
              )}
              
              {error && (
                <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/25 text-red-400 text-sm flex items-start gap-3">
                  <svg className="h-5 w-5 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div className="flex-1">
                    <p className="font-bold mb-1">Error Occurred</p>
                    <p className="text-gray-400 text-xs leading-relaxed">{error}</p>
                    <button 
                      onClick={handleGenerate}
                      className="mt-3 px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/35 text-red-300 text-xs font-bold transition-colors border border-red-500/30"
                    >
                      Try Again
                    </button>
                  </div>
                </div>
              )}
              
              {caption && (
                <div className="flex flex-col gap-3">
                  <span className="text-xs font-bold uppercase tracking-wider text-blue-400 px-1">Generated Caption</span>
                  <div className="relative p-5 rounded-xl border border-gray-800 bg-gray-900/30 backdrop-blur-sm flex items-start justify-between gap-4 shadow-inner">
                    <p className="text-lg font-semibold text-gray-100 leading-relaxed capitalize">
                      {caption}
                    </p>
                    <button 
                      onClick={handleCopy}
                      className={`p-2.5 rounded-xl border transition-colors shrink-0 ${
                        copied 
                          ? "bg-green-500/10 border-green-500/30 text-green-400" 
                          : "bg-gray-950 border-gray-800 text-gray-400 hover:text-white hover:bg-gray-900"
                      }`}
                      title="Copy caption"
                    >
                      {copied ? (
                        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                        </svg>
                      ) : (
                        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                        </svg>
                      )}
                    </button>
                  </div>
                  <button 
                    onClick={handleClear}
                    className="w-full py-3 px-4 rounded-xl bg-gray-950 hover:bg-gray-900 border border-gray-800 text-sm font-bold text-gray-400 hover:text-white active:scale-[0.99] transition-all"
                  >
                    Upload Another Image
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="w-full border-t border-gray-900 py-6 text-center text-xs text-gray-600 bg-gray-950/20 relative z-10">
        <p>© 2026 Pixel_Info. Developed for production image captioning workflows.</p>
      </footer>
    </div>
  );
}
