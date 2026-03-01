"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import dynamic from "next/dynamic";
import {
  Upload,
  Zap,
  Shield,
  ShieldAlert,
  Activity,
  Eye,
  Layers,
  RotateCcw,
} from "lucide-react";

const WarpShaderHero = dynamic(
  () => import("@/components/ui/warp-shader"),
  { ssr: false }
);

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

interface AnalysisResult {
  prediction: string;
  anomaly_score: number;
  confidence: number;
  images: {
    original: string;
    resized: string;
    denoised: string;
    enhanced: string;
    normalized: string;
    gradcam: string;
    overlay: string;
  };
}

export default function HomePage() {
  const [state, setState] = useState<"idle" | "loading" | "results">("idle");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [samples, setSamples] = useState<string[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch(`${API}/sample_images`)
      .then((r) => r.json())
      .then(setSamples)
      .catch(() => { });
  }, []);

  const analyzeFile = useCallback(async (file: File) => {
    setState("loading");
    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await fetch(`${API}/analyze`, { method: "POST", body: fd });
      const data = await r.json();
      setResult(data);
      setState("results");
    } catch {
      alert("Analysis failed — is the Flask server running on :5000?");
      setState("idle");
    }
  }, []);

  const analyzeSample = useCallback(async (name: string) => {
    setState("loading");
    try {
      const r = await fetch(
        `${API}/analyze_sample/${encodeURIComponent(name)}`
      );
      const data = await r.json();
      setResult(data);
      setState("results");
    } catch {
      alert("Analysis failed — is the Flask server running on :5000?");
      setState("idle");
    }
  }, []);

  const reset = () => {
    setState("idle");
    setResult(null);
  };

  const isNormal = result?.prediction === "NORMAL";

  return (
    <main className="relative min-h-screen">
      <WarpShaderHero />

      <div className="relative z-10 min-h-screen">
        {/* ── Nav ──────────────────────────────────────────────── */}
        <nav className="sticky top-0 z-20 border-b border-white/10 backdrop-blur-2xl bg-black/50">
          <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)] animate-pulse" />
              <span className="font-bold text-sm text-white tracking-tight drop-shadow-lg">
                Thermal Analyzer
              </span>
            </div>
            <span className="text-[0.7rem] px-3 py-1 rounded-full border border-white/20 bg-black/30 text-white/80 font-medium backdrop-blur-sm">
              ResNet-18 + Bi-LSTM
            </span>
          </div>
        </nav>

        <div className="max-w-5xl mx-auto px-6 py-12">
          {/* ── IDLE ───────────────────────────────────────────── */}
          {state === "idle" && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="text-center mb-14 pt-12">
                <div
                  className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full
                    bg-black/30 border border-white/15 text-[0.78rem] text-cyan-200 mb-6
                    backdrop-blur-sm font-medium shadow-lg"
                >
                  <Zap className="w-3.5 h-3.5 text-amber-300" />
                  CNN-Based Pattern Analysis
                </div>

                <h1
                  className="text-5xl md:text-7xl font-[900] tracking-[-0.03em] leading-[1.05] mb-5"
                  style={{ textShadow: "0 4px 30px rgba(0,0,0,0.5)" }}
                >
                  <span className="text-white">Detect </span>
                  <span className="bg-gradient-to-r from-cyan-300 via-white to-amber-200 bg-clip-text text-transparent drop-shadow-2xl">
                    Thermal
                  </span>
                  <br />
                  <span className="bg-gradient-to-r from-amber-200 via-white to-cyan-300 bg-clip-text text-transparent drop-shadow-2xl">
                    Anomalies
                  </span>
                </h1>

                <p
                  className="text-white/90 text-lg max-w-lg mx-auto leading-relaxed font-light"
                  style={{ textShadow: "0 2px 12px rgba(0,0,0,0.6)" }}
                >
                  Upload an infrared thermal image and instantly detect abnormal
                  heat patterns using deep learning.
                </p>
              </div>

              {/* Upload Card */}
              <div
                className={`
                  relative max-w-xl mx-auto rounded-2xl border p-12 text-center cursor-pointer
                  transition-all duration-300 backdrop-blur-xl shadow-2xl
                  ${dragOver
                    ? "border-cyan-400 bg-cyan-500/15"
                    : "border-white/15 bg-black/40 hover:bg-black/50 hover:border-white/25"
                  }
                `}
                onClick={() => fileRef.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  if (e.dataTransfer.files.length)
                    analyzeFile(e.dataTransfer.files[0]);
                }}
              >
                <div className="w-16 h-16 mx-auto mb-4 rounded-xl bg-white/10 border border-white/10 flex items-center justify-center">
                  <Upload className="w-7 h-7 text-cyan-300" />
                </div>
                <h3 className="text-lg font-bold text-white mb-1">
                  Drop your thermal image here
                </h3>
                <p className="text-sm text-white/60 mb-5">
                  Supports JPG, PNG, BMP, TIFF
                </p>
                <button
                  className="px-7 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full text-sm font-bold text-white
                    hover:shadow-[0_12px_32px_rgba(6,182,212,0.35)] hover:-translate-y-0.5 transition-all duration-300"
                  onClick={(e) => {
                    e.stopPropagation();
                    fileRef.current?.click();
                  }}
                >
                  ↑ Choose File
                </button>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".jpg,.jpeg,.png,.bmp,.tiff,.tif"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files?.length)
                      analyzeFile(e.target.files[0]);
                  }}
                />
              </div>

              {/* Sample Chips */}
              {samples.length > 0 && (
                <div className="mt-8 text-center">
                  <p
                    className="text-[0.72rem] uppercase tracking-[0.1em] text-white/60 mb-3 font-semibold"
                    style={{ textShadow: "0 1px 6px rgba(0,0,0,0.5)" }}
                  >
                    Or try a sample from the dataset
                  </p>
                  <div className="flex flex-wrap gap-1.5 justify-center max-w-xl mx-auto">
                    {samples.map((name) => (
                      <button
                        key={name}
                        onClick={() => analyzeSample(name)}
                        className="px-3 py-1.5 text-[0.72rem] font-semibold rounded-lg
                          bg-black/35 border border-white/15 text-white/80 backdrop-blur-sm
                          hover:bg-black/50 hover:border-cyan-400/40 hover:text-white
                          transition-all duration-200 hover:-translate-y-px"
                      >
                        {name.replace(/\.[^/.]+$/, "")}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── LOADING ────────────────────────────────────────── */}
          {state === "loading" && (
            <div className="flex flex-col items-center justify-center py-32 animate-in fade-in duration-300">
              <div className="w-12 h-12 border-2 border-white/20 border-t-cyan-400 rounded-full animate-spin mb-6" />
              <p className="text-white font-medium" style={{ textShadow: "0 2px 8px rgba(0,0,0,0.5)" }}>
                Analyzing thermal patterns…
              </p>
              <p className="text-xs text-white/60 mt-1">
                CNN → LSTM → Anomaly Detector
              </p>
            </div>
          )}

          {/* ── RESULTS ────────────────────────────────────────── */}
          {state === "results" && result && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-6 pt-4">
              {/* Prediction Banner */}
              <div
                className={`rounded-2xl p-10 text-center border backdrop-blur-xl shadow-2xl ${isNormal
                  ? "bg-emerald-950/60 border-emerald-400/30"
                  : "bg-red-950/60 border-red-400/30"
                  }`}
              >
                <div className="mb-3">
                  {isNormal ? (
                    <Shield className="w-12 h-12 mx-auto text-emerald-300 drop-shadow-lg" />
                  ) : (
                    <ShieldAlert className="w-12 h-12 mx-auto text-red-300 drop-shadow-lg" />
                  )}
                </div>
                <div
                  className={`inline-flex px-5 py-1.5 rounded-full text-sm font-extrabold tracking-widest ${isNormal
                    ? "bg-emerald-400/20 text-emerald-200 border border-emerald-400/30"
                    : "bg-red-400/20 text-red-200 border border-red-400/30"
                    }`}
                >
                  {result.prediction}
                </div>
                <p className="text-white/80 mt-3 text-sm font-medium">
                  {isNormal
                    ? "Equipment operating within normal thermal parameters"
                    : "Abnormal thermal pattern detected — inspection recommended"}
                </p>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 gap-3">
                <StatCard
                  icon={Activity}
                  label="Anomaly Score"
                  value={result.anomaly_score.toFixed(4)}
                />
                <StatCard
                  icon={Eye}
                  label="Confidence"
                  value={`${result.confidence.toFixed(1)}%`}
                />
                <StatCard
                  icon={isNormal ? Shield : ShieldAlert}
                  label="Verdict"
                  value={result.prediction}
                  valueColor={isNormal ? "text-emerald-300" : "text-red-300"}
                />
              </div>

              {/* Grad-CAM */}
              <div>
                <SectionHeader icon={Eye} label="Grad-CAM Attention Analysis" />
                <div className="grid grid-cols-3 gap-3">
                  <ImgTile src={result.images.enhanced} label="Enhanced Input" />
                  <ImgTile src={result.images.gradcam} label="Attention Map" />
                  <ImgTile src={result.images.overlay} label="Overlay" />
                </div>
              </div>

              {/* Preprocessing */}
              <div>
                <SectionHeader icon={Layers} label="Preprocessing Pipeline" />
                <div className="grid grid-cols-4 gap-3">
                  <ImgTile src={result.images.original} label="Original" />
                  <ImgTile src={result.images.resized} label="Resized" />
                  <ImgTile src={result.images.denoised} label="Denoised" />
                  <ImgTile src={result.images.normalized} label="Normalized" />
                </div>
              </div>

              {/* Reset */}
              <div className="text-center pt-4 pb-8">
                <button
                  onClick={reset}
                  className="inline-flex items-center gap-2 px-8 py-3 rounded-full
                    border border-white/20 bg-black/30 backdrop-blur-sm
                    text-white/80 text-sm font-semibold
                    hover:border-cyan-400/40 hover:text-white hover:bg-black/50
                    transition-all duration-300"
                >
                  <RotateCcw className="w-4 h-4" />
                  Analyze Another Image
                </button>
              </div>
            </div>
          )}
        </div>

        <footer className="border-t border-white/8 py-6 text-center text-[0.72rem] text-white/40 font-medium backdrop-blur-sm bg-black/20">
          CNN-Based Thermal Pattern Analysis of Power Transformers
        </footer>
      </div>
    </main>
  );
}

/* ── Sub-components ────────────────────────────────────────────────── */

function StatCard({
  icon: Icon,
  label,
  value,
  valueColor,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  valueColor?: string;
}) {
  return (
    <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-5 text-center shadow-xl">
      <Icon className="w-4 h-4 mx-auto mb-2 text-cyan-300/70" />
      <p className="text-[0.65rem] uppercase tracking-[0.08em] text-white/50 font-semibold mb-2">
        {label}
      </p>
      <p className={`text-2xl font-extrabold ${valueColor || "text-white"}`}>
        {value}
      </p>
    </div>
  );
}

function SectionHeader({
  icon: Icon,
  label,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
}) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <Icon className="w-4 h-4 text-cyan-300/80" />
      <span
        className="text-sm font-bold text-white"
        style={{ textShadow: "0 1px 6px rgba(0,0,0,0.4)" }}
      >
        {label}
      </span>
      <div className="flex-1 h-px bg-white/10" />
    </div>
  );
}

function ImgTile({ src, label }: { src: string; label: string }) {
  return (
    <div
      className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden
        group hover:-translate-y-1 hover:border-cyan-400/30 hover:shadow-2xl transition-all duration-300"
    >
      <img
        src={`data:image/png;base64,${src}`}
        alt={label}
        className="w-full aspect-square object-contain bg-black/60"
      />
      <div className="px-3 py-2.5 text-center text-[0.72rem] font-semibold text-white/70 border-t border-white/8">
        {label}
      </div>
    </div>
  );
}
