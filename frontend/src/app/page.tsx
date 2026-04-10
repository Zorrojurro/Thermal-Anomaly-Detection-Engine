"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  Upload,
  Zap,
  Shield,
  ShieldAlert,
  Activity,
  Eye,
  Layers,
  RotateCcw,
  Cpu,
  Play,
  X,
  Thermometer as ThermometerIcon,
} from "lucide-react";

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
  const [showVideo, setShowVideo] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    fetch(`${API}/sample_images`)
      .then((r) => r.json())
      .then(setSamples)
      .catch(() => { });
  }, []);

  // Close video modal on Escape key
  useEffect(() => {
    if (!showVideo) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setShowVideo(false);
        videoRef.current?.pause();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [showVideo]);

  const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  const analyzeFile = useCallback(async (file: File) => {
    setState("loading");
    const fd = new FormData();
    fd.append("file", file);
    try {
      const [r] = await Promise.all([
        fetch(`${API}/analyze`, { method: "POST", body: fd }),
        delay(5000),
      ]);
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
      const [r] = await Promise.all([
        fetch(`${API}/analyze_sample/${encodeURIComponent(name)}`),
        delay(5000),
      ]);
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
    <main className="min-h-screen">
      {/* ── Nav ───────────────────────────────── */}
      <nav className="sticky top-0 z-20 border-b border-[#27187E]/8 bg-[#F7F7FF]">
        <div className="max-w-6xl mx-auto px-6 h-12 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_6px_rgba(52,211,153,0.5)] animate-pulse" />
            <span className="font-bold text-sm text-[#27187E] tracking-tight">
              Thermal Analyzer
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-[#758BFD]" />
            <span className="text-[0.7rem] px-3 py-1 rounded-full border border-[#27187E]/12 bg-[#27187E]/5 text-[#27187E]/70 font-medium">
              ResNet-18 + Bi-LSTM
            </span>
          </div>
        </div>
      </nav>

      {/* ── IDLE STATE ───────────────────────── */}
      {state === "idle" && (
        <div className="h-[calc(100vh-49px)] flex flex-col justify-center">
          <div className="max-w-6xl mx-auto px-6 w-full">
            <div className="grid md:grid-cols-2 gap-10 items-center">
              {/* Left — Hero Text */}
              <div>
                <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#27187E]/8 border border-[#27187E]/10 text-[0.78rem] text-[#27187E] mb-5 font-medium">
                  <Zap className="w-3.5 h-3.5 text-amber-500" />
                  CNN-Based Pattern Analysis
                </div>

                <h1 className="text-4xl md:text-5xl font-[900] tracking-[-0.03em] leading-[1.1] mb-4 text-[#27187E]">
                  Detect Thermal{" "}
                  <span className="bg-gradient-to-r from-[#758BFD] to-[#27187E] bg-clip-text text-transparent">
                    Anomalies
                  </span>
                </h1>

                <p className="text-[#27187E]/60 text-base leading-relaxed mb-6 max-w-md">
                  Upload an infrared thermal image and instantly detect abnormal
                  heat patterns using deep learning.
                </p>

                {/* Watch Demo Button */}
                <button
                  onClick={() => setShowVideo(true)}
                  className="group inline-flex items-center gap-2.5 px-5 py-2.5 mb-6
                    rounded-full border border-[#758BFD]/25 bg-white
                    text-[#27187E] text-sm font-semibold
                    hover:bg-[#758BFD]/8 hover:border-[#758BFD]/50 hover:shadow-lg hover:-translate-y-0.5
                    transition-all duration-300"
                >
                  <span className="w-8 h-8 rounded-full bg-gradient-to-br from-[#758BFD] to-[#27187E] flex items-center justify-center
                    shadow-[0_0_12px_rgba(117,139,253,0.4)] group-hover:shadow-[0_0_20px_rgba(117,139,253,0.6)]
                    transition-shadow duration-300">
                    <Play className="w-3.5 h-3.5 text-white ml-0.5" fill="white" />
                  </span>
                  Watch Project Demo
                </button>

                {/* Sample chips */}
                {samples.length > 0 && (
                  <div>
                    <p className="text-[0.72rem] uppercase tracking-[0.1em] text-[#27187E]/40 mb-2.5 font-semibold">
                      Try a sample
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {samples.map((name) => (
                        <button
                          key={name}
                          onClick={() => analyzeSample(name)}
                          className="px-3 py-1.5 text-[0.72rem] font-semibold rounded-lg
                            bg-[#27187E]/6 border border-[#27187E]/10 text-[#27187E]/70
                            hover:bg-[#27187E]/12 hover:border-[#758BFD]/30 hover:text-[#27187E]
                            transition-all duration-200"
                        >
                          {name.replace(/\.[^/.]+$/, "")}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Right — Upload Card */}
              <div
                className={`
                  rounded-2xl border-2 border-dashed p-10 text-center cursor-pointer
                  transition-all duration-300 shadow-sm
                  ${dragOver
                    ? "border-[#758BFD] bg-[#758BFD]/8"
                    : "border-[#27187E]/15 bg-white hover:bg-[#758BFD]/4 hover:border-[#758BFD]/40"
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
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-[#27187E]/8 flex items-center justify-center">
                  <Upload className="w-7 h-7 text-[#27187E]/60" />
                </div>
                <h3 className="text-lg font-bold text-[#27187E] mb-1">
                  Drop your thermal image here
                </h3>
                <p className="text-sm text-[#27187E]/45 mb-6">
                  Supports JPG, PNG, BMP, TIFF
                </p>
                <button
                  className="px-8 py-2.5 bg-[#27187E] rounded-full text-sm font-bold text-white
                    hover:bg-[#27187E]/90 hover:shadow-lg hover:-translate-y-0.5 transition-all duration-300"
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
                    if (e.target.files?.length) analyzeFile(e.target.files[0]);
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── LOADING STATE ────────────────────── */}
      {state === "loading" && <LoadingSequence />}

      {/* ── RESULTS STATE ────────────────────── */}
      {state === "results" && result && (
        <div className="max-w-5xl mx-auto px-6 py-6 space-y-5">
          {/* Prediction Banner */}
          <div
            className={`rounded-2xl p-8 text-center border shadow-sm ${isNormal
                ? "bg-emerald-50 border-emerald-200"
                : "bg-red-50 border-red-200"
              }`}
          >
            <div className="mb-2">
              {isNormal ? (
                <Shield className="w-10 h-10 mx-auto text-emerald-500" />
              ) : (
                <ShieldAlert className="w-10 h-10 mx-auto text-red-500" />
              )}
            </div>
            <div
              className={`inline-flex px-5 py-1.5 rounded-full text-sm font-extrabold tracking-widest ${isNormal
                  ? "bg-emerald-100 text-emerald-700 border border-emerald-200"
                  : "bg-red-100 text-red-700 border border-red-200"
                }`}
            >
              {result.prediction}
            </div>
            <p className="text-[#27187E]/60 mt-2 text-sm font-medium">
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
              valueColor={isNormal ? "text-emerald-600" : "text-red-600"}
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
          <div className="text-center pt-2 pb-6">
            <button
              onClick={reset}
              className="inline-flex items-center gap-2 px-8 py-2.5 rounded-full
                border border-[#27187E]/15 bg-white
                text-[#27187E]/70 text-sm font-semibold
                hover:border-[#758BFD]/30 hover:text-[#27187E] hover:shadow-md
                transition-all duration-300"
            >
              <RotateCcw className="w-4 h-4" />
              Analyze Another Image
            </button>
          </div>
        </div>
      )}

      {/* ── Video Modal ───────────────────────── */}
      {showVideo && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          onClick={() => setShowVideo(false)}
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-[#27187E]/80 backdrop-blur-md animate-[fadeIn_0.3s_ease]" />

          {/* Modal Content */}
          <div
            className="relative z-10 w-full max-w-4xl mx-6 animate-[scaleIn_0.35s_ease]"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close Button */}
            <button
              onClick={() => setShowVideo(false)}
              className="absolute -top-12 right-0 w-9 h-9 rounded-full
                bg-white/10 border border-white/20
                flex items-center justify-center
                text-white/80 hover:bg-white/20 hover:text-white
                transition-all duration-200"
            >
              <X className="w-4 h-4" />
            </button>

            {/* Video Container */}
            <div className="rounded-2xl overflow-hidden border border-white/10 shadow-2xl shadow-black/40 bg-black">
              {/* Header Bar */}
              <div className="bg-gradient-to-r from-[#27187E] to-[#758BFD] px-5 py-3 flex items-center gap-3">
                <div className="w-7 h-7 rounded-lg bg-white/15 flex items-center justify-center">
                  <Play className="w-3.5 h-3.5 text-white ml-0.5" fill="white" />
                </div>
                <div>
                  <p className="text-white text-sm font-bold tracking-tight">Project Demo</p>
                  <p className="text-white/60 text-[0.65rem] font-medium">AI-Powered Thermal Monitoring for Power Transformers</p>
                </div>
              </div>

              {/* Video Player */}
              <video
                ref={videoRef}
                src="/AI_Transformer_Monitoring.mp4"
                controls
                autoPlay
                className="w-full aspect-video"
                playsInline
              />
            </div>

            {/* Hint */}
            <p className="text-center text-white/40 text-[0.7rem] mt-3 font-medium">
              Press Esc or click outside to close
            </p>
          </div>
        </div>
      )}

      {/* ── Footer ───────────────────────────── */}
      {state !== "idle" && (
        <footer className="border-t border-[#27187E]/6 py-4 text-center text-[0.72rem] text-[#27187E]/35 font-medium">
          CNN-Based Thermal Pattern Analysis of Power Transformers
        </footer>
      )}
    </main>
  );
}

/* ── Sub-components ──────────────────────────────────────────────── */

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
    <div className="bg-white border border-[#27187E]/8 rounded-2xl p-5 text-center shadow-sm">
      <Icon className="w-4 h-4 mx-auto mb-2 text-[#758BFD]" />
      <p className="text-[0.65rem] uppercase tracking-[0.08em] text-[#27187E]/45 font-semibold mb-2">
        {label}
      </p>
      <p className={`text-2xl font-extrabold ${valueColor || "text-[#27187E]"}`}>
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
    <div className="flex items-center gap-3 mb-3">
      <Icon className="w-4 h-4 text-[#758BFD]" />
      <span className="text-sm font-bold text-[#27187E]">{label}</span>
      <div className="flex-1 h-px bg-[#27187E]/8" />
    </div>
  );
}

function ImgTile({ src, label }: { src: string; label: string }) {
  return (
    <div
      className="bg-white border border-[#27187E]/8 rounded-2xl overflow-hidden shadow-sm
        group hover:-translate-y-1 hover:border-[#758BFD]/25 hover:shadow-md transition-all duration-300"
    >
      <img
        src={`data:image/png;base64,${src}`}
        alt={label}
        className="w-full aspect-square object-contain bg-[#F7F7FF]"
      />
      <div className="px-3 py-2.5 text-center text-[0.72rem] font-semibold text-[#27187E]/60 border-t border-[#27187E]/6">
        {label}
      </div>
    </div>
  );
}

function LoadingSequence() {
  const messages = ["Loading thermal image...", "Removing noise...", "Resizing...", "Detecting anomaly..."];
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prevIndex) => (prevIndex + 1) % messages.length);
    }, 1250); // 5000ms total / 4 messages = 1250ms per message

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col flex-1 items-center justify-center p-8 text-center bg-white/50 border border-[#27187E]/8 rounded-3xl min-h-[400px]">
      <div className="relative w-24 h-24 mb-6">
        <svg className="animate-spin w-full h-full text-[#758BFD]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <ThermometerIcon className="w-8 h-8 text-[#27187E]" />
        </div>
      </div>
      <div className="h-6 overflow-hidden relative w-full flex justify-center">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`absolute transition-all duration-500 ease-in-out font-medium text-[#27187E]/70
              ${i === index ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}
          >
            {msg}
          </div>
        ))}
      </div>
      <p className="mt-2 text-sm text-[#27187E]/50 font-medium animate-pulse">
        Processing frame...
      </p>
    </div>
  );
}
