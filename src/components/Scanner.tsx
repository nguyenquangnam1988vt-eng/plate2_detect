import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as ort from 'onnxruntime-web';
import { Play, Square, ShieldAlert } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/src/lib/utils';

// Constants
const SIZE = 320;                     // input size for both models
const CONF_THRESH = 0.5;              // confidence threshold
const IOU_THRESH = 0.45;              // NMS IoU threshold
const DETECT_INTERVAL_MS = 900;       // min time between detections

// Mapping for character classes (0-30)
const CHAR_MAP = [
  "0","1","2","3","4","5","6","7","8","9",
  "A","B","C","D","E","F","G","H","K","L",
  "M","N","P","R","S","T","U","V","X","Y","Z"
];

// Google Apps Script URL – replace with your own
const SHEET_URL = "https://script.google.com/macros/s/AKfycbx_nl-R8ggO9ZF_jc7VMnh2YitgssH0kFDQ7xNsx5aZEEJQGHofmfeIh-CVr2cuauhX/exec";

// Reusable buffer for preprocessing (avoid allocations)
let inputBuffer = new Float32Array(3 * SIZE * SIZE);

export default function Scanner() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cropCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [sessionPlate, setSessionPlate] = useState<ort.InferenceSession | null>(null);
  const [sessionChar, setSessionChar] = useState<ort.InferenceSession | null>(null);
  const [modelReady, setModelReady] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState("");
  const [fps, setFps] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const processingRef = useRef(false);
  const lastDetectRef = useRef(0);
  const frameIdRef = useRef<number>(0);
  const streamRef = useRef<MediaStream | null>(null);

  // ---------- Helper: preprocess image/video to tensor ----------
  const preprocess = useCallback((source: HTMLVideoElement | HTMLCanvasElement) => {
    const canvas = offscreenCanvasRef.current!;
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    canvas.width = SIZE;
    canvas.height = SIZE;

    const srcW = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
    const srcH = source instanceof HTMLVideoElement ? source.videoHeight : source.height;
    if (srcW === 0 || srcH === 0) throw new Error("Invalid source dimensions");

    const ratio = Math.min(SIZE / srcW, SIZE / srcH);
    const newW = srcW * ratio;
    const newH = srcH * ratio;
    const dx = (SIZE - newW) / 2;
    const dy = (SIZE - newH) / 2;

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.drawImage(source, dx, dy, newW, newH);

    const imgData = ctx.getImageData(0, 0, SIZE, SIZE).data;
    // Fill buffer in CHW format
    for (let i = 0, p = 0; i < imgData.length; i += 4, p++) {
      inputBuffer[p] = imgData[i] / 255;               // R
      inputBuffer[p + SIZE * SIZE] = imgData[i + 1] / 255; // G
      inputBuffer[p + 2 * SIZE * SIZE] = imgData[i + 2] / 255; // B
    }
    return new ort.Tensor("float32", inputBuffer.slice(), [1, 3, SIZE, SIZE]);
  }, []);

  // ---------- Non‑Maximum Suppression ----------
  const nms = useCallback((boxes: any[]) => {
    if (!boxes.length) return [];
    boxes.sort((a, b) => b.confidence - a.confidence);
    const result: any[] = [];
    const remaining = [...boxes];
    while (remaining.length) {
      const best = remaining.shift()!;
      result.push(best);
      for (let i = remaining.length - 1; i >= 0; i--) {
        const b = remaining[i];
        const x1 = Math.max(best.x, b.x);
        const y1 = Math.max(best.y, b.y);
        const x2 = Math.min(best.x + best.w, b.x + b.w);
        const y2 = Math.min(best.y + best.h, b.y + b.h);
        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const union = best.w * best.h + b.w * b.h - inter;
        const iou = union > 0 ? inter / union : 0;
        if (iou > IOU_THRESH) remaining.splice(i, 1);
      }
    }
    return result;
  }, []);

  // ---------- Parse YOLO output (common for both models) ----------
  const parseYoloOutput = useCallback((
    outputData: Float32Array,
    dims: number[],
    imgW: number,
    imgH: number
  ) => {
    const numBoxes = dims[2];
    const numAttrs = dims[1];
    const boxes: any[] = [];
    for (let i = 0; i < numBoxes; i++) {
      const cx = outputData[i];
      const cy = outputData[i + numBoxes];
      const w = outputData[i + 2 * numBoxes];
      const h = outputData[i + 3 * numBoxes];
      const objConf = outputData[i + 4 * numBoxes];
      let maxScore = 0;
      let bestClass = 0;
      for (let c = 5; c < numAttrs; c++) {
        const score = outputData[i + c * numBoxes];
        if (score > maxScore) {
          maxScore = score;
          bestClass = c - 5;
        }
      }
      const confidence = objConf * maxScore;
      if (confidence < CONF_THRESH) continue;

      const ratio = Math.min(SIZE / imgW, SIZE / imgH);
      const newW = imgW * ratio;
      const newH = imgH * ratio;
      const dx = (SIZE - newW) / 2;
      const dy = (SIZE - newH) / 2;

      let x = (cx - w / 2 - dx) / ratio;
      let y = (cy - h / 2 - dy) / ratio;
      let bw = w / ratio;
      let bh = h / ratio;

      x = Math.max(0, x);
      y = Math.max(0, y);
      bw = Math.min(imgW - x, bw);
      bh = Math.min(imgH - y, bh);

      if (bw > 10 && bh > 10) {
        boxes.push({ x, y, w: bw, h: bh, confidence, classId: bestClass });
      }
    }
    return boxes;
  }, []);

  // ---------- Character recognition from cropped plate ----------
  const recognizePlateText = useCallback(async (cropCanvas: HTMLCanvasElement) => {
    if (!sessionChar) return "";
    const tensor = preprocess(cropCanvas);
    const feeds: Record<string, ort.Tensor> = {};
    feeds[sessionChar.inputNames[0]] = tensor;
    const results = await sessionChar.run(feeds);
    const outputData = results[sessionChar.outputNames[0]].data as Float32Array;
    const dims = results[sessionChar.outputNames[0]].dims;

    // Assume character model is also YOLO (detects individual chars)
    let charBoxes = parseYoloOutput(outputData, dims, cropCanvas.width, cropCanvas.height);
    if (!charBoxes.length) return "";
    charBoxes = nms(charBoxes);
    // Filter out class 31 if present (license plate class)
    charBoxes = charBoxes.filter(b => b.classId !== 31);
    if (!charBoxes.length) return "";
    charBoxes.sort((a, b) => a.x - b.x);
    let plateText = "";
    for (const box of charBoxes) {
      if (box.classId >= 0 && box.classId < CHAR_MAP.length) {
        plateText += CHAR_MAP[box.classId];
      } else {
        plateText += "?";
      }
    }
    return plateText;
  }, [sessionChar, preprocess, parseYoloOutput, nms]);

  // ---------- Main detection loop (one iteration) ----------
  const detectOnce = useCallback(async () => {
    if (!sessionPlate || !sessionChar || !scanning || processingRef.current) return;
    const video = videoRef.current;
    if (!video || video.readyState !== 4 || video.videoWidth === 0) return;

    const now = Date.now();
    if (now - lastDetectRef.current < DETECT_INTERVAL_MS) return;
    lastDetectRef.current = now;

    processingRef.current = true;
    try {
      // 1. Detect license plate
      const plateTensor = preprocess(video);
      const plateFeeds = { [sessionPlate.inputNames[0]]: plateTensor };
      const plateResult = await sessionPlate.run(plateFeeds);
      const plateOutput = plateResult[sessionPlate.outputNames[0]].data as Float32Array;
      const plateDims = plateResult[sessionPlate.outputNames[0]].dims;

      let plateBoxes = parseYoloOutput(plateOutput, plateDims, video.videoWidth, video.videoHeight);
      if (!plateBoxes.length) return;
      plateBoxes = nms(plateBoxes);
      const bestPlate = plateBoxes[0];

      // Draw bounding box on canvas
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d")!;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "#00FF9C";
        ctx.lineWidth = 4;
        ctx.strokeRect(bestPlate.x, bestPlate.y, bestPlate.w, bestPlate.h);
      }

      // Crop plate region
      const cropCanvas = cropCanvasRef.current!;
      cropCanvas.width = bestPlate.w;
      cropCanvas.height = bestPlate.h;
      const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true })!;
      cropCtx.drawImage(video, bestPlate.x, bestPlate.y, bestPlate.w, bestPlate.h, 0, 0, bestPlate.w, bestPlate.h);

      // 2. Recognize characters
      const plateText = await recognizePlateText(cropCanvas);
      if (plateText && plateText.length >= 3) {
        const formatted = `${plateText.slice(0, 3)}-${plateText.slice(3, 8)}`;
        setResult(formatted);
        // Send to Google Sheets (non-blocking)
        fetch(SHEET_URL, {
          method: "POST",
          mode: "no-cors",
          body: JSON.stringify({ plate: formatted, timestamp: new Date().toISOString() })
        }).catch(e => console.warn("Sheet send error", e));
      }

      // Update FPS indicator
      const elapsed = Date.now() - now;
      setFps(Math.round(1000 / Math.max(elapsed, 10)));
    } catch (err) {
      console.error("Detection error:", err);
    } finally {
      processingRef.current = false;
    }
  }, [sessionPlate, sessionChar, scanning, preprocess, parseYoloOutput, nms, recognizePlateText]);

  // ---------- Animation loop (drives detection periodically) ----------
  const loop = useCallback(() => {
    if (!scanning) return;
    detectOnce();
    frameIdRef.current = requestAnimationFrame(loop);
  }, [scanning, detectOnce]);

  useEffect(() => {
    if (scanning) {
      frameIdRef.current = requestAnimationFrame(loop);
    } else {
      if (frameIdRef.current) cancelAnimationFrame(frameIdRef.current);
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
    return () => {
      if (frameIdRef.current) cancelAnimationFrame(frameIdRef.current);
    };
  }, [scanning, loop]);

  // ---------- Camera & Model initialization ----------
  useEffect(() => {
    // Create offscreen canvases
    offscreenCanvasRef.current = document.createElement("canvas");
    cropCanvasRef.current = document.createElement("canvas");

    // ONNX Runtime configuration for Safari (SIMD off, single thread)
    ort.env.wasm.simd = false;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";

    const loadModels = async () => {
      try {
        const plate = await ort.InferenceSession.create("/model/bienso1.onnx", { executionProviders: ["wasm"] });
        const charModel = await ort.InferenceSession.create("/model/character.onnx", { executionProviders: ["wasm"] });
        setSessionPlate(plate);
        setSessionChar(charModel);
        setModelReady(true);
        console.log("✅ Models loaded");
      } catch (err) {
        console.error("Model load error", err);
        setError("Không thể tải model ONNX. Kiểm tra thư mục /public/model/");
      }
    };
    loadModels();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // ---------- Start / Stop scanning ----------
  const startScan = async () => {
    if (!modelReady) {
      setError("Model chưa sẵn sàng");
      return;
    }
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setScanning(true);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Không thể truy cập camera sau");
    }
  };

  const stopScan = () => {
    setScanning(false);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
  };

  return (
    <div className="min-h-screen flex flex-col bg-bg text-ink overflow-hidden">
      <header className="h-20 border-b border-white/10 flex items-center justify-between px-6 md:px-10">
        <div className="text-xl md:text-2xl font-black tracking-tighter uppercase">ALPR.SCAN.01</div>
        <div className="flex items-center gap-2 text-accent text-xs font-bold uppercase tracking-widest">
          <div className="w-2 h-2 bg-accent rounded-full animate-pulse shadow-[0_0_10px_#00FF9C]" />
          System Live
        </div>
      </header>

      <main className="flex-1 scanner-grid overflow-hidden">
        <section className="p-6 md:p-10 flex flex-col gap-6 bg-radial-gradient">
          <div className="label-micro">Lens Capture</div>
          <div className="flex-1 relative bg-black border border-white/10 overflow-hidden rounded-xl">
            <video
              ref={videoRef}
              className="absolute inset-0 w-full h-full object-cover opacity-50"
              muted
              playsInline
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full object-cover z-10"
            />
            <AnimatePresence>
              {!scanning && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 flex items-center justify-center bg-black/80 z-20 backdrop-blur-sm"
                >
                  <button
                    onClick={startScan}
                    disabled={!modelReady}
                    className="flex flex-col items-center gap-4 text-accent transition-transform hover:scale-105 active:scale-95"
                  >
                    <div className="w-24 h-24 border-2 border-accent rounded-full flex items-center justify-center">
                      <Play className="w-10 h-10 ml-1" />
                    </div>
                    <span className="label-micro">Start Neural Engine</span>
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </section>

        <aside className="bg-surface p-6 md:p-10 flex flex-col border-l border-white/10">
          <div className="label-micro mb-4">Identification</div>
          <div className="min-h-[160px] flex items-center mb-10">
            <h1 className="plate-display text-4xl md:text-6xl break-all">
              {result || (error ? "Error" : "No Signal")}
            </h1>
          </div>

          <div className="grid grid-cols-2 gap-4 mb-auto">
            <div className="border border-white/10 p-5 rounded-lg">
              <div className="label-micro mb-1">FPS</div>
              <div className="text-2xl font-bold">{fps}Hz</div>
            </div>
            <div className="border border-white/10 p-5 rounded-lg text-right">
              <div className="label-micro mb-1">WASM</div>
              <div className="text-2xl font-bold text-accent">STABLE</div>
            </div>
          </div>

          {error && (
            <div className="my-4 p-3 bg-red-500/20 border border-red-500 rounded-lg flex items-center gap-2 text-red-400 text-xs">
              <ShieldAlert className="w-4 h-4" />
              {error}
            </div>
          )}

          <div className="flex gap-2 mt-6">
            <button
              onClick={scanning ? stopScan : startScan}
              className="flex-1 bg-accent text-black p-4 font-black uppercase tracking-widest text-xs rounded-lg hover:bg-accent/80 transition"
            >
              {scanning ? "Abort" : "Deploy"}
            </button>
            <button
              onClick={() => setResult("")}
              className="border border-white/20 p-4 font-black uppercase text-xs rounded-lg hover:bg-white/5 transition"
            >
              Clear
            </button>
          </div>
        </aside>
      </main>
    </div>
  );
}
