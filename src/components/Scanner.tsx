import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as ort from 'onnxruntime-web';
import { Play, ShieldAlert } from 'lucide-react';
import { cn } from '@/src/lib/utils';
import { motion, AnimatePresence } from 'motion/react';

const SIZE = 320;
const CHAR_MAP = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","K","L","M","N","P","R","S","T","U","V","X","Y","Z"];
const SHEET_URL = "URL_APPS_SCRIPT_CUA_BAN";
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
  const requestRef = useRef(0);

  useEffect(() => {
    offscreenCanvasRef.current = document.createElement("canvas");
    cropCanvasRef.current = document.createElement("canvas");
    ort.env.wasm.simd = false;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";

    async function init() {
      try {
        const p = await ort.InferenceSession.create("/model/bienso1.onnx", { executionProviders: ['wasm'] });
        const c = await ort.InferenceSession.create("/model/character.onnx", { executionProviders: ['wasm'] });
        setSessionPlate(p); setSessionChar(c); setModelReady(true);
      } catch (e) { setError("Lỗi tải tệp ONNX trong /public/model/"); }
    }
    init();
    return () => cancelAnimationFrame(requestRef.current);
  }, []);

  const preprocess = (source: any) => {
    const canvas = offscreenCanvasRef.current!;
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    canvas.width = canvas.height = SIZE;
    const w = (source as any).videoWidth || source.width;
    const h = (source as any).videoHeight || source.height;
    const ratio = Math.min(SIZE/w, SIZE/h);
    const nW = w*ratio; const nH = h*ratio;
    ctx.fillStyle = "black"; ctx.fillRect(0,0,SIZE,SIZE);
    ctx.drawImage(source, (SIZE-nW)/2, (SIZE-nH)/2, nW, nH);
    const imgData = ctx.getImageData(0,0,SIZE,SIZE).data;
    for (let i=0, p=0; i<imgData.length; i+=4, p++) {
      inputBuffer[p] = imgData[i]/255;
      inputBuffer[p+SIZE*SIZE] = imgData[i+1]/255;
      inputBuffer[p+2*SIZE*SIZE] = imgData[i+2]/255;
    }
    return new ort.Tensor("float32", inputBuffer, [1, 3, SIZE, SIZE]);
  };

  const nms = (boxes: any[]) => {
    if (!boxes.length) return [];
    boxes.sort((a, b) => b.confidence - a.confidence);
    const result: any[] = [];
    let remaining = [...boxes];
    while (remaining.length) {
      const best = remaining.shift()!;
      result.push(best);
      remaining = remaining.filter(b => {
        const x1 = Math.max(best.x, b.x); const y1 = Math.max(best.y, b.y);
        const x2 = Math.min(best.x + best.w, b.x + b.w); const y2 = Math.min(best.y + best.h, b.y + b.h);
        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const union = best.w * best.h + b.w * b.h - inter;
        return union > 0 && (inter / union) < 0.45;
      });
    }
    return result;
  };

  const parseOutput = (output: any, dims: any, imgW: number, imgH: number) => {
    const boxes = [];
    const numBoxes = dims[2]; const numAttrs = dims[1];
    for (let i = 0; i < numBoxes; i++) {
        const cx = output[i]; const cy = output[i+numBoxes];
        const w = output[i+2*numBoxes]; const h = output[i+3*numBoxes];
        const obj = output[i+4*numBoxes];
        let maxS = 0; let bestC = 0;
        for (let c=5; c<numAttrs; c++) {
          const s = output[i+c*numBoxes];
          if (s > maxS) { maxS = s; bestC = c-5; }
        }
        const conf = obj * maxS;
        if (conf < 0.5) continue;
        const ratio = Math.min(SIZE/imgW, SIZE/imgH);
        const dx = (SIZE-imgW*ratio)/2; const dy = (SIZE-imgH*ratio)/2;
        boxes.push({ x: (cx-w/2-dx)/ratio, y: (cy-h/2-dy)/ratio, w: w/ratio, h: h/ratio, confidence: conf });
    }
    return boxes;
  };

  const loop = useCallback(async () => {
    if (!scanning || processingRef.current || !videoRef.current || !sessionPlate || !sessionChar) {
       requestRef.current = requestAnimationFrame(loop);
       return;
    }
    const now = Date.now();
    if (now - lastDetectRef.current < 900) { requestRef.current = requestAnimationFrame(loop); return; }
    
    processingRef.current = true;
    try {
      const video = videoRef.current;
      const tensor = preprocess(video);
      const res = await sessionPlate.run({ [sessionPlate.inputNames[0]]: tensor });
      let boxes = parseOutput(res[sessionPlate.outputNames[0]].data, res[sessionPlate.outputNames[0]].dims, video.videoWidth, video.videoHeight);
      boxes = nms(boxes);

      if (boxes.length && canvasRef.current) {
        const best = boxes[0];
        const ctx = canvasRef.current.getContext("2d")!;
        canvasRef.current.width = video.videoWidth; canvasRef.current.height = video.videoHeight;
        ctx.clearRect(0,0, canvasRef.current.width, canvasRef.current.height);
        ctx.strokeStyle = "#00FF9C"; ctx.lineWidth = 6;
        ctx.strokeRect(best.x, best.y, best.w, best.h);

        const cropCanvas = cropCanvasRef.current!;
        const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true })!;
        cropCanvas.width = best.w; cropCanvas.height = best.h;
        cropCtx.drawImage(video, best.x, best.y, best.w, best.h, 0, 0, best.w, best.h);
        const charTensor = preprocess(cropCanvas);
        const charRes = await sessionChar.run({ [sessionChar.inputNames[0]]: charTensor });
        const charOutput = charRes[charRes.outputNames[0]].data as Float32Array;
        let text = "";
        for (let i=0; i<charOutput.length && i<CHAR_MAP.length; i++) {
          if (charOutput[i] > 0.5) text += CHAR_MAP[i];
        }
        if (text.length >= 3) {
          const formatted = text.slice(0, 3) + "-" + text.slice(3, 8);
          setResult(formatted);
          fetch(SHEET_URL, { method: "POST", mode: 'no-cors', body: JSON.stringify({ plate: formatted }) });
        }
      }
      setFps(Math.round(1000/(Date.now()-lastDetectRef.current)));
      lastDetectRef.current = now;
    } catch (e) { console.error(e); }
    processingRef.current = false;
    requestRef.current = requestAnimationFrame(loop);
  }, [scanning, sessionPlate, sessionChar]);

  useEffect(() => {
    if (scanning) requestRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(requestRef.current);
  }, [scanning, loop]);

  const startScan = async () => {
    if (!modelReady) return;
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    if (videoRef.current) videoRef.current.srcObject = stream;
    setScanning(true);
  };

  const stopScan = () => {
    setScanning(true);
    if (videoRef.current?.srcObject) (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
    setScanning(false);
  };

  return (
    <div className="min-h-screen flex flex-col bg-bg text-ink overflow-hidden selection:bg-accent selection:text-black">
      {/* (Phần JSX "Bold Typography" như đã cung cấp ở lượt trước) */}
      <header className="h-20 border-b border-white/10 flex items-center justify-between px-10">
        <div className="text-2xl font-black tracking-tighter uppercase">ALPR.SCAN.01</div>
        <div className="flex items-center gap-2 text-accent text-xs font-bold uppercase tracking-widest">
           <div className="w-2 h-2 bg-accent rounded-full animate-pulse shadow-[0_0_10px_#00FF9C]" />
           System Live
        </div>
      </header>
      <main className="flex-1 scanner-grid overflow-hidden">
        <section className="p-10 flex flex-col gap-6 bg-radial-gradient">
           <div className="label-micro">Lens Capture</div>
           <div className="flex-1 relative bg-black border border-white/10 overflow-hidden">
              <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover opacity-50" muted playsInline />
              <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover z-10" />
              <AnimatePresence>
                {!scanning && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-20 backdrop-blur-sm">
                    <button onClick={startScan} disabled={!modelReady} className="flex flex-col items-center gap-4 text-accent transition-transform hover:scale-105 active:scale-95">
                       <div className="w-24 h-24 border-2 border-accent flex items-center justify-center relative">
                          <Play className="w-10 h-10 ml-1" />
                          <div className="absolute inset-0 border border-accent animate-pulse" />
                       </div>
                       <span className="label-micro">Start Neural Engine</span>
                    </button>
                  </div>
                )}
              </AnimatePresence>
           </div>
        </section>
        <aside className="bg-surface p-10 flex flex-col border-l border-white/10">
           <div className="label-micro mb-4">Identification</div>
           <div className="min-h-[160px] flex items-center mb-10">
             <h1 className="plate-display">{result || "No Signal"}</h1>
           </div>
           <div className="grid grid-cols-2 gap-4 mb-auto">
              <div className="border border-white/10 p-5">
                 <div className="label-micro mb-1">FPS</div>
                 <div className="text-2xl font-bold">{fps}Hz</div>
              </div>
              <div className="border border-white/10 p-5 text-right">
                 <div className="label-micro mb-1">WASM</div>
                 <div className="text-2xl font-bold text-accent">STABLE</div>
              </div>
           </div>
           <div className="flex gap-2">
             <button onClick={scanning ? stopScan : startScan} className="flex-1 bg-accent text-black p-5 font-black uppercase tracking-widest text-xs">{scanning ? "Abort" : "Deploy"}</button>
             <button onClick={() => setResult("")} className="border border-white/20 p-5 font-black uppercase text-xs">Clear</button>
           </div>
        </aside>
      </main>
    </div>
  );
}
