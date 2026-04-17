import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as ort from 'onnxruntime-web';
import { Play, ShieldAlert } from 'lucide-react';
import { cn } from '../lib/utils';
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
  const processingRef = useRef(false);
  const lastDetectRef = useRef(0);
  const requestRef = useRef(0);

  useEffect(() => {
    offscreenCanvasRef.current = document.createElement("canvas");
    cropCanvasRef.current = document.createElement("canvas");
    ort.env.wasm.simd = false;
    ort.env.wasm.numThreads = 1;
    // Tải WASM từ CDN để chạy tốt trên GitHub Pages
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";
    
    async function init() {
      try {
        const p = await ort.InferenceSession.create("/model/bienso1.onnx", { executionProviders: ['wasm'] });
        const c = await ort.InferenceSession.create("/model/character.onnx", { executionProviders: ['wasm'] });
        setSessionPlate(p); setSessionChar(c); setModelReady(true);
      } catch (e) { console.error(e); }
    }
    init();
  }, []);

  const preprocess = (source: any) => {
    const canvas = offscreenCanvasRef.current!;
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    canvas.width = canvas.height = SIZE;
    ctx.drawImage(source, 0, 0, SIZE, SIZE);
    const imgData = ctx.getImageData(0, 0, SIZE, SIZE).data;
    for (let i = 0, p = 0; i < imgData.length; i += 4, p++) {
      inputBuffer[p] = imgData[i] / 255;
      inputBuffer[p + SIZE*SIZE] = imgData[i+1] / 255;
      inputBuffer[p + 2*SIZE*SIZE] = imgData[i+2] / 255;
    }
    return new ort.Tensor("float32", inputBuffer, [1, 3, SIZE, SIZE]);
  };

  const parseAndNMS = (output: any, dims: any, w: number, h: number) => {
    // Logic Parse YOLO & NMS (Rút gọn)
    return [{ x: 50, y: 50, w: 200, h: 100, confidence: 0.9 }]; 
  };

  const loop = useCallback(async () => {
    if (!scanning || processingRef.current || !videoRef.current || !sessionPlate) {
       requestRef.current = requestAnimationFrame(loop);
       return;
    }
    const now = Date.now();
    if (now - lastDetectRef.current < 900) { requestRef.current = requestAnimationFrame(loop); return; }
    
    processingRef.current = true;
    try {
      const tensor = preprocess(videoRef.current);
      const res = await sessionPlate.run({ [sessionPlate.inputNames[0]]: tensor });
      // Xanh hóa khung nhận diện...
      lastDetectRef.current = now;
      setFps(Math.round(1000/(Date.now()-now)));
    } finally {
      processingRef.current = false;
      requestRef.current = requestAnimationFrame(loop);
    }
  }, [scanning, sessionPlate]);

  useEffect(() => {
    if (scanning) requestRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(requestRef.current);
  }, [scanning, loop]);

  return (
    <div className="min-h-screen flex flex-col bg-bg text-ink">
       {/* (Mã JSX theme Bold Typography đã cung cấp ở lượt trước) */}
       {/* Đảm bảo copy toàn bộ phần return của Scanner.tsx ở bước trước vào đây */}
    </div>
  );
}
