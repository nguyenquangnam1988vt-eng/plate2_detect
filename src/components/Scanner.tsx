import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as ort from 'onnxruntime-web';
import { Camera, StopCircle, Play, ShieldAlert, Loader2 } from 'lucide-react';
import { cn } from '../lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

const SIZE = 320;
const CHAR_MAP = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","K","L","M","N","P","R","S","T","U","V","X","Y","Z"];
const SHEET_URL = "YOUR_GOOGLE_SHEET_URL";

// CHỐNG CRASH IOS: Sử dụng lại bộ nhớ đệm
let inputBuffer = new Float32Array(3 * SIZE * SIZE);

export default function Scanner() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [sessionPlate, setSessionPlate] = useState<any>(null);
  const [sessionChar, setSessionChar] = useState<any>(null);
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState("");
  const [fps, setFps] = useState(0);
  const processingRef = useRef(false);
  const lastDetectRef = useRef(0);

  useEffect(() => {
    // Cấu hình tối ưu cho Safari
    ort.env.wasm.simd = false;
    ort.env.wasm.numThreads = 1;

    async function loadModels() {
      try {
        const s1 = await ort.InferenceSession.create('./model/bienso1.onnx', { executionProviders: ['wasm'] });
        const s2 = await ort.InferenceSession.create('./model/character.onnx', { executionProviders: ['wasm'] });
        setSessionPlate(s1);
        setSessionChar(s2);
      } catch (e) {
        console.error("Model load error", e);
      }
    }
    loadModels();
  }, []);

  const preprocess = (source: HTMLVideoElement | HTMLCanvasElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = SIZE; canvas.height = SIZE;
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(source, 0, 0, SIZE, SIZE);
    const data = ctx.getImageData(0, 0, SIZE, SIZE).data;
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      inputBuffer[p] = data[i] / 255;
      inputBuffer[p + SIZE*SIZE] = data[i+1] / 255;
      inputBuffer[p + 2*SIZE*SIZE] = data[i+2] / 255;
    }
    return new ort.Tensor("float32", inputBuffer, [1, 3, SIZE, SIZE]);
  };

  const loop = useCallback(async () => {
    if (!scanning || processingRef.current || !videoRef.current || !sessionPlate) return;
    const now = Date.now();
    if (now - lastDetectRef.current < 800) { // Quét mỗi 800ms để tránh nóng máy/văng
      requestAnimationFrame(loop);
      return;
    }
    
    processingRef.current = true;
    try {
      const tensor = preprocess(videoRef.current);
      const output = await sessionPlate.run({ [sessionPlate.inputNames[0]]: tensor });
      // ... Logic xử lý Box & OCR tương tự như đã tối ưu ở các bước trước ...
      // Ở đây tôi lược bớt để tập trung vào tính tương thích hệ thống
      setFps(Math.round(1000 / (Date.now() - now)));
      lastDetectRef.current = now;
    } finally {
      processingRef.current = false;
      requestAnimationFrame(loop);
    }
  }, [scanning, sessionPlate]);

  // Kết xuất giao diện Bold Typography (đã cung cấp ở lượt trước)
  return (
    <div className="min-h-screen flex flex-col bg-[#050505]">
       {/* (Sử dụng mã JSX Bold Typography tôi đã viết ở lượt trước) */}
    </div>
  );
}
