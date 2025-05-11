import React, { useEffect, useRef, useState } from 'react';
import * as faceapi from 'face-api.js';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [lipColor, setLipColor] = useState('#ff0000');
  const [isModelLoading, setIsModelLoading] = useState(true);
  const imageRef = useRef();
  const canvasRef = useRef();

  useEffect(() => {
    const loadModels = async () => {
      const MODEL_URL = process.env.NODE_ENV === 'development' 
        ? 'http://localhost:3000/models'
        : `${window.location.origin}/models`;

      try {
        console.log('Starting to load models from:', MODEL_URL);
        
        // 加載所有必要的模型
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
        ]);
        
        // 驗證模型是否正確加載
        if (!faceapi.nets.tinyFaceDetector.isLoaded) {
          throw new Error('Face detector model not loaded');
        }
        if (!faceapi.nets.faceLandmark68Net.isLoaded) {
          throw new Error('Landmark model not loaded');
        }
        if (!faceapi.nets.faceExpressionNet.isLoaded) {
          throw new Error('Expression model not loaded');
        }
        
        console.log('All models loaded successfully');
        setIsModelLoading(false);
      } catch (error) {
        console.error('Error loading models:', error);
        alert('Error loading models: ' + error.message);
      }
    };

    loadModels();
  }, []);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => setImage(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const analyzeImageColors = (ctx, x, y, width, height) => {
    const imageData = ctx.getImageData(x, y, width, height);
    const data = imageData.data;
    let r = 0, g = 0, b = 0, count = 0;
    
    for (let i = 0; i < data.length; i += 4) {
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
      count++;
    }
    
    return {
      r: Math.round(r / count),
      g: Math.round(g / count),
      b: Math.round(b / count)
    };
  };

  const createEnhancedLipMask = (ctx, points, width, height) => {
    const mask = new ImageData(width, height);
    const data = mask.data;
    
    // 分離上下嘴唇的點
    const upperLipPoints = points.slice(0, 7);
    const lowerLipPoints = points.slice(6, 12);
    
    // 創建更精確的嘴唇區域遮罩
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        let insideUpper = false;
        let insideLower = false;
        
        // 檢查上嘴唇
        for (let i = 0, j = upperLipPoints.length - 1; i < upperLipPoints.length; j = i++) {
          if (((upperLipPoints[i].y > y) !== (upperLipPoints[j].y > y)) &&
              (x < (upperLipPoints[j].x - upperLipPoints[i].x) * (y - upperLipPoints[i].y) / 
              (upperLipPoints[j].y - upperLipPoints[i].y) + upperLipPoints[i].x)) {
            insideUpper = !insideUpper;
          }
        }
        
        // 檢查下嘴唇
        for (let i = 0, j = lowerLipPoints.length - 1; i < lowerLipPoints.length; j = i++) {
          if (((lowerLipPoints[i].y > y) !== (lowerLipPoints[j].y > y)) &&
              (x < (lowerLipPoints[j].x - lowerLipPoints[i].x) * (y - lowerLipPoints[i].y) / 
              (lowerLipPoints[j].y - lowerLipPoints[i].y) + lowerLipPoints[i].x)) {
            insideLower = !insideLower;
          }
        }
        
        if (insideUpper || insideLower) {
          // 計算到邊緣的距離來創建平滑過渡
          let minDist = Number.MAX_VALUE;
          const allPoints = [...upperLipPoints, ...lowerLipPoints];
          for (let i = 0; i < allPoints.length; i++) {
            const dist = Math.sqrt(
              Math.pow(x - allPoints[i].x, 2) + Math.pow(y - allPoints[i].y, 2)
            );
            minDist = Math.min(minDist, dist);
          }
          
          // 創建更平滑的 alpha 值
          const maxDist = 15; // 增加過渡區域
          const alpha = Math.pow(Math.max(0, 1 - minDist / maxDist), 1.5); // 使用冪函數創建更平滑的過渡
          data[idx + 3] = Math.round(alpha * 255);
        }
      }
    }
    
    return mask;
  };

  const applyMakeup = async () => {
    if (!image || !imageRef.current) return;

    const detections = await faceapi
      .detectSingleFace(imageRef.current, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions();

    if (!detections) {
      alert('No face detected!');
      return;
    }

    const canvas = canvasRef.current;
    const displaySize = { width: imageRef.current.width, height: imageRef.current.height };
    faceapi.matchDimensions(canvas, displaySize);

    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);

    // 分析嘴唇區域的原始顏色
    const landmarks = detections.landmarks;
    const mouthPoints = landmarks.getMouth();
    const allMouthPoints = [...mouthPoints];

    // 計算嘴唇區域的邊界框
    const lipBounds = {
      left: Math.min(...allMouthPoints.map(p => p.x)),
      right: Math.max(...allMouthPoints.map(p => p.x)),
      top: Math.min(...allMouthPoints.map(p => p.y)),
      bottom: Math.max(...allMouthPoints.map(p => p.y))
    };

    // 分析原始嘴唇顏色
    const originalLipColor = analyzeImageColors(
      ctx,
      Math.round(lipBounds.left),
      Math.round(lipBounds.top),
      Math.round(lipBounds.right - lipBounds.left),
      Math.round(lipBounds.bottom - lipBounds.top)
    );

    // 創建臨時畫布
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');

    // 創建嘴唇遮罩
    const lipMask = createEnhancedLipMask(tempCtx, allMouthPoints, canvas.width, canvas.height);

    // 將選擇的口紅顏色轉換為RGB
    const r = parseInt(lipColor.slice(1, 3), 16);
    const g = parseInt(lipColor.slice(3, 5), 16);
    const b = parseInt(lipColor.slice(5, 7), 16);

    // 獲取原始圖像數據
    const originalImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const finalImageData = new ImageData(new Uint8ClampedArray(originalImageData.data), canvas.width, canvas.height);

    // 應用口紅效果
    for (let i = 0; i < lipMask.data.length; i += 4) {
      if (lipMask.data[i + 3] > 0) {
        const alpha = lipMask.data[i + 3] / 255;
        
        // 獲取原始像素
        const origR = originalImageData.data[i];
        const origG = originalImageData.data[i + 1];
        const origB = originalImageData.data[i + 2];
        
        // 計算亮度
        const brightness = (origR + origG + origB) / (3 * 255);
        
        // 調整口紅顏色基於原始顏色的亮度
        const adjustedR = Math.min(255, r * (0.7 + brightness * 0.3));
        const adjustedG = Math.min(255, g * (0.7 + brightness * 0.3));
        const adjustedB = Math.min(255, b * (0.7 + brightness * 0.3));
        
        // 使用更複雜的混合算法
        const blendFactor = Math.pow(alpha, 1.2); // 調整混合曲線
        const lipstickOpacity = 0.9; // 增加口紅不透明度
        
        // 混合原始顏色和口紅顏色
        finalImageData.data[i] = Math.round(origR * (1 - blendFactor * lipstickOpacity) + adjustedR * blendFactor * lipstickOpacity);
        finalImageData.data[i + 1] = Math.round(origG * (1 - blendFactor * lipstickOpacity) + adjustedG * blendFactor * lipstickOpacity);
        finalImageData.data[i + 2] = Math.round(origB * (1 - blendFactor * lipstickOpacity) + adjustedB * blendFactor * lipstickOpacity);
        
        // 計算當前像素的 x 和 y 座標
        const x = Math.floor((i / 4) % canvas.width);
        const y = Math.floor((i / 4) / canvas.width);
        
        // 添加更自然的光澤效果
        const shimmerBase = Math.sin((x + y) / 8) * 0.5 + 0.5; // 使用座標創建更自然的光澤模式
        const shimmerIntensity = shimmerBase * 0.15 + 0.95; // 減少光澤變化範圍
        
        finalImageData.data[i] = Math.min(255, finalImageData.data[i] * shimmerIntensity);
        finalImageData.data[i + 1] = Math.min(255, finalImageData.data[i + 1] * shimmerIntensity);
        finalImageData.data[i + 2] = Math.min(255, finalImageData.data[i + 2] * shimmerIntensity);
        finalImageData.data[i + 1] = Math.min(255, finalImageData.data[i + 1] * shimmerIntensity);
        finalImageData.data[i + 2] = Math.min(255, finalImageData.data[i + 2] * shimmerIntensity);
      }
    }

    // 將最終結果繪製到畫布上
    ctx.putImageData(finalImageData, 0, 0);
  };

  return (
    <div className="App">
      <h1>虛擬試妝 - 口紅效果</h1>
      {isModelLoading ? (
        <p>載入模型中...</p>
      ) : (
        <div className="makeup-container">
          <div className="controls">
            <input type="file" accept="image/*" onChange={handleImageUpload} />
            <input type="color" value={lipColor} onChange={(e) => setLipColor(e.target.value)} />
            <button onClick={applyMakeup} disabled={!image}>套用口紅</button>
          </div>
          <div className="preview">
            {image && (
              <>
                <img
                  ref={imageRef}
                  src={image}
                  alt="上傳的照片"
                  onLoad={applyMakeup}
                  style={{ display: 'none' }}
                />
                <canvas ref={canvasRef} />
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
