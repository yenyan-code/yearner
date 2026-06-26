import { useEffect, useRef, useState } from 'react'
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision'

// 1. The list of memes we want to support.
// These match the files in your public/memes/ folder exactly!
const MEME_LIST = [
  { id: 'alden-recharge', name: 'Alden Recharge', filename: 'alden recharge.jpg' },
  { id: 'bleh', name: 'Bleh', filename: 'bleh.jpg' },
  { id: 'dunno-marvin', name: 'Dunno What to Say (Marvin)', filename: 'dunno what to say (marvin fojas).jpg' },
  { id: 'heart-queen-lengleng', name: 'Heart Queen Lengleng', filename: 'heart queen lengleng.jpg' },
  { id: 'holding-phone-ruffa', name: 'Holding Phone (Ruffa Mae)', filename: 'holding phone (ruffa mae).jpg' },
  { id: 'jose-rizal', name: 'Jose Rizal Meme', filename: 'jose rizal meme.jpg' },
  { id: 'kilig-manny', name: 'Kilig (Manny)', filename: 'kilig (manny).jpg' },
  { id: 'luh-mami-oni', name: 'Luh (Mami Oni)', filename: 'luh (mami oni).jpg' },
  { id: 'melai', name: 'Melai Meme', filename: 'melai meme.jpg' },
  { id: 'mind-reading-kris', name: 'Mind Reading (Kris)', filename: 'mind reading (kirs aquino).jpg' },
  { id: 'ok-naka-white', name: 'OK (Lalaking Naka White)', filename: 'ok (lalaking naka white).jpg' },
  { id: 'ples-yassi', name: 'Ples Ples Ples (Yassi)', filename: 'ples ples ples (yassi).jpg' },
  { id: 'shingkit-bayani', name: 'Shingkit (Bayani)', filename: 'shingkit (bayani agbayani).jpg' },
  { id: 'shocked-robin', name: 'Shocked (Robin)', filename: 'shocked (robin padilla).jpg' },
  { id: 'man-cant-be-moved', name: "The Man Who Can't Be Moved", filename: "the man who can't be moved.jpg" }
];

// Bones to draw on the screen (indices correspond to MediaPipe landmarks)
const SKELETON_CONNECTIONS = [
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // torso sides
  [23, 24]  // hips
];

const COMPARISON_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24];

export default function App() {
  // --- Refs ---
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const animationRef = useRef(null);
  const templatesRef = useRef({});

  // --- States ---
  const [modelLoading, setModelLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState("Initializing...");
  const [matchedMeme, setMatchedMeme] = useState(null);
  const [matchScore, setMatchScore] = useState(0);
  const [statusText, setStatusText] = useState("Align your pose to mirror a meme");

  // --- Load MediaPipe on Mount ---
  useEffect(() => {
    async function loadAIModel() {
      try {
        setLoadingProgress("Initializing AI Core...");
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
        );

        setLoadingProgress("Fetching Neural Weights (6MB)...");
        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numPoses: 1
        });

        landmarkerRef.current = landmarker;
        loadMemeTemplates(landmarker);
      } catch (error) {
        console.error(error);
        setLoadingProgress("Initialization error. Check connection.");
      }
    }
    loadAIModel();

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, []);

  // --- Auto-Calibration from Images ---
  async function loadMemeTemplates(landmarker) {
    setLoadingProgress("Calibrating Target Vectors...");
    let calibrationIndex = 1;

    for (const meme of MEME_LIST) {
      const img = new Image();
      img.src = `/memes/${meme.filename}`;
      img.crossOrigin = "anonymous";

      img.onload = () => {
        try {
          const tempCanvas = document.createElement('canvas');
          const tempCtx = tempCanvas.getContext('2d');
          tempCanvas.width = img.width;
          tempCanvas.height = img.height;
          tempCtx.drawImage(img, 0, 0);

          // Use a strictly increasing small index (1, 2, 3...) to avoid timestamp clashes
          const timestamp = calibrationIndex++;
          const result = landmarker.detectForVideo(tempCanvas, timestamp);
          if (result && result.landmarks && result.landmarks[0]) {
            const normalized = normalizePose(result.landmarks[0]);
            if (normalized) {
              templatesRef.current[meme.id] = normalized;
              console.log(`Calibrated meme template: ${meme.name}`);
            }
          }
        } catch (err) {
          console.warn(`Error calibrating ${meme.name}:`, err);
        }
      };
    }

    setTimeout(() => {
      startWebcam();
    }, 1500);
  }

  // --- Webcam Access ---
  async function startWebcam() {
    try {
      setLoadingProgress("Powering Camera Module...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setModelLoading(false);
          startMatchingLoop();
        };
      }
    } catch (err) {
      console.error(err);
      setLoadingProgress("Camera blocked. Grant permissions and reload.");
    }
  }

  // --- Main Detection Loop ---
  function startMatchingLoop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    function renderFrame() {
      if (!video || video.paused || video.ended || !landmarkerRef.current) {
        animationRef.current = requestAnimationFrame(renderFrame);
        return;
      }

      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      try {
        const result = landmarkerRef.current.detectForVideo(video, performance.now());

        if (result && result.landmarks && result.landmarks[0]) {
          const userPoints = result.landmarks[0];
          drawUserSkeleton(ctx, userPoints);
          comparePoseToTemplates(userPoints);
        } else {
          setMatchedMeme(null);
          setMatchScore(0);
          setStatusText("Step back into camera view");
        }
      } catch (err) {
        console.error("Inference Error:", err);
      }

      animationRef.current = requestAnimationFrame(renderFrame);
    }

    animationRef.current = requestAnimationFrame(renderFrame);
  }

  // --- Comparison & Scoring ---
  function comparePoseToTemplates(userPoints) {
    const userNorm = normalizePose(userPoints);
    if (!userNorm) return;

    let bestMeme = null;
    let highestScore = 0;

    for (const meme of MEME_LIST) {
      const targetNorm = templatesRef.current[meme.id];
      if (!targetNorm) continue;

      const score = calculateSimilarity(userNorm, targetNorm);
      if (score > highestScore) {
        highestScore = score;
        bestMeme = meme;
      }
    }

    if (highestScore >= 80 && bestMeme) {
      setMatchedMeme(bestMeme);
      setMatchScore(highestScore);
      setStatusText(`NAILED IT! - ${bestMeme.name}`);
    } else {
      setMatchedMeme(null);
      setMatchScore(highestScore);
      setStatusText(highestScore > 50 ? `Matching... (${highestScore}%)` : "Mirror a meme pose");
    }
  }

  // --- Normalization math ---
  function normalizePose(landmarks) {
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];

    if (!leftShoulder || !rightShoulder) return null;

    const originX = (leftShoulder.x + rightShoulder.x) / 2;
    const originY = (leftShoulder.y + rightShoulder.y) / 2;
    const originZ = (leftShoulder.z + rightShoulder.z) / 2;

    const translated = landmarks.map(point => ({
      x: point.x - originX,
      y: point.y - originY,
      z: point.z - originZ
    }));

    const dx = leftShoulder.x - rightShoulder.x;
    const dy = leftShoulder.y - rightShoulder.y;
    const dz = leftShoulder.z - rightShoulder.z;
    const sizeFactor = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1.0;

    return translated.map(point => ({
      x: point.x / sizeFactor,
      y: point.y / sizeFactor,
      z: point.z / sizeFactor
    }));
  }

  function calculateSimilarity(userNorm, targetNorm) {
    let totalDist = 0;
    let validJoints = 0;

    for (const index of COMPARISON_JOINTS) {
      const u = userNorm[index];
      const t = targetNorm[index];

      if (u && t) {
        const dx = u.x - t.x;
        const dy = u.y - t.y;
        const dz = u.z - t.z;
        totalDist += Math.sqrt(dx*dx + dy*dy + dz*dz);
        validJoints++;
      }
    }

    if (validJoints === 0) return 0;
    const avgDist = totalDist / validJoints;
    const percentage = Math.max(0, 100 - (avgDist * 160));
    return Math.round(percentage);
  }

  // --- Draw overlay (using colors from the design system) ---
  function drawUserSkeleton(ctx, landmarks) {
    // Connect points with Primary Red lines
    ctx.strokeStyle = '#E53935'; 
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';

    for (const [startIdx, endIdx] of SKELETON_CONNECTIONS) {
      const start = landmarks[startIdx];
      const end = landmarks[endIdx];

      if (start && end && start.visibility > 0.5 && end.visibility > 0.5) {
        ctx.beginPath();
        ctx.moveTo(start.x * ctx.canvas.width, start.y * ctx.canvas.height);
        ctx.lineTo(end.x * ctx.canvas.width, end.y * ctx.canvas.height);
        ctx.stroke();
      }
    }

    // Draw joints with Secondary Yellow dots
    for (const idx of COMPARISON_JOINTS) {
      const point = landmarks[idx];
      if (point && point.visibility > 0.5) {
        ctx.beginPath();
        ctx.arc(point.x * ctx.canvas.width, point.y * ctx.canvas.height, 6, 0, 2 * Math.PI);
        ctx.fillStyle = '#FBC02D'; 
        ctx.fill();
      }
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 md:p-12">
      
      {/* Design System Header */}
      <div className="text-center mb-8">
        <div className="flex items-center justify-center gap-2 mb-2">
          <span className="text-3xl">🎨</span>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight font-['Bricolage_Grotesque'] text-[#E53935]">
            MemeMirror
          </h1>
        </div>
        <p className="text-sm font-medium tracking-wide uppercase text-slate-400 font-['Plus_Jakarta_Sans']">
          {statusText}
        </p>
      </div>

      {/* Bento Grid layout matching screenshot aesthetics */}
      <div className="flex flex-col lg:flex-row gap-8 items-center justify-center w-full max-w-5xl">
        
        {/* Bento Box 1: Camera Feed */}
        <div className="w-full lg:w-[600px] aspect-[4/3] bg-[#EAEAEA] border border-white/10 rounded-[28px] p-6 shadow-2xl relative overflow-hidden flex flex-col justify-between">
          <div className="flex justify-between items-center mb-3">
            <span className="text-xs font-bold uppercase tracking-wider text-[#222222] font-['Plus_Jakarta_Sans']">
              Live Feed
            </span>
            <div className="h-2.5 w-2.5 rounded-full bg-[#E53935] animate-pulse"></div>
          </div>

          <div className="relative flex-grow bg-[#111111] rounded-[18px] overflow-hidden border border-[#222222]">
            {modelLoading ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-center p-6 bg-[#111111]">
                <div className="animate-spin-slow rounded-full h-10 w-10 border-2 border-slate-700 border-t-[#E53935]"></div>
                <span className="text-sm text-slate-400 font-['Plus_Jakarta_Sans']">{loadingProgress}</span>
              </div>
            ) : (
              <>
                <video
                  ref={videoRef}
                  className="w-full h-full object-cover scale-x-[-1]"
                  playsInline
                  muted
                />
                <canvas
                  ref={canvasRef}
                  className="absolute inset-0 w-full h-full object-cover scale-x-[-1]"
                />
              </>
            )}
          </div>
        </div>

        {/* Bento Box 2: Reveal Panel */}
        <div className="w-full sm:w-[320px] aspect-[4/3] lg:aspect-[1/1] bg-[#EAEAEA] border border-white/10 rounded-[28px] p-6 shadow-2xl flex flex-col justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-[#222222] font-['Plus_Jakarta_Sans']">
            Mirror Match
          </span>

          <div className="relative flex-grow bg-[#111111] rounded-[18px] overflow-hidden border border-[#222222] flex flex-col items-center justify-center my-3 p-4">
            {matchedMeme ? (
              <div className="w-full h-full flex flex-col items-center justify-center animate-fade-in">
                <img
                  src={`/memes/${matchedMeme.filename}`}
                  alt={matchedMeme.name}
                  className="w-full h-[75%] object-cover rounded-[12px] border-2 border-[#10b981]"
                  onError={(e) => {
                    e.target.style.display = 'none';
                  }}
                />
                <div className="mt-3 text-center">
                  <h3 className="text-sm font-bold text-white font-['Bricolage_Grotesque']">
                    {matchedMeme.name}
                  </h3>
                  <span className="text-xs font-bold text-[#10b981] font-mono">
                    Match: {matchScore}%
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-center flex flex-col items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-[#222222] flex items-center justify-center border border-white/5">
                  <span className="text-xl">📷</span>
                </div>
                <div className="space-y-1">
                  <h3 className="text-sm font-bold text-white font-['Bricolage_Grotesque']">
                    Awaiting Match
                  </h3>
                  <p className="text-xs text-slate-500 max-w-[180px] leading-relaxed font-['Plus_Jakarta_Sans']">
                    Pose like one of your {MEME_LIST.length} templates.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

      </div>

    </div>
  )
}
