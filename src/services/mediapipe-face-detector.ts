import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';

const VISION_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const FACE_MODEL_ASSET_URL =
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite';

let visionResolverPromise: Promise<Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>> | null = null;

async function getVisionResolver() {
  if (!visionResolverPromise) {
    visionResolverPromise = FilesetResolver.forVisionTasks(VISION_WASM_URL);
  }
  return visionResolverPromise;
}

export async function createFaceDetector(): Promise<FaceDetector> {
  const vision = await getVisionResolver();
  return FaceDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: FACE_MODEL_ASSET_URL,
    },
    runningMode: 'VIDEO',
  });
}
