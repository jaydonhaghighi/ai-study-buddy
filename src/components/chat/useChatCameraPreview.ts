import { useEffect, useRef, useState } from 'react';
import { acquireWebcamStream } from '../../services/webcam-manager';

type UseChatCameraPreviewParams = {
  mainView: 'chat' | 'dashboard';
  showCalibrationModal: boolean;
};

export function useChatCameraPreview({
  mainView,
  showCalibrationModal,
}: UseChatCameraPreviewParams) {
  const [cameraPreviewEnabled, setCameraPreviewEnabled] = useState(() => {
    return localStorage.getItem('cameraPreviewEnabled') !== '0';
  });
  const [cameraPreviewAfterCalibration, setCameraPreviewAfterCalibration] = useState(false);
  const previewVideoRef = useRef<HTMLVideoElement>(null);
  const previewReleaseRef = useRef<null | (() => void)>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  useEffect(() => {
    localStorage.setItem('cameraPreviewEnabled', cameraPreviewEnabled ? '1' : '0');
  }, [cameraPreviewEnabled]);

  useEffect(() => {
    const shouldShowPreview = cameraPreviewEnabled && cameraPreviewAfterCalibration && mainView === 'chat';
    let cancelled = false;

    const stopPreview = () => {
      if (previewReleaseRef.current) {
        previewReleaseRef.current();
        previewReleaseRef.current = null;
      }
      if (previewVideoRef.current) {
        previewVideoRef.current.pause();
        previewVideoRef.current.srcObject = null;
      }
    };

    if (!shouldShowPreview) {
      stopPreview();
      return;
    }

    void (async () => {
      try {
        setPreviewError(null);
        const { stream, release } = await acquireWebcamStream();
        if (cancelled) {
          release();
          return;
        }
        previewReleaseRef.current = release;
        const el = previewVideoRef.current;
        if (!el) return;
        el.srcObject = stream;
        await el.play();
      } catch (e: any) {
        setPreviewError(e?.message || 'Could not start camera preview');
      }
    })();

    return () => {
      cancelled = true;
      stopPreview();
    };
  }, [cameraPreviewEnabled, cameraPreviewAfterCalibration, mainView]);

  useEffect(() => {
    if (!showCalibrationModal) return;
    if (previewReleaseRef.current) {
      previewReleaseRef.current();
      previewReleaseRef.current = null;
    }
    if (previewVideoRef.current) {
      previewVideoRef.current.pause();
      previewVideoRef.current.srcObject = null;
    }
  }, [showCalibrationModal]);

  return {
    cameraPreviewEnabled,
    setCameraPreviewEnabled,
    cameraPreviewAfterCalibration,
    setCameraPreviewAfterCalibration,
    previewVideoRef,
    previewError,
  };
}
