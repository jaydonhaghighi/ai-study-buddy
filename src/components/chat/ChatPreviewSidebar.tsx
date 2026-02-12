import type { RefObject } from 'react';

type ChatPreviewSidebarProps = {
  show: boolean;
  isLocalTrackerRunning: boolean;
  cameraPreviewAfterCalibration: boolean;
  previewError: string | null;
  previewVideoRef: RefObject<HTMLVideoElement>;
};

export default function ChatPreviewSidebar({
  show,
  isLocalTrackerRunning,
  cameraPreviewAfterCalibration,
  previewError,
  previewVideoRef,
}: ChatPreviewSidebarProps) {
  if (!show) return null;

  return (
    <div className="preview-sidebar" aria-label="Camera preview sidebar">
      <div className="preview-sidebar-header">
        <h3>Camera</h3>
      </div>

      <div className="preview-sidebar-body">
        {isLocalTrackerRunning ? (
          previewError ? (
            <div className="preview-sidebar-empty">
              <p>Camera preview unavailable: {previewError}</p>
            </div>
          ) : (
            <video ref={previewVideoRef} className="preview-video" playsInline muted />
          )
        ) : !cameraPreviewAfterCalibration ? (
          <div className="preview-sidebar-empty">
            <p>
              Start Focus to calibrate your webcam. Once calibration is complete, the live preview will stay here while you study.
            </p>
          </div>
        ) : (
          previewError ? (
            <div className="preview-sidebar-empty">
              <p>Camera preview unavailable: {previewError}</p>
            </div>
          ) : (
            <video ref={previewVideoRef} className="preview-video" playsInline muted />
          )
        )}
      </div>
    </div>
  );
}
