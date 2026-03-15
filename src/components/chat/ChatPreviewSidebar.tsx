import type { ReactNode, RefObject } from 'react';

type ChatPreviewSidebarProps = {
  show: boolean;
  studyContent?: ReactNode;
  materialsContent?: ReactNode;
  isLocalTrackerRunning: boolean;
  cameraPreviewEnabled: boolean;
  cameraPreviewAfterCalibration: boolean;
  previewError: string | null;
  previewVideoRef: RefObject<HTMLVideoElement>;
};

export default function ChatPreviewSidebar({
  show,
  studyContent,
  materialsContent,
  isLocalTrackerRunning,
  cameraPreviewEnabled,
  cameraPreviewAfterCalibration,
  previewError,
  previewVideoRef,
}: ChatPreviewSidebarProps) {
  if (!show) return null;

  const renderCameraBody = () => {
    if (!cameraPreviewEnabled) {
      return (
        <div className="preview-sidebar-empty">
          <p>Camera preview is off. Enable it from Settings when you want live video here.</p>
        </div>
      );
    }

    if (isLocalTrackerRunning) {
      if (previewError) {
        return (
          <div className="preview-sidebar-empty">
            <p>Camera preview unavailable: {previewError}</p>
          </div>
        );
      }
      return <video ref={previewVideoRef} className="preview-video" playsInline muted />;
    }

    if (!cameraPreviewAfterCalibration) {
      return (
        <div className="preview-sidebar-empty">
          <p>
            Start Focus to calibrate your webcam. Once calibration is complete, the live preview will appear here.
          </p>
        </div>
      );
    }

    if (previewError) {
      return (
        <div className="preview-sidebar-empty">
          <p>Camera preview unavailable: {previewError}</p>
        </div>
      );
    }

    return <video ref={previewVideoRef} className="preview-video" playsInline muted />;
  };

  return (
    <div className="preview-sidebar" aria-label="Right side panel">
      <div className="preview-sidebar-header">
        <h3>Study Panel</h3>
      </div>

      <div className="preview-sidebar-body">
        <div className="preview-sidebar-section">{studyContent}</div>
        {materialsContent && (
          <div className="preview-sidebar-section">
            <h4 className="preview-sidebar-subtitle">Course Materials</h4>
            {materialsContent}
          </div>
        )}
        <div className="preview-sidebar-section">
          <h4 className="preview-sidebar-subtitle">Camera Preview</h4>
          {renderCameraBody()}
        </div>
      </div>
    </div>
  );
}
