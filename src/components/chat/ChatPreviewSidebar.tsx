import { useEffect, useState } from 'react';
import type { ReactNode, RefObject } from 'react';
import { BookOpenCheck, Camera, Files, PanelRightClose, Timer } from 'lucide-react';

type ChatPreviewSidebarProps = {
  show: boolean;
  studyContent?: ReactNode;
  materialsContent?: ReactNode;
  activeRecallContent?: ReactNode;
  isLocalTrackerRunning: boolean;
  cameraPreviewEnabled: boolean;
  cameraPreviewAfterCalibration: boolean;
  previewError: string | null;
  previewVideoRef: RefObject<HTMLVideoElement>;
  onClose: () => void;
};

export default function ChatPreviewSidebar({
  show,
  studyContent,
  materialsContent,
  activeRecallContent,
  isLocalTrackerRunning,
  cameraPreviewEnabled,
  cameraPreviewAfterCalibration,
  previewError,
  previewVideoRef,
  onClose,
}: ChatPreviewSidebarProps) {
  const [activeTabId, setActiveTabId] = useState<string>('study');

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

  const cameraContent = renderCameraBody();

  const tabs: Array<{ id: string; title: string; content: ReactNode; icon: ReactNode }> = [];
  if (studyContent) {
    tabs.push({
      id: 'study',
      title: 'Study Mode',
      icon: <Timer size={15} aria-hidden="true" />,
      content: (
        <>
          {studyContent}
          <div className="preview-sidebar-inline-divider" />
          <h5 className="preview-sidebar-inline-title">Camera Preview</h5>
          {cameraContent}
        </>
      ),
    });
  } else {
    tabs.push({ id: 'camera', title: 'Camera', icon: <Camera size={15} aria-hidden="true" />, content: cameraContent });
  }
  if (materialsContent) {
    tabs.push({ id: 'materials', title: 'Materials', icon: <Files size={15} aria-hidden="true" />, content: materialsContent });
  }
  if (activeRecallContent) {
    tabs.push({ id: 'recall', title: 'Active Recall', icon: <BookOpenCheck size={15} aria-hidden="true" />, content: activeRecallContent });
  }

  const firstTabId = tabs[0]?.id || 'study';
  const hasActiveTab = tabs.some((tab) => tab.id === activeTabId);

  useEffect(() => {
    if (!hasActiveTab) {
      setActiveTabId(firstTabId);
    }
  }, [firstTabId, hasActiveTab]);

  const activeTab = tabs.find((tab) => tab.id === activeTabId) ?? tabs[0] ?? null;

  if (!show) return null;

  return (
    <div className="preview-sidebar" aria-label="Right side panel">
      <div className="preview-sidebar-header">
        <h3>Study Panel</h3>
        <button
          type="button"
          className="preview-sidebar-close-btn"
          onClick={onClose}
          aria-label="Close right panel"
          title="Close right panel"
        >
          <PanelRightClose size={16} aria-hidden="true" />
        </button>
      </div>

      <div className="preview-sidebar-tabs" role="tablist" aria-label="Study panel sections">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={tab.id === activeTab?.id}
            className={`preview-sidebar-tab-btn ${tab.id === activeTab?.id ? 'active' : ''}`}
            onClick={() => setActiveTabId(tab.id)}
          >
            <span className="preview-sidebar-tab-icon" aria-hidden="true">{tab.icon}</span>
            <span>{tab.title}</span>
          </button>
        ))}
      </div>

      <div className="preview-sidebar-body">
        {tabs.map((tab) => (
          <div
            key={tab.id}
            role="tabpanel"
            aria-hidden={tab.id !== activeTab?.id}
            className={`preview-sidebar-section ${tab.id === activeTab?.id ? '' : 'preview-sidebar-section-hidden'}`}
          >
            <h4 className="preview-sidebar-subtitle">{tab.title}</h4>
            {tab.content}
          </div>
        ))}
      </div>
    </div>
  );
}
