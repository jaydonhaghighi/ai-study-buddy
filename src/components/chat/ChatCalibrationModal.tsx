import WebcamCalibrationPreview from '../WebcamCalibrationPreview';

type ChatCalibrationModalProps = {
  show: boolean;
  onCancel: () => void;
  onAlignedStable: () => void;
};

export default function ChatCalibrationModal({
  show,
  onCancel,
  onAlignedStable,
}: ChatCalibrationModalProps) {
  if (!show) return null;

  return (
    <div
      className="modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Camera calibration"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) {
          onCancel();
        }
      }}
    >
      <div className="modal modal--wide">
        <div className="modal-header">
          <div>
            <h3 className="modal-title">Camera calibration</h3>
            <p className="modal-subtitle">Align your laptop webcam before focus tracking starts.</p>
          </div>
          <button
            className="modal-close"
            onClick={onCancel}
            aria-label="Close"
            type="button"
          >
            ×
          </button>
        </div>

        <div className="modal-body">
          <WebcamCalibrationPreview
            variant="embedded"
            autoStart
            mode="calibration"
            autoStopAfterAlignedSeconds={3}
            stopOnAlignedStable={false}
            onRequestClose={onCancel}
            onAlignedStable={onAlignedStable}
          />
        </div>
      </div>
    </div>
  );
}
