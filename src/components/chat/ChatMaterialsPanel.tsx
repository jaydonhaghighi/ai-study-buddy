import { useRef, useState } from 'react';
import {
  FileImage,
  FileQuestion,
  FileSpreadsheet,
  FileText,
  Presentation,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import type { CourseMaterial } from '../../types';
import {
  ACCEPTED_MATERIAL_FILE_TYPES,
  formatMaterialSize,
} from '../../services/materials-service';

type ChatMaterialsPanelProps = {
  selectedChatId: string | null;
  materials: CourseMaterial[];
  materialsUploading: boolean;
  onUploadFiles: (files: FileList | File[] | null) => void;
  onDeleteMaterial: (materialId: string) => void;
};

function statusLabel(material: CourseMaterial): string {
  switch (material.status) {
    case 'uploading':
      return 'Uploading';
    case 'uploaded':
      return 'Uploaded';
    case 'processing':
      return 'Indexing';
    case 'indexed':
      return typeof material.chunkCount === 'number' ? `Indexed (${material.chunkCount})` : 'Indexed';
    case 'failed':
      return 'Failed';
    default:
      return material.status;
  }
}

function fileIcon(material: CourseMaterial) {
  switch (material.fileType) {
    case 'pdf':
      return <FileText size={16} aria-hidden="true" />;
    case 'docx':
      return <FileText size={16} aria-hidden="true" />;
    case 'spreadsheet':
      return <FileSpreadsheet size={16} aria-hidden="true" />;
    case 'slides':
      return <Presentation size={16} aria-hidden="true" />;
    case 'txt':
      return <FileText size={16} aria-hidden="true" />;
    case 'image':
      return <FileImage size={16} aria-hidden="true" />;
    default:
      return <FileQuestion size={16} aria-hidden="true" />;
  }
}

export default function ChatMaterialsPanel({
  selectedChatId,
  materials,
  materialsUploading,
  onUploadFiles,
  onDeleteMaterial,
}: ChatMaterialsPanelProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  if (!selectedChatId) {
    return (
      <div className="preview-sidebar-empty">
        <p>Select a chat first, then upload study materials for grounded answers with citations.</p>
      </div>
    );
  }

  return (
    <div>
      <div
        className={`files-drop-zone ${isDragging ? 'dragging' : ''} ${materialsUploading ? 'uploading' : ''}`}
        onDragOver={(event) => {
          event.preventDefault();
          if (!materialsUploading) setIsDragging(true);
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          setIsDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          if (materialsUploading) return;
          onUploadFiles(event.dataTransfer.files);
        }}
        onClick={() => {
          if (!materialsUploading) inputRef.current?.click();
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_MATERIAL_FILE_TYPES}
          multiple
          style={{ display: 'none' }}
          onChange={(event) => {
            onUploadFiles(event.target.files);
            event.currentTarget.value = '';
          }}
          disabled={materialsUploading}
        />
        <div className="drop-zone-content">
          <div className="drop-zone-icon" aria-hidden="true">
            <UploadCloud size={36} strokeWidth={1.7} />
          </div>
          <p className="drop-zone-text">
            Drop files here or click to upload
            <br />
            PDF, DOCX, Excel, Slides, TXT, Images (max 25MB each)
          </p>
          <button
            type="button"
            className="drop-zone-button"
            disabled={materialsUploading}
            onClick={(event) => {
              event.stopPropagation();
              inputRef.current?.click();
            }}
          >
            {materialsUploading ? 'Uploading...' : 'Choose Files'}
          </button>
        </div>
      </div>

      <div className="files-list">
        {materials.length === 0 ? (
          <div className="files-empty">No materials uploaded for this chat yet.</div>
        ) : (
          materials.map((material) => (
            <div key={material.id} className="file-item">
              <div className="file-info">
                <div className="file-icon">{fileIcon(material)}</div>
                <div className="file-details">
                  <div className="file-name">{material.fileName}</div>
                  <div className="file-size">
                    {formatMaterialSize(material.sizeBytes)} • {statusLabel(material)}
                  </div>
                  {material.status === 'failed' && material.errorMessage && (
                    <div className="material-error">{material.errorMessage}</div>
                  )}
                </div>
              </div>

              <div className="file-actions">
                <button
                  type="button"
                  className="file-action-btn"
                  title="Delete material"
                  onClick={() => onDeleteMaterial(material.id)}
                  disabled={materialsUploading}
                >
                  <Trash2 size={14} aria-hidden="true" />
                  <span>Delete</span>
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
