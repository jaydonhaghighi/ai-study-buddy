export interface DataItem {
  id: string;
  text: string;
  userId: string;
  createdAt: Date | null;
}

export type Citation = {
  id: string;
  materialId: string;
  fileName: string;
  fileType: 'pdf' | 'docx' | 'spreadsheet' | 'slides' | 'txt' | 'image';
  locationType: 'page' | 'sheet' | 'slide' | 'line' | 'image';
  locationLabel: string;
  snippet: string;
  score?: number;
};

export type CourseMaterialStatus = 'uploading' | 'uploaded' | 'processing' | 'indexed' | 'failed';

export type CourseMaterial = {
  id: string;
  userId: string;
  courseId: string | null;
  sessionId: string | null;
  chatId: string | null;
  fileName: string;
  extension: string;
  mimeType: string;
  sizeBytes: number;
  storagePath: string;
  fileType: 'pdf' | 'docx' | 'spreadsheet' | 'slides' | 'txt' | 'image' | '';
  status: CourseMaterialStatus;
  chunkCount?: number;
  errorMessage?: string | null;
  processingMs?: number;
  createdAt: Date | null;
  updatedAt: Date | null;
};
