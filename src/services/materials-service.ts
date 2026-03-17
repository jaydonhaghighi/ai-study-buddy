import {
  collection,
  doc,
  serverTimestamp,
  setDoc,
  updateDoc,
} from 'firebase/firestore';
import {
  ref,
  uploadBytes,
} from 'firebase/storage';
import { db, storage } from '../firebase-config';
import { fetchFunctionsEndpoint } from './functions-http';

const SUPPORTED_EXTENSIONS = new Set([
  'pdf',
  'docx',
  'xlsx',
  'xls',
  'pptx',
  'ppt',
  'txt',
  'png',
  'jpg',
  'jpeg',
  'webp',
]);

export const ACCEPTED_MATERIAL_FILE_TYPES =
  '.pdf,.docx,.xlsx,.xls,.pptx,.ppt,.txt,.png,.jpg,.jpeg,.webp';

export const MAX_MATERIAL_FILE_SIZE_BYTES = 25 * 1024 * 1024;

type SupportedFileType = 'pdf' | 'docx' | 'spreadsheet' | 'slides' | 'txt' | 'image' | '';

export type UploadAndIndexMaterialParams = {
  file: File;
  userId: string;
  courseId: string | null;
  sessionId: string | null;
  chatId: string;
};

export type DeleteMaterialParams = {
  materialId: string;
  userId: string;
};

function getExtension(fileName: string): string {
  const idx = fileName.lastIndexOf('.');
  if (idx < 0 || idx === fileName.length - 1) return '';
  return fileName.slice(idx + 1).toLowerCase();
}

function inferFileType(fileName: string, mimeType: string): SupportedFileType {
  const ext = getExtension(fileName);
  if (ext === 'pdf' || mimeType === 'application/pdf') return 'pdf';
  if (ext === 'docx' || mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') return 'docx';
  if (ext === 'xlsx' || ext === 'xls' || mimeType.includes('spreadsheet') || mimeType === 'application/vnd.ms-excel') return 'spreadsheet';
  if (ext === 'pptx' || ext === 'ppt' || mimeType.includes('presentation') || mimeType === 'application/vnd.ms-powerpoint') return 'slides';
  if (ext === 'txt' || mimeType === 'text/plain') return 'txt';
  if (['png', 'jpg', 'jpeg', 'webp'].includes(ext) || mimeType.startsWith('image/')) return 'image';
  return '';
}

function sanitizeFileName(fileName: string): string {
  return fileName.replace(/[^a-zA-Z0-9._-]/g, '_');
}

export function isSupportedMaterialFile(file: File): boolean {
  const ext = getExtension(file.name);
  return SUPPORTED_EXTENSIONS.has(ext);
}

export function formatMaterialSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
}

export async function uploadAndIndexMaterial({
  file,
  userId,
  courseId,
  sessionId,
  chatId,
}: UploadAndIndexMaterialParams): Promise<{ materialId: string }> {
  if (!isSupportedMaterialFile(file)) {
    throw new Error(`Unsupported file type: ${file.name}`);
  }
  if (file.size > MAX_MATERIAL_FILE_SIZE_BYTES) {
    throw new Error(`File too large: ${file.name} (max 25MB)`);
  }

  const materialRef = doc(collection(db, 'courseMaterials'));
  const extension = getExtension(file.name);
  const fileType = inferFileType(file.name, file.type || '');
  const createdAt = serverTimestamp();

  await setDoc(materialRef, {
    id: materialRef.id,
    userId,
    courseId,
    sessionId,
    chatId,
    fileName: file.name,
    extension,
    mimeType: file.type || 'application/octet-stream',
    fileType,
    sizeBytes: file.size,
    storagePath: '',
    status: 'uploading',
    chunkCount: 0,
    errorMessage: null,
    createdAt,
    updatedAt: createdAt,
  });

  const safeName = sanitizeFileName(file.name);
  const storagePath = `users/${userId}/courseMaterials/${materialRef.id}/${Date.now()}-${safeName}`;

  try {
    const storageRef = ref(storage, storagePath);
    await uploadBytes(storageRef, file, {
      contentType: file.type || 'application/octet-stream',
    });

    await updateDoc(materialRef, {
      storagePath,
      status: 'uploaded',
      errorMessage: null,
      updatedAt: serverTimestamp(),
    });

    await fetchFunctionsEndpoint('/materialIndex', {
      method: 'POST',
      body: JSON.stringify({ userId, materialId: materialRef.id }),
    });

    return { materialId: materialRef.id };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown upload error';
    await updateDoc(materialRef, {
      status: 'failed',
      errorMessage: message,
      updatedAt: serverTimestamp(),
    });
    throw error;
  }
}

export async function deleteMaterial({ materialId, userId }: DeleteMaterialParams): Promise<void> {
  await fetchFunctionsEndpoint('/materialDelete', {
    method: 'POST',
    body: JSON.stringify({ materialId, userId }),
  });
}
