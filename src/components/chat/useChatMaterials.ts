import { useCallback, useEffect, useState } from 'react';
import type { User } from 'firebase/auth';
import {
  collection,
  onSnapshot,
  query,
  Timestamp,
  where,
} from 'firebase/firestore';
import { db } from '../../firebase-config';
import type { CourseMaterial } from '../../types';
import {
  deleteMaterial,
  isSupportedMaterialFile,
  uploadAndIndexMaterial,
} from '../../services/materials-service';

type ToastVariant = 'success' | 'warning' | 'info';

type CurrentChatLike = {
  id: string;
  courseId: string;
  sessionId: string;
};

type UseChatMaterialsParams = {
  user: User | null;
  selectedChatId: string | null;
  currentChat: CurrentChatLike | null;
  showToast: (message: string, variant?: ToastVariant) => void;
};

function normalizeStatus(input: unknown): CourseMaterial['status'] {
  if (input === 'uploading' || input === 'uploaded' || input === 'processing' || input === 'indexed' || input === 'failed') {
    return input;
  }
  return 'uploaded';
}

export function useChatMaterials({
  user,
  selectedChatId,
  currentChat,
  showToast,
}: UseChatMaterialsParams) {
  const [materials, setMaterials] = useState<CourseMaterial[]>([]);
  const [materialsUploading, setMaterialsUploading] = useState(false);

  useEffect(() => {
    if (!user || !selectedChatId) {
      setMaterials([]);
      return;
    }

    const q = query(collection(db, 'courseMaterials'), where('userId', '==', user.uid));

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const next: CourseMaterial[] = snapshot.docs
        .map((docSnap) => {
          const data = docSnap.data();
          return {
            id: docSnap.id,
            userId: data.userId,
            courseId: typeof data.courseId === 'string' ? data.courseId : null,
            sessionId: typeof data.sessionId === 'string' ? data.sessionId : null,
            chatId: typeof data.chatId === 'string' ? data.chatId : null,
            fileName: typeof data.fileName === 'string' ? data.fileName : 'Untitled',
            extension: typeof data.extension === 'string' ? data.extension : '',
            mimeType: typeof data.mimeType === 'string' ? data.mimeType : '',
            sizeBytes: typeof data.sizeBytes === 'number' ? data.sizeBytes : 0,
            storagePath: typeof data.storagePath === 'string' ? data.storagePath : '',
            fileType: typeof data.fileType === 'string' ? data.fileType : '',
            status: normalizeStatus(data.status),
            chunkCount: typeof data.chunkCount === 'number' ? data.chunkCount : undefined,
            errorMessage: typeof data.errorMessage === 'string' ? data.errorMessage : null,
            processingMs: typeof data.processingMs === 'number' ? data.processingMs : undefined,
            createdAt: data.createdAt instanceof Timestamp ? data.createdAt.toDate() : null,
            updatedAt: data.updatedAt instanceof Timestamp ? data.updatedAt.toDate() : null,
          } as CourseMaterial;
        })
        .filter((material) => material.chatId === selectedChatId)
        .sort((a, b) => {
          const aTime = a.createdAt ? a.createdAt.getTime() : 0;
          const bTime = b.createdAt ? b.createdAt.getTime() : 0;
          return bTime - aTime;
        });

      setMaterials(next);
    });

    return () => unsubscribe();
  }, [selectedChatId, user]);

  const handleUploadMaterialFiles = useCallback(async (fileList: FileList | File[] | null) => {
    if (!fileList || fileList.length === 0) return;
    if (!user || !selectedChatId || !currentChat) {
      showToast('Select a chat before uploading materials.', 'warning');
      return;
    }

    const files = Array.from(fileList);
    const unsupported = files.filter((file) => !isSupportedMaterialFile(file));
    if (unsupported.length > 0) {
      showToast(`Unsupported file type: ${unsupported[0].name}`, 'warning');
      return;
    }

    setMaterialsUploading(true);
    try {
      for (const file of files) {
        // eslint-disable-next-line no-await-in-loop
        await uploadAndIndexMaterial({
          file,
          userId: user.uid,
          courseId: currentChat.courseId,
          sessionId: currentChat.sessionId,
          chatId: selectedChatId,
        });
        showToast(`Indexed ${file.name}`, 'success');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to upload material';
      showToast(message, 'warning');
    } finally {
      setMaterialsUploading(false);
    }
  }, [currentChat, selectedChatId, showToast, user]);

  const handleDeleteMaterial = useCallback(async (materialId: string) => {
    if (!user) return;

    try {
      await deleteMaterial({ materialId, userId: user.uid });
      showToast('Material deleted', 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to delete material';
      showToast(message, 'warning');
    }
  }, [showToast, user]);

  return {
    materials,
    materialsUploading,
    handleUploadMaterialFiles,
    handleDeleteMaterial,
  };
}
