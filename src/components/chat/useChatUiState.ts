import { useCallback, useEffect, useRef, useState } from 'react';

type ToastVariant = 'success' | 'warning' | 'info';

export function useChatUiState() {
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [toastVariant, setToastVariant] = useState<ToastVariant>('success');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const settingsRef = useRef<HTMLDivElement>(null);

  const showToast = useCallback((message: string, variant: ToastVariant = 'success') => {
    setToastVariant(variant);
    setToastMessage(message);
  }, []);

  useEffect(() => {
    if (!settingsOpen) return;
    const onMouseDown = (e: MouseEvent) => {
      const el = settingsRef.current;
      if (!el) return;
      if (e.target instanceof Node && !el.contains(e.target)) {
        setSettingsOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSettingsOpen(false);
    };
    document.addEventListener('mousedown', onMouseDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [settingsOpen]);

  useEffect(() => {
    if (!toastMessage) return;
    const timer = setTimeout(() => setToastMessage(null), 3000);
    return () => clearTimeout(timer);
  }, [toastMessage]);

  return {
    toastMessage,
    toastVariant,
    showToast,
    settingsOpen,
    setSettingsOpen,
    settingsRef,
  };
}
