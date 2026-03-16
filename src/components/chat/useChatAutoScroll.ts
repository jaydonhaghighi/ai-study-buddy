import { useEffect, useRef } from 'react';

const AUTO_SCROLL_BOTTOM_THRESHOLD_PX = 72;

type UseChatAutoScrollParams = {
  selectedChatId: string | null;
  mainView: 'chat' | 'dashboard';
  messages: unknown[];
  loading: boolean;
};

export function useChatAutoScroll({
  selectedChatId,
  mainView,
  messages,
  loading,
}: UseChatAutoScrollParams) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const autoScrollEnabledRef = useRef(true);
  const wasLoadingRef = useRef(false);

  const isNearBottom = () => {
    const container = messagesContainerRef.current;
    if (!container) return true;
    const distanceToBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    return distanceToBottom <= AUTO_SCROLL_BOTTOM_THRESHOLD_PX;
  };

  const handleMessagesScroll = () => {
    autoScrollEnabledRef.current = isNearBottom();
  };

  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  };

  const enableAutoScroll = () => {
    autoScrollEnabledRef.current = true;
  };

  useEffect(() => {
    autoScrollEnabledRef.current = true;
    const raf = window.requestAnimationFrame(() => {
      scrollToBottom('auto');
    });
    return () => window.cancelAnimationFrame(raf);
  }, [selectedChatId, mainView]);

  useEffect(() => {
    const justFinishedGeneration = wasLoadingRef.current && !loading;
    wasLoadingRef.current = loading;
    if (justFinishedGeneration) return;
    if (!autoScrollEnabledRef.current) return;
    scrollToBottom(loading ? 'auto' : 'smooth');
  }, [messages, loading]);

  return {
    messagesEndRef,
    messagesContainerRef,
    handleMessagesScroll,
    enableAutoScroll,
  };
}
