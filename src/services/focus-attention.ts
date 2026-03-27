const focusedAttentionLabels = ['screen', 'away_down'] as const;

export const FOCUSED_ATTENTION_LABELS = new Set<string>(focusedAttentionLabels);

export function isFocusedAttentionLabel(label: string | null | undefined): boolean {
  return label != null && FOCUSED_ATTENTION_LABELS.has(label);
}
