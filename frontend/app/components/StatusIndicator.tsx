"use client";

import type { StreamStatus } from "../lib/types";

const LABELS: Record<StreamStatus, string> = {
  routing: "Understanding your question",
  retrieving: "Searching your materials",
  reranking: "Ranking the best excerpts",
  generating: "Drafting an answer",
  cache_hit: "Recalled from cache",
  chitchat: "Replying",
  refused: "Refused",
  done: "Done",
};

export default function StatusIndicator({ status }: { status: StreamStatus | null }) {
  if (!status || status === "done") return null;
  return (
    <div
      className="inline-flex items-center gap-2 rounded-full border border-border bg-panel px-3 py-1 text-[12px] text-subtext"
      role="status"
      aria-live="polite"
    >
      <span className="status-pulse" aria-hidden />
      <span>{LABELS[status] ?? status}</span>
    </div>
  );
}
