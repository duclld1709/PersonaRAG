"use client";

import { Compass, FileText } from "lucide-react";

import { cn } from "../lib/cn";
import type { Citation } from "../lib/types";

type CitationPanelProps = {
  citations: Citation[];
  hovered: Citation | null;
  isStreaming: boolean;
};

export default function CitationPanel({ citations, hovered, isStreaming }: CitationPanelProps) {
  return (
    <aside className="flex h-full w-[340px] shrink-0 flex-col border-l border-border bg-panel">
      <div className="border-b border-border px-5 py-4">
        <div className="flex items-center gap-2">
          <Compass className="h-4 w-4 text-primary" aria-hidden />
          <h2 className="text-sm font-semibold">Context</h2>
          {citations.length > 0 && (
            <span className="chip tabular-nums">{citations.length}</span>
          )}
        </div>
        <p className="mt-0.5 text-[11px] text-subtext">
          The excerpts the assistant used to answer. Hover a citation number in the reply to focus one here.
        </p>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3">
        {citations.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-6 py-10 text-center text-[12px] text-subtext">
            <FileText className="h-5 w-5" aria-hidden />
            <p>
              {isStreaming
                ? "Searching your materials…"
                : "No retrieved context yet. Ask a question to see the sources the assistant draws from."}
            </p>
          </div>
        ) : (
          <ol className="space-y-2">
            {citations.map((citation, idx) => (
              <li
                key={citation.chunkId ?? `${citation.source}-${idx}`}
                className={cn(
                  "rounded-xl border p-3 transition-colors",
                  hovered && hovered.chunkId === citation.chunkId
                    ? "border-primary/40 bg-primary-soft"
                    : "border-border bg-surface"
                )}
              >
                <div className="mb-1 flex items-center gap-1.5 text-[11px] text-subtext">
                  <span className="citation-pill">[{idx + 1}]</span>
                  <span className="line-clamp-1 font-medium text-text">{citation.filename}</span>
                </div>
                <div className="flex flex-wrap items-center gap-1 text-[11px] text-subtext">
                  {citation.tag && (
                    <span className="chip capitalize">{citation.tag.replace("_", " ")}</span>
                  )}
                  {typeof citation.score === "number" && (
                    <span className="chip tabular-nums">score {citation.score.toFixed(3)}</span>
                  )}
                  {typeof citation.page === "number" && (
                    <span className="chip tabular-nums">p.{citation.page}</span>
                  )}
                </div>
                <p className="mt-2 line-clamp-[10] text-[12px] leading-5 text-text">
                  {citation.content}
                </p>
              </li>
            ))}
          </ol>
        )}
      </div>
    </aside>
  );
}
