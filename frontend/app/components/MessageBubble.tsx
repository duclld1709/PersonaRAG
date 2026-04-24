"use client";

import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import React, { useMemo, useState } from "react";

import { cn } from "../lib/cn";
import type { Citation, ConversationMessage } from "../lib/types";

type MessageBubbleProps = {
  message: ConversationMessage;
  isStreaming?: boolean;
  onCitationHover?: (citation: Citation | null) => void;
};

export default function MessageBubble({ message, isStreaming, onCitationHover }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const citations = message.citations ?? [];
  const [showSources, setShowSources] = useState(false);

  const contentComponents = useMemo(() => buildMarkdownComponents(citations, onCitationHover), [
    citations,
    onCitationHover,
  ]);

  return (
    <div
      className={cn(
        "flex w-full gap-3",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {!isUser && (
        <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary-soft text-primary">
          <Bot className="h-4 w-4" aria-hidden />
        </div>
      )}
      <div
        className={cn(
          "group max-w-2xl rounded-2xl px-4 py-3",
          isUser
            ? "bg-primary text-primary-foreground"
            : "border border-border bg-surface text-text shadow-soft"
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap text-[15px] leading-7">{message.content}</p>
        ) : (
          <div className="prose-chat">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={contentComponents}>
              {message.content || (isStreaming ? " " : "")}
            </ReactMarkdown>
            {isStreaming && (
              <span className="ml-1 inline-block h-3 w-1.5 translate-y-[3px] animate-pulse-soft bg-text" />
            )}
          </div>
        )}

        {!isUser && citations.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-divider pt-2">
            <button
              type="button"
              onClick={() => setShowSources((prev) => !prev)}
              className="text-[11px] font-medium text-subtext hover:text-text"
            >
              {showSources ? "Hide" : "Show"} {citations.length} source
              {citations.length > 1 ? "s" : ""}
            </button>
            {message.usage && (
              <span className="chip tabular-nums">
                {message.usage.promptTokens + message.usage.completionTokens} tok
              </span>
            )}
            {typeof message.latencyMs === "number" && (
              <span className="chip tabular-nums">{(message.latencyMs / 1000).toFixed(2)}s</span>
            )}
            {message.mode && message.mode !== "answer" && (
              <span className="chip capitalize">{message.mode}</span>
            )}
          </div>
        )}

        {!isUser && showSources && citations.length > 0 && (
          <ol className="mt-2 space-y-2 border-t border-divider pt-2 text-[12px]">
            {citations.map((citation, idx) => (
              <li key={citation.chunkId ?? `${citation.source}-${idx}`} className="flex gap-2">
                <span className="citation-pill shrink-0">[{idx + 1}]</span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-1 text-[11px] text-subtext">
                    <span className="font-medium text-text">{citation.filename}</span>
                    {citation.tag && <span className="chip capitalize">{citation.tag.replace("_", " ")}</span>}
                    {typeof citation.score === "number" && (
                      <span className="tabular-nums">score {citation.score.toFixed(3)}</span>
                    )}
                  </div>
                  <p className="mt-1 line-clamp-3 text-subtext">{citation.content}</p>
                </div>
              </li>
            ))}
          </ol>
        )}
      </div>
      {isUser && (
        <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-muted text-subtext">
          <User className="h-4 w-4" aria-hidden />
        </div>
      )}
    </div>
  );
}

function buildMarkdownComponents(
  citations: Citation[],
  onCitationHover?: (citation: Citation | null) => void
) {
  const wrap = (children: React.ReactNode): React.ReactNode =>
    processChildren(children, citations, onCitationHover);
  return {
    p: ({ children }: any) => <p>{wrap(children)}</p>,
    li: ({ children }: any) => <li>{wrap(children)}</li>,
    td: ({ children }: any) => <td>{wrap(children)}</td>,
    th: ({ children }: any) => <th>{wrap(children)}</th>,
    a: ({ children, href }: any) => (
      <a href={href} target="_blank" rel="noreferrer" className="text-primary hover:underline">
        {children}
      </a>
    ),
  } as const;
}

function processChildren(
  children: React.ReactNode,
  citations: Citation[],
  onCitationHover?: (citation: Citation | null) => void
): React.ReactNode {
  if (typeof children === "string") {
    return linkifyCitations(children, citations, onCitationHover);
  }
  if (Array.isArray(children)) {
    return children.map((child, idx) =>
      typeof child === "string" ? (
        <React.Fragment key={idx}>{linkifyCitations(child, citations, onCitationHover)}</React.Fragment>
      ) : (
        child
      )
    );
  }
  return children;
}

function linkifyCitations(
  text: string,
  citations: Citation[],
  onCitationHover?: (citation: Citation | null) => void
): React.ReactNode {
  if (!text.includes("[")) return text;
  const parts: React.ReactNode[] = [];
  const regex = /\[(\d+)\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const idx = Number(match[1]) - 1;
    const citation = citations[idx];
    if (citation) {
      parts.push(
        <span
          key={`cite-${match.index}`}
          className="citation-pill cursor-help"
          onMouseEnter={() => onCitationHover?.(citation)}
          onMouseLeave={() => onCitationHover?.(null)}
          title={`${citation.filename} — ${citation.content.slice(0, 140)}…`}
        >
          [{idx + 1}]
        </span>
      );
    } else {
      parts.push(match[0]);
    }
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return <>{parts}</>;
}
