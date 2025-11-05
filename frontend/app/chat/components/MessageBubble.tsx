"use client";

import clsx from "clsx";
import { Info } from "lucide-react";
import Image from "next/image";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { ConversationMessage } from "../../lib/types";

type Props = {
  message: ConversationMessage;
  isStreaming?: boolean;
};

const roleMeta = {
  user: {
    avatarSrc: "/avatars/user.png",
    avatarColorFallback: "bg-primary text-primary-foreground",
    bubbleColor: "bg-primary text-primary-foreground",
    bubbleHover: "hover:brightness-95",
    initials: "U"
  },
  assistant: {
    avatarSrc: "/avatars/assistant.jpg",
    avatarColorFallback: "bg-assistant text-text",
    bubbleColor: "bg-assistant text-text",
    bubbleHover: "hover:bg-muted",
    initials: "AI"
  }
} as const;

type RoleMeta = (typeof roleMeta)[keyof typeof roleMeta];

export function MessageBubble({ message, isStreaming = false }: Props) {
  const isUser = message.role === "user";
  const meta: RoleMeta = isUser ? roleMeta.user : roleMeta.assistant;
  const [showCitations, setShowCitations] = useState(false);
  const hasAvatarSrc = Boolean(meta.avatarSrc);
  const avatarFallbackClass = hasAvatarSrc ? undefined : meta.avatarColorFallback;
  const messageContent = message.content ?? "";
  const hasStreamedContent = messageContent.trim().length > 0;

  const hasCitations = !isUser && Array.isArray(message.citations) && message.citations.length > 0;
  const showThinkingIndicator = !isUser && isStreaming && !hasStreamedContent;

  return (
    <div
      className={clsx(
        "group flex w-full items-start gap-3",
        isUser ? "flex-row-reverse text-right" : "text-left"
      )}
    >
      <span
        className={clsx(
          "flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-sm font-semibold text-white shadow-lg shadow-black/20 transition-transform group-hover:scale-105",
          avatarFallbackClass
        )}
      >
        {hasAvatarSrc ? (
          <Image
            src={meta.avatarSrc}
            alt={`${message.role} avatar`}
            width={40}
            height={40}
            className="h-10 w-10 rounded-full object-cover"
          />
        ) : (
          meta.initials ?? message.role.slice(0, 2).toUpperCase()
        )}
      </span>

      {showThinkingIndicator && (
        <div className="thinking-status" role="status" aria-live="polite">
          <span className="thinking-status__text">Thinking</span>
        </div>
      )}

      <div className={clsx("flex min-w-0 flex-1 flex-col gap-2", isUser ? "items-end" : "items-start")}>
        {(!showThinkingIndicator || isUser) && (
          <div
            className={clsx(
              "relative w-fit max-w-3xl rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm transition-colors duration-200",
              meta.bubbleColor,
              meta.bubbleHover,
              isUser ? "ml-auto" : "mr-auto"
            )}
          >
            {isUser ? (
              <p className="whitespace-pre-wrap">{messageContent}</p>
            ) : (
              <div className="markdown-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{messageContent}</ReactMarkdown>
              </div>
            )}
          </div>
        )}

        {!showThinkingIndicator && hasCitations && (
          <>
            <div
              className={clsx(
                "flex w-full",
                isUser ? "justify-end" : "justify-start"
              )}
            >
              <button
                type="button"
                onClick={() => setShowCitations((current) => !current)}
                className={clsx(
                  "inline-flex items-center gap-2 rounded-full border border-border bg-card/80 px-3 py-1 text-xs font-medium text-subtext transition",
                  "hover:bg-muted hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                )}
                aria-expanded={showCitations}
                aria-label={showCitations ? "Hide citations" : "Show citations"}
              >
                <Info className="h-4 w-4" aria-hidden="true" />
                <span>{showCitations ? "Hide sources" : "View sources"}</span>
              </button>
            </div>

            {showCitations && (
              <div
                className={clsx(
                  "w-full max-w-3xl",
                  isUser ? "ml-auto text-right" : "mr-auto text-left"
                )}
              >
                <div className="space-y-3 rounded-xl border border-border bg-card/95 p-3 text-sm text-text shadow-sm">
                  {message.citations!.map((citation, index) => (
                    <div
                      key={`${citation.source}-${index}`}
                      className="space-y-2 rounded-lg border border-border/50 bg-muted/80 p-3 text-left"
                    >
                      <div className="flex items-center justify-between text-xs font-semibold text-subtext">
                        <span className="text-text">
                          {citation.filename || "Unknown source"}
                        </span>
                        {citation.page != null && (
                          <span className="ml-2 text-xs text-subtext">
                            Page {citation.page}
                          </span>
                        )}
                      </div>
                      <div className="rounded-md bg-card/80 px-3 py-2 text-xs text-text shadow-inner shadow-black/5">
                        <p className="whitespace-pre-wrap">{citation.content}</p>
                      </div>
                      <p className="text-[11px] text-subtext">
                        {citation.source}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default MessageBubble;
