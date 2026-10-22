"use client";

import { BookOpen, Lightbulb, ListChecks, Scale } from "lucide-react";

import { cn } from "../lib/cn";
import { useAutoScroll } from "../hooks/useAutoScroll";
import type { Citation, ChatMode, ConversationDetail, StreamStatus } from "../lib/types";
import MessageBubble from "./MessageBubble";
import StatusIndicator from "./StatusIndicator";

type MessageListProps = {
  conversation: ConversationDetail | null;
  streamingAnswer: string;
  streamingCitations: Citation[];
  isStreaming: boolean;
  status: StreamStatus | null;
  error: string | null;
  onCitationHover?: (citation: Citation | null) => void;
  onSuggestion: (question: string, mode: ChatMode) => void;
};

const PROMPT_SUGGESTIONS: Array<{ icon: React.ReactNode; label: string; question: string; mode: ChatMode }> = [
  {
    icon: <Lightbulb className="h-4 w-4" aria-hidden />,
    label: "Explain a concept",
    question: "Explain overfitting with an example from my notes.",
    mode: "explain",
  },
  {
    icon: <BookOpen className="h-4 w-4" aria-hidden />,
    label: "Summarise a source",
    question: "Summarise the main arguments in my uploaded paper.",
    mode: "summarize",
  },
  {
    icon: <Scale className="h-4 w-4" aria-hidden />,
    label: "Compare two sources",
    question: "Compare qualitative and quantitative research as described in my materials.",
    mode: "compare",
  },
  {
    icon: <ListChecks className="h-4 w-4" aria-hidden />,
    label: "Generate a study quiz",
    question: "Generate a 5-question quiz from my lecture notes.",
    mode: "quiz",
  },
];

export default function MessageList({
  conversation,
  streamingAnswer,
  streamingCitations,
  isStreaming,
  status,
  error,
  onCitationHover,
  onSuggestion,
}: MessageListProps) {
  const scrollRef = useAutoScroll<HTMLDivElement>(
    (conversation?.messages.length ?? 0) + streamingAnswer.length + (status ?? "")
  );

  const messages = conversation?.messages ?? [];
  const isEmpty = messages.length === 0 && !isStreaming;

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto scroll-smooth"
      aria-label="Conversation transcript"
    >
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-6">
        {isEmpty ? (
          <EmptyState onSuggestion={onSuggestion} />
        ) : (
          <>
            {messages.map((msg, idx) => (
              <MessageBubble key={idx} message={msg} onCitationHover={onCitationHover} />
            ))}
            {isStreaming && (
              <MessageBubble
                message={{
                  role: "assistant",
                  content: streamingAnswer,
                  createdAt: new Date().toISOString(),
                  citations: streamingCitations,
                }}
                isStreaming
                onCitationHover={onCitationHover}
              />
            )}
          </>
        )}

        {status && (
          <div className={cn("pl-11", isEmpty && "pl-0")}>
            <StatusIndicator status={status} />
          </div>
        )}
        {error && (
          <div className="card border-danger/30 bg-danger/5 p-3 text-sm text-danger">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

function EmptyState({ onSuggestion }: { onSuggestion: (question: string, mode: ChatMode) => void }) {
  return (
    <div className="mt-8 flex flex-col items-center gap-6 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary-soft text-primary">
        <BookOpen className="h-7 w-7" aria-hidden />
      </div>
      <div className="space-y-2">
        <h2 className="text-2xl font-semibold tracking-tight">Ask your materials anything</h2>
        <p className="max-w-md text-sm text-subtext">
          Upload lecture notes, papers, or a textbook chapter on the left, then ask grounded questions.
          Every answer is cited back to your sources.
        </p>
      </div>
      <div className="grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
        {PROMPT_SUGGESTIONS.map((s) => (
          <button
            key={s.label}
            type="button"
            onClick={() => onSuggestion(s.question, s.mode)}
            className="card flex items-start gap-3 p-3 text-left transition-colors hover:border-primary/40 hover:bg-primary-soft"
          >
            <span className="mt-0.5 text-primary">{s.icon}</span>
            <span className="flex-1">
              <span className="block text-sm font-medium text-text">{s.label}</span>
              <span className="block text-[12px] text-subtext">{s.question}</span>
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
