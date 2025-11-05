"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ConversationDetail } from "../../lib/types";
import ChatInput from "./ChatInput";
import { MessageBubble } from "./MessageBubble";

type ModelOption = {
  value: string;
  label: string;
};

type Props = {
  conversation: ConversationDetail | null;
  streamingAnswer: string;
  isStreaming: boolean;
  onSend: (question: string) => Promise<void>;
  modelOptions?: ModelOption[];
  selectedModel?: string;
  onSelectModel?: (value: string) => void;
  disableModelSelect?: boolean;
};

export function ChatWindow({
  conversation,
  streamingAnswer,
  isStreaming,
  onSend,
  modelOptions,
  selectedModel,
  onSelectModel,
  disableModelSelect
}: Props) {
  const [question, setQuestion] = useState("");
  const messagesRef = useRef<HTMLElement | null>(null);
  const hasScrolled = useRef(false);

  const messages = conversation?.messages ?? [];

  const scrollToBottom = useCallback(
    (behavior: ScrollBehavior = "auto") => {
      const container = messagesRef.current;
      if (!container) {
        return;
      }
      try {
        container.scrollTo({ top: container.scrollHeight, behavior });
      } catch {
        container.scrollTop = container.scrollHeight;
      }
    },
    []
  );

  useEffect(() => {
    const behavior = hasScrolled.current ? ("smooth" as ScrollBehavior) : "auto";
    scrollToBottom(behavior);
    hasScrolled.current = true;
  }, [messages.length, scrollToBottom]);

  useEffect(() => {
    if (isStreaming || streamingAnswer) {
      scrollToBottom("smooth");
    }
  }, [isStreaming, streamingAnswer, scrollToBottom]);

  const handleSend = async (input: string) => {
    if (isStreaming || !input.trim()) {
      return;
    }
    setQuestion("");
    await onSend(input);
  };

  return (
    <div className="flex h-full w-full min-h-0 min-w-0 flex-1 flex-col">
      <section
        id="messages"
        ref={messagesRef}
        className="flex-1 min-h-0 overflow-y-auto px-4 py-6 scroll-smooth md:px-6"
      >
        <div className="space-y-4">
          {messages.length === 0 && !isStreaming ? (
            <div className="rounded-2xl border border-border bg-muted px-4 py-6 text-sm text-subtext">
              No messages yet. Ask your first question to get started.
            </div>
          ) : (
            <>
              {messages.map((message) => (
                <MessageBubble key={`${message.role}-${message.createdAt}`} message={message} />
              ))}
              {isStreaming && (
                <MessageBubble
                  message={{
                    role: "assistant",
                    content: streamingAnswer,
                    createdAt: new Date().toISOString()
                  }}
                  isStreaming
                />
              )}
            </>
          )}
        </div>
      </section>

      <footer className="sticky bottom-0 border-t border-border bg-background px-4 pb-4 pt-4 backdrop-blur md:px-6 md:pt-6">
        <div className="mx-auto max-w-5xl pb-[max(1rem,env(safe-area-inset-bottom))]">
          <ChatInput
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onSend={handleSend}
            placeholder="Ask anything..."
            disabled={isStreaming}
            className="w-full"
            modelOptions={modelOptions}
            selectedModel={selectedModel}
            onSelectModel={onSelectModel}
            disableModelSelect={disableModelSelect}
          />
        </div>
      </footer>
    </div>
  );
}

export default ChatWindow;
