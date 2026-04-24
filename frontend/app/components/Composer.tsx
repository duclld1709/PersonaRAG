"use client";

import { CornerDownLeft, Square } from "lucide-react";
import { useEffect, useRef, type KeyboardEvent } from "react";

import type { ChatMode } from "../lib/types";
import ModeSelect from "./ModeSelect";

type ComposerProps = {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled: boolean;
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
  placeholder?: string;
};

export default function Composer({
  value,
  onChange,
  onSend,
  onStop,
  isStreaming,
  disabled,
  mode,
  onModeChange,
  placeholder,
}: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(220, el.scrollHeight)}px`;
  }, [value]);

  const handleKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!isStreaming && value.trim()) onSend();
    }
  };

  return (
    <div className="border-t border-border bg-surface px-3 py-3 md:px-6">
      <div className="mx-auto flex max-w-3xl flex-col gap-2">
        <ModeSelect value={mode} onChange={onModeChange} compact />
        <div className="card flex items-end gap-2 p-2">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKey}
            rows={1}
            placeholder={
              placeholder ??
              "Ask a grounded question about your materials — e.g. 'Explain cross-validation with an example.'"
            }
            disabled={disabled}
            className="textarea border-0 bg-transparent px-2 py-1.5 text-[15px] shadow-none focus:ring-0"
          />
          {isStreaming ? (
            <button type="button" onClick={onStop} className="btn-secondary h-9">
              <Square className="h-4 w-4" aria-hidden />
              <span>Stop</span>
            </button>
          ) : (
            <button
              type="button"
              onClick={onSend}
              disabled={disabled || !value.trim()}
              className="btn-primary h-9"
            >
              <CornerDownLeft className="h-4 w-4" aria-hidden />
              <span>Send</span>
            </button>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2 px-1 text-[11px] text-subtext">
          <span>Enter to send · Shift+Enter for newline</span>
          <span className="dot" />
          <span>Answers are grounded only in your uploaded materials.</span>
        </div>
      </div>
    </div>
  );
}
