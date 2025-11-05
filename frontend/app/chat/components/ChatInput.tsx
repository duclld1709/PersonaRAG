"use client";

import clsx from "clsx";
import {
  ChangeEvent,
  KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef
} from "react";

type ModelOption = {
  value: string;
  label: string;
};

type ChatInputProps = {
  value: string;
  onChange: (event: ChangeEvent<HTMLTextAreaElement>) => void;
  onSend: (message: string) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  modelOptions?: ModelOption[];
  selectedModel?: string;
  onSelectModel?: (value: string) => void;
  disableModelSelect?: boolean;
};

const LINE_HEIGHT_PX = 24; // matches leading-6
const MAX_LINES = 3;
const MAX_HEIGHT_PX = LINE_HEIGHT_PX * MAX_LINES;

const SendIcon = ({ className }: { className?: string }) => (
  <svg
    className={className}
    viewBox="0 0 20 20"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    aria-hidden="true"
  >
    <path d="M3.75 3.5L16.25 10L3.75 16.5L5.75 11.25H11.25L5.75 8.75L3.75 3.5Z" fill="currentColor" />
  </svg>
);

export function ChatInput({
  value,
  onChange,
  onSend,
  placeholder = "Type a message...",
  disabled = false,
  className,
  modelOptions,
  selectedModel,
  onSelectModel,
  disableModelSelect = false
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const trimmed = value.trim();
  const isSendDisabled = disabled || trimmed.length === 0;

  const adjustHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    textarea.style.height = "auto";
    const scrollHeight = textarea.scrollHeight;
    const clampedHeight = Math.min(scrollHeight, MAX_HEIGHT_PX);
    textarea.style.height = `${Math.max(LINE_HEIGHT_PX, clampedHeight)}px`;
    textarea.style.overflowY = scrollHeight > MAX_HEIGHT_PX ? "auto" : "hidden";
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [value, adjustHeight]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    const observer = new ResizeObserver(() => adjustHeight());
    observer.observe(textarea);
    return () => observer.disconnect();
  }, [adjustHeight]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        if (!isSendDisabled) {
          onSend(trimmed);
        }
      }
    },
    [isSendDisabled, onSend, trimmed]
  );

  const handleClickSend = useCallback(() => {
    if (!isSendDisabled) {
      onSend(trimmed);
    }
  }, [isSendDisabled, onSend, trimmed]);

  const containerClass = useMemo(
    () =>
      clsx(
        "flex w-full max-w-3xl items-center gap-3 rounded-3xl border border-border bg-card px-4 py-3 shadow-lg shadow-black/10 transition focus-within:border-primary focus-within:shadow-[0_0_0_4px_rgba(16,185,129,0.15)] sm:px-5",
        "overflow-visible bg-clip-padding",
        className
      ),
    [className]
  );

  const shouldRenderModelSelect = Array.isArray(modelOptions) && modelOptions.length > 0 && typeof onSelectModel === "function";

  return (
    <div className={containerClass}>
      <textarea
        ref={textareaRef}
        aria-label="Message"
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        onKeyDown={handleKeyDown}
        rows={1}
        disabled={disabled}
        className="flex-1 resize-none bg-transparent text-sm leading-6 text-text placeholder:text-subtext focus:outline-none focus-visible:outline-none disabled:text-subtext"
      />
      {shouldRenderModelSelect ? (
        <div className="inline-flex items-center gap-2 rounded-full border border-border bg-muted/60 px-2 py-1 text-xs font-medium text-subtext transition hover:border-primary hover:text-text sm:px-3 sm:py-2">
          <span className="text-xs uppercase tracking-wide text-subtext">Model</span>
          <select
            value={selectedModel}
            onChange={(event) => onSelectModel(event.target.value)}
            disabled={disableModelSelect || disabled}
            className="rounded-md border border-transparent bg-transparent text-xs font-semibold text-text focus:border-primary focus:outline-none disabled:text-subtext"
          >
            {modelOptions!.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      ) : null}
      <button
        type="button"
        aria-label="Send message"
        onClick={handleClickSend}
        disabled={isSendDisabled}
        className="inline-flex items-center gap-2 rounded-full bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition hover:brightness-95 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary disabled:cursor-not-allowed disabled:bg-muted disabled:text-subtext"
      >
        <SendIcon className="h-4 w-4" />
        <span>Send</span>
      </button>
    </div>
  );
}

export default ChatInput;

