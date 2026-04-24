"use client";

import { BookOpen, FileStack, ListChecks, Scale, Sparkles, Wand2 } from "lucide-react";

import { cn } from "../lib/cn";
import type { ChatMode } from "../lib/types";

type Option = {
  value: ChatMode;
  label: string;
  hint: string;
  icon: React.ReactNode;
};

const OPTIONS: Option[] = [
  { value: "answer", label: "Answer", hint: "Direct answer grounded in your sources.", icon: <Sparkles className="h-3.5 w-3.5" aria-hidden /> },
  { value: "explain", label: "Explain", hint: "First-year friendly explanation.", icon: <Wand2 className="h-3.5 w-3.5" aria-hidden /> },
  { value: "summarize", label: "Summarize", hint: "Structured summary of the excerpts.", icon: <FileStack className="h-3.5 w-3.5" aria-hidden /> },
  { value: "compare", label: "Compare", hint: "Compare viewpoints across sources.", icon: <Scale className="h-3.5 w-3.5" aria-hidden /> },
  { value: "outline", label: "Outline", hint: "Essay / report outline with a thesis.", icon: <BookOpen className="h-3.5 w-3.5" aria-hidden /> },
  { value: "quiz", label: "Quiz", hint: "5-question study quiz with answers.", icon: <ListChecks className="h-3.5 w-3.5" aria-hidden /> },
];

export default function ModeSelect({
  value,
  onChange,
  compact,
}: {
  value: ChatMode;
  onChange: (mode: ChatMode) => void;
  compact?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap gap-1",
        compact ? "gap-1" : "gap-1.5"
      )}
      role="radiogroup"
      aria-label="Response mode"
    >
      {OPTIONS.map((opt) => {
        const selected = value === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={selected}
            onClick={() => onChange(opt.value)}
            title={opt.hint}
            className={cn(
              "inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[12px] font-medium transition-colors",
              selected
                ? "border-primary/30 bg-primary-soft text-primary"
                : "border-border bg-panel text-subtext hover:text-text"
            )}
          >
            {opt.icon}
            <span>{opt.label}</span>
          </button>
        );
      })}
    </div>
  );
}
