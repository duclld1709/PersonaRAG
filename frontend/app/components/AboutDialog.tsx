"use client";

import clsx from "clsx";
import { X } from "lucide-react";
import { ReactNode } from "react";

type AboutDialogProps = {
  open: boolean;
  onClose: () => void;
  title?: string;
  children?: ReactNode;
};

export function AboutDialog({ open, onClose, title = "About this application", children }: AboutDialogProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className={clsx("w-full max-w-xl rounded-2xl border border-border bg-card shadow-2xl shadow-black/10")}>
        <header className="flex items-center justify-between border-b border-border px-6 py-4">
          <h2 className="text-lg font-semibold text-text">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full p-2 text-subtext transition hover:bg-muted hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
            aria-label="Close about dialog"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </header>
        <div className="px-6 py-5 text-sm leading-relaxed text-text">
          {children ?? (
            <div className="space-y-3">
              <p>
                This Retrieval-Augmented Generation chatbot helps you explore knowledge sources that you upload,
                combining FastAPI, LangChain, and a modern Next.js frontend for a responsive experience.
              </p>
              <p>
                Upload documents, curate your knowledge base, and ask questions to receive contextual answers powered by
                vector search and large language models.
              </p>
              <p className="text-subtext">
                Built for experimentation and rapid iteration. Contributions and feedback are welcome!
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AboutDialog;
