"use client";

import { FileText, Loader2, Trash2, Upload } from "lucide-react";
import { useRef, useState } from "react";

import { cn } from "../lib/cn";
import type { DocumentItem, DocumentTag } from "../lib/types";

const TAG_OPTIONS: { value: DocumentTag; label: string }[] = [
  { value: "lecture_notes", label: "Lecture notes" },
  { value: "paper", label: "Paper" },
  { value: "textbook", label: "Textbook" },
  { value: "assignment", label: "Assignment" },
  { value: "reference", label: "Reference" },
  { value: "other", label: "Other" },
];

type DocumentPanelProps = {
  documents: DocumentItem[];
  isBusy: boolean;
  error: string | null;
  onUpload: (file: File, tag: DocumentTag) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
};

export default function DocumentPanel({
  documents,
  isBusy,
  error,
  onUpload,
  onDelete,
}: DocumentPanelProps) {
  const [tag, setTag] = useState<DocumentTag>("lecture_notes");
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    await onUpload(files[0], tag);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-border px-5 py-4">
        <h2 className="text-base font-semibold">Library</h2>
        <p className="text-[12px] text-subtext">
          Your uploaded course materials. Answers are grounded only in these sources.
        </p>
      </div>

      <div className="px-5 pt-4">
        <label className="mb-1 block text-[11px] font-medium uppercase tracking-wide text-subtext">
          Tag this upload
        </label>
        <select
          className="input"
          value={tag}
          onChange={(e) => setTag(e.target.value as DocumentTag)}
        >
          {TAG_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      <div
        className={cn(
          "mx-5 mt-3 flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-6 text-center transition-colors",
          isDragging ? "border-primary bg-primary-soft" : "border-border bg-panel"
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          void handleFiles(e.dataTransfer.files);
        }}
      >
        {isBusy ? (
          <Loader2 className="h-6 w-6 animate-spin text-primary" aria-hidden />
        ) : (
          <Upload className="h-6 w-6 text-subtext" aria-hidden />
        )}
        <div className="text-sm">
          Drag & drop a file or{" "}
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="text-primary underline-offset-2 hover:underline"
          >
            browse
          </button>
        </div>
        <div className="text-[11px] text-subtext">
          Accepts .pdf, .docx, .md, .txt — up to 10 MB.
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt,.md"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {error && (
        <div className="mx-5 mt-3 rounded-lg border border-danger/30 bg-danger/5 p-2 text-xs text-danger">
          {error}
        </div>
      )}

      <div className="mt-4 flex-1 overflow-y-auto px-2 pb-4">
        {documents.length === 0 ? (
          <div className="px-3 py-6 text-center text-xs text-subtext">
            No documents yet.
          </div>
        ) : (
          <ul className="space-y-1" role="list">
            {documents.map((doc) => (
              <li
                key={doc.id}
                className="group relative flex items-start gap-3 rounded-lg px-3 py-2 transition-colors hover:bg-muted"
              >
                <FileText className="mt-0.5 h-4 w-4 shrink-0 text-subtext" aria-hidden />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className="line-clamp-1 text-sm font-medium text-text">
                      {doc.filename}
                    </span>
                    <span className="chip capitalize">{doc.tag.replace("_", " ")}</span>
                  </div>
                  <div className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[11px] text-subtext">
                    <span>{doc.chunkCount} chunks</span>
                    <span className="dot" />
                    <span>{new Date(doc.updatedAt).toLocaleDateString()}</span>
                  </div>
                  {doc.summary && (
                    <p className="mt-1 line-clamp-2 text-[12px] text-subtext">{doc.summary}</p>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => onDelete(doc.id)}
                  className="opacity-0 transition-opacity group-hover:opacity-100"
                  aria-label={`Delete ${doc.filename}`}
                  title="Delete"
                >
                  <Trash2 className="h-4 w-4 text-subtext hover:text-danger" aria-hidden />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
