"use client";

import { ChangeEvent, FormEvent, useRef, useState } from "react";

import { DocumentDetail, DocumentItem } from "../../lib/types";

type Props = {
  documents: DocumentItem[];
  onUpsert: (payload: { id?: string; filename: string; content?: string; file?: File }) => Promise<void>;
  onDelete: (documentId: string) => Promise<void>;
  onLoadDocument: (documentId: string) => Promise<DocumentDetail>;
};

type FormState = {
  id?: string;
  filename: string;
  content: string;
  file: File | null;
};

const emptyForm: FormState = { filename: "", content: "", file: null };

export function DocumentManager({ documents, onUpsert, onDelete, onLoadDocument }: Props) {
  const [form, setForm] = useState<FormState>(emptyForm);
  const [isBusy, setIsBusy] = useState(false);
  const [mode, setMode] = useState<"create" | "edit">("create");
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const hydratedDocuments = documents.slice(0, 50); // defensive safeguard

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedFilename = (form.filename || form.file?.name || "").trim();

    setIsBusy(true);
    setError(null);
    try {
      if (mode === "create") {
        if (!form.file) {
          throw new Error("Please choose a document to upload.");
        }
        if (!trimmedFilename) {
          throw new Error("Filename is required.");
        }
        await onUpsert({
          filename: trimmedFilename,
          file: form.file
        });
        setForm(emptyForm);
        setMode("create");
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      } else {
        if (!trimmedFilename || !form.content.trim()) {
          throw new Error("Filename and content cannot be empty.");
        }
        await onUpsert({
          id: form.id,
          filename: trimmedFilename,
          content: form.content
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save document.");
    } finally {
      setIsBusy(false);
    }
  };

  const handleEdit = async (documentId: string) => {
    setIsBusy(true);
    setError(null);
    try {
      const detail = await onLoadDocument(documentId);
      setForm({
        id: detail.id,
        filename: detail.filename,
        content: detail.content,
        file: null
      });
      setMode("edit");
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load document.");
    } finally {
      setIsBusy(false);
    }
  };

  const handleDelete = async (documentId: string) => {
    if (!confirm("Delete this document? This will remove it from the vector store.")) {
      return;
    }
    setIsBusy(true);
    setError(null);
    try {
      await onDelete(documentId);
      if (form.id === documentId) {
        setForm(emptyForm);
        setMode("create");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete document.");
    } finally {
      setIsBusy(false);
    }
  };

  return (
    <section className="document-manager">
      <header className="document-manager__header">
        <div>
          <h2>Documents</h2>
          <span>Upload and maintain the knowledge base for retrieval.</span>
        </div>
        <button
          className="document-manager__toggle"
          onClick={() => {
            setMode("create");
            setForm(emptyForm);
            if (fileInputRef.current) {
              fileInputRef.current.value = "";
            }
          }}
          disabled={isBusy}
        >
          New Document
        </button>
      </header>

      <div className="document-manager__content">
        <div className="document-manager__list">
          {hydratedDocuments.length === 0 && <p className="document-manager__empty">No documents uploaded yet.</p>}
          {hydratedDocuments.map((doc) => (
            <article key={doc.id} className="document-card">
              <div className="document-card__header">
                <div>
                  <h3>{doc.filename}</h3>
                  {doc.summary && <p className="document-card__summary">{doc.summary}</p>}
                </div>
                <div className="document-card__actions">
                  <button onClick={() => void handleEdit(doc.id)} disabled={isBusy}>
                    Edit
                  </button>
                  <button onClick={() => void handleDelete(doc.id)} disabled={isBusy}>
                    Delete
                  </button>
                </div>
              </div>
              <dl className="document-card__meta">
                <div>
                  <dt>Uploaded</dt>
                  <dd>{new Date(doc.uploadedAt).toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Updated</dt>
                  <dd>{new Date(doc.updatedAt).toLocaleString()}</dd>
                </div>
              </dl>
            </article>
          ))}
        </div>

        <form className="document-form" onSubmit={handleSubmit}>
          <h3>{mode === "create" ? "Upload new document" : "Edit document"}</h3>
          {mode === "create" ? (
            <label>
              Document
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.pdf,.docx"
                onChange={(event: ChangeEvent<HTMLInputElement>) => {
                  const nextFile = event.target.files?.[0] ?? null;
                  setForm((current) => ({
                    ...current,
                    file: nextFile,
                    filename: nextFile ? nextFile.name : current.filename
                  }));
                }}
                disabled={isBusy}
              />
              <span className="document-form__hint">Supports .txt, .pdf, and .docx files.</span>
            </label>
          ) : (
            <>
              <label>
                Filename
                <input
                  type="text"
                  value={form.filename}
                  onChange={(event) => setForm((current) => ({ ...current, filename: event.target.value }))}
                  placeholder="knowledge-base.txt"
                  disabled={isBusy}
                />
              </label>
              <label>
                Content
                <textarea
                  value={form.content}
                  onChange={(event) => setForm((current) => ({ ...current, content: event.target.value }))}
                  rows={12}
                  disabled={isBusy}
                />
              </label>
            </>
          )}
          {error && <p className="document-form__error">{error}</p>}
          <div className="document-form__actions">
            {mode === "edit" && (
              <button
                type="button"
                onClick={() => {
                  setMode("create");
                  setForm(emptyForm);
                  setError(null);
                  if (fileInputRef.current) {
                    fileInputRef.current.value = "";
                  }
                }}
                disabled={isBusy}
              >
                Cancel
              </button>
            )}
            <button type="submit" disabled={isBusy}>
              {mode === "create" ? "Upload" : "Save changes"}
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}

export default DocumentManager;
