"use client";

import clsx from "clsx";
import { useEffect, useMemo, useState } from "react";

import { AppSettings, AppSettingsOptions, SettingsUpdatePayload } from "../../lib/types";

type SettingsDialogProps = {
  open: boolean;
  initialSettings: AppSettings | null;
  options: AppSettingsOptions | null;
  isSaving: boolean;
  error?: string | null;
  onClose: () => void;
  onSubmit: (payload: SettingsUpdatePayload) => void | Promise<void>;
};

const EMBEDDING_PROVIDER_LABELS: Record<AppSettings["embeddingProvider"], string> = {
  openai: "text-embedding-3-small",
  huggingface: "Vietnamese_Embedding_v2"
};

const LLM_PROVIDER_LABELS: Record<AppSettings["llmProvider"], string> = {
  openai: "OpenAI",
  gemini: "Gemini"
};

export function SettingsDialog({
  open,
  initialSettings,
  options,
  isSaving,
  error,
  onClose,
  onSubmit
}: SettingsDialogProps) {
  const [formState, setFormState] = useState<AppSettings | null>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    if (initialSettings) {
      setFormState({ ...initialSettings });
    } else {
      setFormState(null);
    }
  }, [initialSettings, open]);

  const formattedLambda = useMemo(() => {
    return formState ? formState.retrievalLambda.toFixed(2) : "0.00";
  }, [formState]);

  if (!open) {
    return null;
  }

  const handleChange = <Key extends keyof AppSettings>(key: Key, value: AppSettings[Key]) => {
    setFormState((prev) => (prev ? { ...prev, [key]: value } : prev));
  };

  const clampValue = (value: number, min: number, max: number) => {
    return Math.min(Math.max(value, min), max);
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!formState) {
      return;
    }
    onSubmit({
      queryVariations: formState.queryVariations,
      retrievalK: formState.retrievalK,
      retrievalLambda: formState.retrievalLambda,
      embeddingProvider: formState.embeddingProvider,
      llmProvider: formState.llmProvider
    });
  };

  const isDisabled = isSaving || !formState || !options;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="w-full max-w-2xl rounded-2xl border border-border bg-card shadow-2xl shadow-black/15">
        <form onSubmit={handleSubmit} className="flex flex-col">
          <header className="flex items-center justify-between border-b border-border px-6 py-4">
            <h2 className="text-lg font-semibold text-text">Runtime settings</h2>
            <button
              type="button"
              onClick={onClose}
              className="rounded-full p-2 text-subtext transition hover:bg-muted hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
            >
              <span className="sr-only">Close settings</span>
              <svg
                className="h-4 w-4"
                viewBox="0 0 16 16"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
              >
                <path d="M12.5 4.12 11.38 3 8 6.38 4.62 3 3.5 4.12 6.88 7.5 3.5 10.88 4.62 12l3.38-3.38L11.38 12l1.12-1.12L9.12 7.5 12.5 4.12Z" fill="currentColor" />
              </svg>
            </button>
          </header>

          <div className="max-h-[70vh] overflow-y-auto px-6 py-5">
            {!formState || !options ? (
              error ? (
                <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-6 text-sm text-red-700">
                  {error}
                </div>
              ) : (
                <div className="rounded-xl border border-border bg-muted px-4 py-6 text-sm text-subtext">
                  Loading settings...
                </div>
              )
            ) : (
              <div className="space-y-6 text-sm text-text">
                <section className="space-y-3">
                  <h3 className="text-sm font-semibold uppercase tracking-wide text-subtext">Multi-query</h3>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <label className="flex flex-col gap-2">
                      <span className="text-xs font-medium uppercase tracking-wide text-subtext">
                        Query variations
                      </span>
                      <input
                        type="number"
                        className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/30"
                        min={options.queryVariations.min}
                        max={options.queryVariations.max}
                        step={options.queryVariations.step}
                        value={formState.queryVariations}
                        onChange={(event) => {
                          const raw = Number.parseInt(event.target.value, 10);
                          const resolved = Number.isNaN(raw)
                            ? options.queryVariations.min
                            : clampValue(raw, options.queryVariations.min, options.queryVariations.max);
                          handleChange("queryVariations", resolved);
                        }}
                      />
                      <span className="text-xs text-subtext">
                        Controls how many alternative queries are generated for retrieval fusion.
                      </span>
                    </label>
                  </div>
                </section>

                <section className="space-y-3">
                  <h3 className="text-sm font-semibold uppercase tracking-wide text-subtext">Retrieval</h3>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <label className="flex flex-col gap-2">
                      <span className="text-xs font-medium uppercase tracking-wide text-subtext">Top-k results</span>
                      <input
                        type="number"
                        className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/30"
                        min={options.retrievalK.min}
                        max={options.retrievalK.max}
                        step={options.retrievalK.step}
                        value={formState.retrievalK}
                        onChange={(event) => {
                          const raw = Number.parseInt(event.target.value, 10);
                          const resolved = Number.isNaN(raw)
                            ? options.retrievalK.min
                            : clampValue(raw, options.retrievalK.min, options.retrievalK.max);
                          handleChange("retrievalK", resolved);
                        }}
                      />
                      <span className="text-xs text-subtext">
                        Number of chunks retrieved per query before fusion.
                      </span>
                    </label>
                    <label className="flex flex-col gap-2">
                      <span className="text-xs font-medium uppercase tracking-wide text-subtext">
                        Lambda (MMR balance)
                      </span>
                      <div className="flex items-center gap-3">
                        <input
                          type="range"
                          className="flex-1 accent-primary"
                          min={options.retrievalLambda.min}
                          max={options.retrievalLambda.max}
                          step={options.retrievalLambda.step}
                          value={formState.retrievalLambda}
                          onChange={(event) => {
                            const raw = Number.parseFloat(event.target.value);
                            const resolved = Number.isNaN(raw)
                              ? options.retrievalLambda.min
                              : clampValue(raw, options.retrievalLambda.min, options.retrievalLambda.max);
                            handleChange("retrievalLambda", Number(resolved.toFixed(3)));
                          }}
                        />
                        <span className="w-14 text-right text-xs font-semibold text-text">{formattedLambda}</span>
                      </div>
                      <span className="text-xs text-subtext">
                        Adjust balance between relevance and diversity during retrieval.
                      </span>
                    </label>
                  </div>
                </section>

                <section className="space-y-3">
                  <h3 className="text-sm font-semibold uppercase tracking-wide text-subtext">Providers</h3>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <label className="flex flex-col gap-2">
                      <span className="text-xs font-medium uppercase tracking-wide text-subtext">Embeddings</span>
                      <select
                        className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/30"
                        value={formState.embeddingProvider}
                        onChange={(event) =>
                          handleChange("embeddingProvider", event.target.value as AppSettings["embeddingProvider"])
                        }
                      >
                        {options.embeddingProviders.map((provider) => (
                          <option key={provider} value={provider}>
                            {EMBEDDING_PROVIDER_LABELS[provider] ?? provider}
                          </option>
                        ))}
                      </select>
                      <span className="text-xs text-subtext">
                        Select the embedding model family for indexing and similarity search.
                      </span>
                    </label>
                    <label className="flex flex-col gap-2">
                      <span className="text-xs font-medium uppercase tracking-wide text-subtext">LLM</span>
                      <select
                        className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/30"
                        value={formState.llmProvider}
                        onChange={(event) =>
                          handleChange("llmProvider", event.target.value as AppSettings["llmProvider"])
                        }
                      >
                        {options.llmProviders.map((provider) => (
                          <option key={provider} value={provider}>
                            {LLM_PROVIDER_LABELS[provider] ?? provider}
                          </option>
                        ))}
                      </select>
                      <span className="text-xs text-subtext">
                        Choose which large language model API powers answers.
                      </span>
                    </label>
                  </div>
                </section>

                {error && (
                  <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                    {error}
                  </div>
                )}
              </div>
            )}
          </div>

          <footer className="flex items-center justify-end gap-3 border-t border-border px-6 py-4">
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl border border-border px-4 py-2 text-sm font-medium text-subtext transition hover:bg-muted hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isDisabled}
              className={clsx(
                "rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
                isDisabled && "cursor-not-allowed opacity-60"
              )}
            >
              {isSaving ? "Saving..." : "Save changes"}
            </button>
          </footer>
        </form>
      </div>
    </div>
  );
}

export default SettingsDialog;

