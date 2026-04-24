"use client";

import { ShieldCheck, X } from "lucide-react";
import { useEffect, useState } from "react";

import type { AppSettings, SettingsResponse, SettingsUpdatePayload, UsageSummary } from "../lib/types";

type SettingsDrawerProps = {
  open: boolean;
  onClose: () => void;
  settings: SettingsResponse | null;
  usage: UsageSummary | null;
  isSaving: boolean;
  error: string | null;
  onSave: (payload: SettingsUpdatePayload) => Promise<void>;
};

export default function SettingsDrawer({
  open,
  onClose,
  settings,
  usage,
  isSaving,
  error,
  onSave,
}: SettingsDrawerProps) {
  const [draft, setDraft] = useState<AppSettings | null>(null);

  useEffect(() => {
    if (open && settings) {
      const { options: _options, ...rest } = settings;
      setDraft({ ...rest });
    }
  }, [open, settings]);

  if (!open || !settings || !draft) return null;

  const diff = buildDiff(settings, draft);
  const hasChanges = Object.keys(diff).length > 0;

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div
        role="button"
        aria-label="Close settings"
        tabIndex={0}
        onClick={onClose}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") onClose();
        }}
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
        className="relative z-10 flex h-full w-full max-w-md flex-col bg-surface shadow-float"
      >
        <div className="flex items-center justify-between border-b border-border px-5 py-4">
          <div>
            <h2 className="text-base font-semibold">Settings</h2>
            <p className="text-[11px] text-subtext">Tune retrieval, generation, and safety.</p>
          </div>
          <button onClick={onClose} className="btn-ghost h-9 w-9 p-0" aria-label="Close settings">
            <X className="h-4 w-4" aria-hidden />
          </button>
        </div>

        <div className="flex-1 space-y-6 overflow-y-auto px-5 py-5">
          <section className="space-y-2">
            <SectionTitle>Models</SectionTitle>
            <Field label="Chat model">
              <select
                className="input"
                value={draft.chatModel}
                onChange={(e) => setDraft({ ...draft, chatModel: e.target.value })}
              >
                {settings.options.chatModels.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Utility model (router & reranker)">
              <select
                className="input"
                value={draft.utilityModel}
                onChange={(e) => setDraft({ ...draft, utilityModel: e.target.value })}
              >
                {settings.options.utilityModels.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Embedding model">
              <input className="input" value={draft.embeddingModel} disabled />
            </Field>
          </section>

          <section className="space-y-3">
            <SectionTitle>Retrieval</SectionTitle>
            <RangeField
              label="Query variations"
              description="Multi-query expansion count (0 disables)."
              range={settings.options.queryVariations}
              value={draft.queryVariations}
              onChange={(v) => setDraft({ ...draft, queryVariations: v })}
            />
            <RangeField
              label="Retrieve K"
              description="Candidates per query variation before reranking."
              range={settings.options.retrievalK}
              value={draft.retrievalK}
              onChange={(v) => setDraft({ ...draft, retrievalK: v })}
            />
            <RangeField
              label="Rerank top N"
              description="Excerpts kept after the LLM reranker."
              range={settings.options.rerankTopN}
              value={draft.rerankTopN}
              onChange={(v) => setDraft({ ...draft, rerankTopN: v })}
            />
            <RangeField
              label="Cache threshold"
              description="Cosine similarity above which a past answer is reused."
              range={settings.options.semanticCacheThreshold}
              value={draft.semanticCacheThreshold}
              onChange={(v) => setDraft({ ...draft, semanticCacheThreshold: v })}
              format={(v) => v.toFixed(2)}
            />
          </section>

          <section className="space-y-2">
            <SectionTitle>Safety</SectionTitle>
            <label className="flex items-start gap-3 rounded-lg border border-border bg-panel p-3">
              <input
                type="checkbox"
                className="mt-1 h-4 w-4 accent-primary"
                checked={draft.enableModeration}
                onChange={(e) => setDraft({ ...draft, enableModeration: e.target.checked })}
              />
              <span>
                <span className="flex items-center gap-1 text-sm font-medium text-text">
                  <ShieldCheck className="h-4 w-4 text-primary" aria-hidden /> OpenAI moderation
                </span>
                <span className="text-[12px] text-subtext">
                  Pre-flight check on user input. Recommended for classroom deployments.
                </span>
              </span>
            </label>
          </section>

          {usage && (
            <section className="space-y-2">
              <SectionTitle>Usage (last {usage.totalEvents} events)</SectionTitle>
              <div className="grid grid-cols-3 gap-2 text-[12px]">
                <Stat label="Tokens" value={usage.totalTokens.toLocaleString()} />
                <Stat label="Cost" value={`$${usage.totalCostUsd.toFixed(4)}`} />
                <Stat label="Cache hits" value={`${usage.cacheHits}`} />
              </div>
              <div className="grid grid-cols-2 gap-2 text-[12px]">
                <Stat label="Avg latency" value={`${(usage.avgLatencyMs / 1000).toFixed(2)}s`} />
                <Stat
                  label="Models used"
                  value={Object.keys(usage.byModel).length.toString()}
                />
              </div>
            </section>
          )}
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-border px-5 py-3">
          <span className="text-[11px] text-subtext">
            {hasChanges ? `${Object.keys(diff).length} change(s) pending` : "No changes"}
          </span>
          <div className="flex gap-2">
            <button onClick={onClose} className="btn-secondary" type="button">
              Close
            </button>
            <button
              type="button"
              disabled={!hasChanges || isSaving}
              onClick={() => void onSave(diff)}
              className="btn-primary"
            >
              {isSaving ? "Saving…" : "Save"}
            </button>
          </div>
        </div>
        {error && <div className="px-5 py-2 text-[12px] text-danger">{error}</div>}
      </div>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-[11px] font-semibold uppercase tracking-wide text-subtext">{children}</h3>
  );
}

function Field({
  label,
  children,
  description,
}: {
  label: string;
  children: React.ReactNode;
  description?: string;
}) {
  return (
    <div>
      <label className="mb-1 block text-[12px] font-medium text-text">{label}</label>
      {children}
      {description && <p className="mt-1 text-[11px] text-subtext">{description}</p>}
    </div>
  );
}

function RangeField({
  label,
  description,
  range,
  value,
  onChange,
  format,
}: {
  label: string;
  description?: string;
  range: { min: number; max: number; step: number };
  value: number;
  onChange: (value: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[12px] text-text">
        <span className="font-medium">{label}</span>
        <span className="tabular-nums text-subtext">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range"
        className="w-full accent-primary"
        min={range.min}
        max={range.max}
        step={range.step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {description && <p className="mt-1 text-[11px] text-subtext">{description}</p>}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-panel p-2">
      <div className="text-[11px] uppercase tracking-wide text-subtext">{label}</div>
      <div className="tabular-nums text-sm font-semibold text-text">{value}</div>
    </div>
  );
}

function buildDiff(current: AppSettings, draft: AppSettings): SettingsUpdatePayload {
  const diff: SettingsUpdatePayload = {};
  if (current.chatModel !== draft.chatModel) diff.chatModel = draft.chatModel;
  if (current.utilityModel !== draft.utilityModel) diff.utilityModel = draft.utilityModel;
  if (current.queryVariations !== draft.queryVariations) diff.queryVariations = draft.queryVariations;
  if (current.retrievalK !== draft.retrievalK) diff.retrievalK = draft.retrievalK;
  if (current.rerankTopN !== draft.rerankTopN) diff.rerankTopN = draft.rerankTopN;
  if (current.semanticCacheThreshold !== draft.semanticCacheThreshold)
    diff.semanticCacheThreshold = draft.semanticCacheThreshold;
  if (current.enableModeration !== draft.enableModeration)
    diff.enableModeration = draft.enableModeration;
  return diff;
}
