"use client";

import { Gauge, Zap } from "lucide-react";

type CostMeterProps = {
  costUsd: number;
  latencyMs: number;
  cacheHit: boolean;
  /** USD per pipeline stage (router, expand, retrieve, rerank, generate, ...). */
  breakdown?: Record<string, number>;
};

export default function CostMeter({ costUsd, latencyMs, cacheHit, breakdown = {} }: CostMeterProps) {
  const cost = costUsd > 0 ? `$${costUsd.toFixed(4)}` : "—";
  const latency = latencyMs > 0 ? `${(latencyMs / 1000).toFixed(2)}s` : "—";
  const stages = Object.entries(breakdown)
    .filter(([, usd]) => usd > 0)
    .sort(([, a], [, b]) => b - a)
    .map(([stage, usd]) => `${stage}: $${usd.toFixed(5)}`);
  const title = ["Last response: total cost (all model calls) / latency", ...stages].join("\n");
  return (
    <div
      className="flex items-center gap-3 rounded-lg border border-border bg-panel px-3 py-1.5 text-[11px] text-subtext"
      title={title}
    >
      <span className="inline-flex items-center gap-1">
        <Gauge className="h-3 w-3" aria-hidden />
        <span className="tabular-nums">{latency}</span>
      </span>
      <span className="inline-flex items-center gap-1">
        <Zap className="h-3 w-3" aria-hidden />
        <span className="tabular-nums">{cost}</span>
      </span>
      {cacheHit && (
        <span className="rounded-full bg-primary-soft px-2 py-0.5 text-[10px] font-medium text-primary">
          cache
        </span>
      )}
    </div>
  );
}
