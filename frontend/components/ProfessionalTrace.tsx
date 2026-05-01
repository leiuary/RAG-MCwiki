"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { TraceData, Source } from "../app/services/api";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";

interface Props {
  trace: TraceData;
  sources?: Source[];
}

function timingPct(ms: number, maxMs: number) {
  return maxMs > 0 ? Math.min((ms / maxMs) * 100, 100) : 0;
}

export default function ProfessionalTrace({ trace, sources }: Props) {
  const [showPrompt, setShowPrompt] = useState(false);
  const [openChunks, setOpenChunks] = useState<Record<number, boolean>>({});

  const stepEntries = trace.step_durations_ms
    ? Object.entries(trace.step_durations_ms)
    : [];
  const maxStepMs = Math.max(...stepEntries.map(([, ms]) => ms), 1);

  return (
    <div className="mt-8 space-y-5">
      {/* Overview metrics */}
      <div className="flex flex-wrap gap-2">
        <Metric label="TTFT" value={`${trace.ttft_ms || 0}ms`} />
        <Metric label="总耗时" value={`${((trace.total_time_ms || 0) / 1000).toFixed(2)}s`} />
        <Metric label="检索片段" value={trace.retrieved_chunk_count ?? 0} />
        <Metric label="上下文" value={`${((trace.context_total_chars || 0) / 1000).toFixed(1)}k 字`} />
        <Metric label="输出" value={`${trace.output_chars || 0} 字`} />
        <Metric label="输入 Token" value={`${trace.token_estimated ? "~" : ""}${trace.prompt_tokens || 0}`} />
        <Metric label="输出 Token" value={`${trace.token_estimated ? "~" : ""}${trace.completion_tokens || 0}`} />
        <Metric label="总 Token" value={`${trace.token_estimated ? "~" : ""}${trace.total_tokens || 0}`} />
        <Metric label="模式" value={trace.execution_mode || "stream"} mono />
      </div>

      {/* Stage timing bars */}
      {stepEntries.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-[11px] uppercase tracking-wider">阶段耗时</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {stepEntries.map(([step, ms]) => (
              <div key={step} className="grid grid-cols-[100px_56px_1fr] items-center gap-2.5">
                <span className="text-right text-xs text-muted-foreground">{step}</span>
                <span className="text-xs font-semibold tabular-nums text-foreground">{ms}ms</span>
                <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-muted-foreground/50 transition-all duration-500"
                    style={{ width: `${timingPct(ms, maxStepMs)}%`, minWidth: 2 }}
                  />
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Query rewrite tags */}
      {trace.search_terms && trace.search_terms.length > 1 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-[11px] uppercase tracking-wider">查询改写</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-1.5">
              {trace.search_terms.map((t, i) => (
                <Badge key={i} variant="secondary" className="text-xs">
                  {t}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Retrieved chunks */}
      {sources && sources.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-[11px] uppercase tracking-wider">
              检索片段 · {sources.length} 条
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-0">
            {sources.map((src, idx) => (
              <div key={idx} className="border-b border-border last:border-b-0">
                <button
                  className="flex w-full items-center gap-2 py-2 text-left text-[13px] text-muted-foreground transition-colors hover:text-foreground"
                  onClick={() =>
                    setOpenChunks((prev) => ({ ...prev, [idx]: !prev[idx] }))
                  }
                >
                  {openChunks[idx] ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                  <span className="min-w-6 text-[11px] font-semibold text-muted-foreground">
                    #{idx + 1}
                  </span>
                  <span className="flex-1 truncate">
                    {src.title}{src.section ? ` › ${src.section}` : ""}
                  </span>
                  <span className="shrink-0 text-[11px] tabular-nums text-muted-foreground">
                    {src.content_length || 0} 字
                  </span>
                  {src.source_url && (
                    <a
                      href={src.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="shrink-0 rounded border border-border px-1.5 py-0.5 text-[11px] text-muted-foreground no-underline transition-colors hover:text-foreground hover:border-foreground"
                      onClick={(e) => e.stopPropagation()}
                    >
                      源
                    </a>
                  )}
                </button>
                {openChunks[idx] && src.content_preview && (
                  <div className="max-h-96 overflow-y-auto pb-3 pl-10 text-xs leading-relaxed text-muted-foreground whitespace-pre-wrap">
                    {src.content_preview}
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* System prompt */}
      {trace.system_prompt && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-[11px] uppercase tracking-wider">
              系统提示词
              <button
                onClick={() => setShowPrompt(!showPrompt)}
                className="ml-auto rounded border border-border px-2 py-0.5 text-[11px] font-medium text-muted-foreground transition-colors hover:text-foreground hover:border-foreground"
              >
                {showPrompt ? "收起" : "展开"}
              </button>
            </CardTitle>
          </CardHeader>
          {showPrompt && (
            <CardContent>
              <pre className="max-h-60 overflow-y-auto rounded-md bg-muted p-3 text-[11px] leading-relaxed text-muted-foreground whitespace-pre-wrap break-all">
                {trace.system_prompt}
              </pre>
            </CardContent>
          )}
        </Card>
      )}

      {/* LLM + Embedding config */}
      {trace.llm_config && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-[11px] uppercase tracking-wider">请求配置</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              {Object.entries(trace.llm_config).map(([k, v]) => (
                <div key={k} className="flex items-center justify-between border-b border-border py-1">
                  <span className="text-[11px] text-muted-foreground">{k}</span>
                  <span className="text-xs font-medium text-foreground font-mono max-w-[140px] truncate">
                    {String(v)}
                  </span>
                </div>
              ))}
              {trace.embedding_model && (
                <div className="flex items-center justify-between border-b border-border py-1">
                  <span className="text-[11px] text-muted-foreground">embed_model</span>
                  <span className="text-xs font-medium text-foreground font-mono max-w-[140px] truncate">
                    {trace.embedding_model}
                  </span>
                </div>
              )}
              {trace.embedding_device && (
                <div className="flex items-center justify-between border-b border-border py-1">
                  <span className="text-[11px] text-muted-foreground">embed_device</span>
                  <span className="text-xs font-medium text-foreground font-mono max-w-[140px] truncate">
                    {trace.embedding_device}
                  </span>
                </div>
              )}
            </div>
            {trace.fallback_used && (
              <div className="mt-3 rounded bg-amber-50 px-2.5 py-1.5 text-[11px] text-amber-800 dark:bg-amber-950 dark:text-amber-200">
                流式降级为 invoke · 原因: {trace.fallback_reason?.slice(0, 80)}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function Metric({ label, value, mono }: { label: string; value: string | number; mono?: boolean }) {
  return (
    <div className="flex flex-col gap-0.5 rounded-lg border border-border bg-card px-3.5 py-2">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <span className={`text-[15px] font-semibold text-foreground ${mono ? "font-mono text-[13px]" : ""}`}>
        {value}
      </span>
    </div>
  );
}
