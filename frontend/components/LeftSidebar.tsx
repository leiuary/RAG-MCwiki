"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Box, Database, RefreshCw, Zap, ChevronDown } from "lucide-react";
import { KnowledgeBaseStatus, EmbeddingModelInfo } from "../app/services/api";

interface LeftSidebarProps {
  kbStatus: KnowledgeBaseStatus | null;
  onRefresh: () => void;
  embeddingModelId: string;
  embeddingModels: EmbeddingModelInfo[];
  onSwitchEmbeddingModel: (modelId: string) => void;
}

export default function LeftSidebar({
  kbStatus,
  onRefresh,
  embeddingModelId,
  embeddingModels,
  onSwitchEmbeddingModel,
}: LeftSidebarProps) {
  const [modelSelectOpen, setModelSelectOpen] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(0);
  const [switching, setSwitching] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const currentModel = embeddingModels.find((m) => m.is_active);
  const builtModels = embeddingModels.filter((m) => m.is_built);

  // 关闭下拉时重置高亮
  useEffect(() => {
    if (!modelSelectOpen) {
      const activeIdx = embeddingModels.findIndex((m) => m.is_active);
      setHighlightIndex(activeIdx >= 0 ? activeIdx : 0);
    }
  }, [modelSelectOpen, embeddingModels]);

  // 点击外部关闭下拉
  useEffect(() => {
    if (!modelSelectOpen) return;
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setModelSelectOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [modelSelectOpen]);

  const selectableModels = embeddingModels.filter((m) => m.is_built);

  const handleSwitch = async (modelId: string) => {
    if (modelId === embeddingModelId || switching) return;
    setSwitching(true);
    try {
      await onSwitchEmbeddingModel(modelId);
    } finally {
      setSwitching(false);
      setModelSelectOpen(false);
    }
  };

  const navigateHighlight = useCallback((delta: number) => {
    setHighlightIndex((prev) => {
      const max = embeddingModels.length - 1;
      let next = prev + delta;
      if (next < 0) next = max;
      if (next > max) next = 0;
      // 跳过不可构建的
      if (!embeddingModels[next].is_built) return prev;
      return next;
    });
  }, [embeddingModels]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        navigateHighlight(1);
        break;
      case "ArrowUp":
        e.preventDefault();
        navigateHighlight(-1);
        break;
      case "Enter":
        e.preventDefault();
        const target = embeddingModels[highlightIndex];
        if (target && target.is_built) {
          handleSwitch(target.id);
        }
        break;
      case "Escape":
        e.preventDefault();
        setModelSelectOpen(false);
        break;
    }
  }, [embeddingModels, highlightIndex, navigateHighlight]);

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-border bg-card p-6">
      {/* Brand */}
      <div className="mb-10 flex items-center gap-3 text-foreground">
        <Box size={22} strokeWidth={2.5} />
        <h1 className="text-lg font-semibold tracking-tight">MCwiki 助手</h1>
      </div>

      {/* Status sections */}
      <div className="flex flex-col gap-8">
        {/* Core engine status */}
        <div className="flex flex-col gap-3.5">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
            <Zap size={15} />
            <span>核心引擎状态</span>
          </div>
          <div className="flex flex-col gap-3.5 rounded-xl border border-border bg-muted/50 p-4.5">
            <div className="flex items-center justify-between text-sm">
              <span className="text-[13px] font-medium text-muted-foreground">服务状态</span>
              <span className="font-semibold text-foreground">在线</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-[13px] font-medium text-muted-foreground">内核版本</span>
              <span className="font-semibold text-foreground">v0.1.0-alpha</span>
            </div>
          </div>
        </div>

        {/* Vector DB status */}
        <div className="flex flex-col gap-3.5">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
            <Database size={15} />
            <span>向量数据库 (RAG)</span>
            <button
              className="ml-auto text-muted-foreground transition-colors hover:text-foreground"
              onClick={onRefresh}
              aria-label="刷新向量数据库状态"
              title="刷新状态"
            >
              <RefreshCw size={13} />
            </button>
          </div>
          <div className="flex flex-col gap-3.5 rounded-xl border border-border bg-muted/50 p-4.5">
            <div className="flex items-center justify-between text-sm">
              <span className="text-[13px] font-medium text-muted-foreground">运行状态</span>
              <span
                className={`font-semibold ${
                  kbStatus?.status === "ready"
                    ? "text-foreground"
                    : "text-muted-foreground"
                }`}
              >
                {kbStatus?.status === "ready" ? "● 已就绪" : "○ 未初始化"}
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-[13px] font-medium text-muted-foreground">数据版本</span>
              <span className="font-semibold text-foreground">{kbStatus?.version || "Unknown"}</span>
            </div>

            {/* Embedding Model Selector */}
            <div className="flex flex-col items-start gap-2 text-sm">
              <span className="text-[13px] font-medium text-muted-foreground">Embedding 模型</span>
              <div className="relative w-full">
                <button
                  className="flex w-full items-center justify-between rounded-lg border border-border bg-card px-3 py-2.5 text-xs text-foreground transition-colors hover:border-foreground/30"
                  onClick={() => setModelSelectOpen(!modelSelectOpen)}
                  onKeyDown={modelSelectOpen ? handleKeyDown : (e) => { if (e.key === "ArrowDown" || e.key === "Enter") { e.preventDefault(); setModelSelectOpen(true); } }}
                  disabled={switching}
                >
                  <span className="truncate text-left">
                    {switching ? "切换中..." : (currentModel?.name || "未加载")}
                  </span>
                  <ChevronDown
                    size={14}
                    className={`ml-1 shrink-0 text-muted-foreground transition-transform ${
                      modelSelectOpen ? "rotate-180" : ""
                    }`}
                  />
                </button>
                {modelSelectOpen && (
                  <div
                    ref={dropdownRef}
                    className="absolute left-0 right-0 top-full z-20 mt-1 rounded-lg border border-border bg-card py-1 shadow-lg"
                    onKeyDown={handleKeyDown}
                  >
                    {embeddingModels.map((m, idx) => (
                      <button
                        key={m.id}
                        className={`flex w-full items-center gap-2 px-3 py-2 text-xs transition-colors ${
                          idx === highlightIndex
                            ? "bg-accent text-foreground"
                            : m.is_active
                              ? "bg-muted text-foreground font-medium"
                              : m.is_built
                                ? "text-foreground hover:bg-muted/50"
                                : "text-muted-foreground/50 cursor-not-allowed"
                        }`}
                        onClick={() => m.is_built && handleSwitch(m.id)}
                        onMouseEnter={() => m.is_built && setHighlightIndex(idx)}
                        disabled={!m.is_built}
                        title={m.description}
                      >
                        <span
                          className={`size-1.5 shrink-0 rounded-full ${
                            m.is_active
                              ? "bg-green-500"
                              : m.is_built
                                ? "bg-yellow-500"
                                : "bg-neutral-400"
                          }`}
                        />
                        <span className="flex-1 truncate text-left">{m.name}</span>
                        <span className="text-[10px] text-muted-foreground">{m.dim}d</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
              {builtModels.length > 1 && (
                <p className="text-[10px] text-muted-foreground">
                  已构建 {builtModels.length}/{embeddingModels.length} 个模型
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-auto border-t border-border pt-5 text-center text-xs font-medium text-muted-foreground">
        © 2026 RAG-MCWIKI
      </div>
    </aside>
  );
}
