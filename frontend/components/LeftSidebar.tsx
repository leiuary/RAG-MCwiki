"use client";

import { Box, Database, RefreshCw, Zap } from "lucide-react";
import { KnowledgeBaseStatus } from "../app/services/api";

interface LeftSidebarProps {
  kbStatus: KnowledgeBaseStatus | null;
  onRefresh: () => void;
}

export default function LeftSidebar({ kbStatus, onRefresh }: LeftSidebarProps) {
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
            <div className="flex flex-col items-start gap-2.5 text-sm">
              <span className="text-[13px] font-medium text-muted-foreground">Embedding 模型</span>
              <div className="w-full rounded-lg border border-border bg-card px-3 py-2.5 font-mono text-xs leading-relaxed text-foreground break-all shadow-[inset_0_1px_2px_rgba(0,0,0,0.02)]">
                {kbStatus?.embedding_model || "未加载模型"}
              </div>
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
