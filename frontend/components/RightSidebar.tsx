"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { getModels } from "../app/services/api";
import { Sliders, Globe, Key, ListFilter, RefreshCw, Edit3 } from "lucide-react";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { Switch } from "./ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

interface RightSidebarProps {
  modelChoice: string;
  setModelChoice: (val: string) => void;
  answerDetail: string;
  setAnswerDetail: (val: string) => void;
  professionalMode: boolean;
  setProfessionalMode: (val: boolean) => void;

  baseUrl: string;
  setBaseUrl: (val: string) => void;
  apiKey: string;
  setApiKey: (val: string) => void;
  modelName: string;
  setModelName: (val: string) => void;
}

export default function RightSidebar({
  modelChoice,
  setModelChoice,
  answerDetail,
  setAnswerDetail,
  professionalMode,
  setProfessionalMode,
  baseUrl,
  setBaseUrl,
  apiKey,
  setApiKey,
  modelName,
  setModelName,
}: RightSidebarProps) {
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isRefreshingModels, setIsRefreshingModels] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [isManualMode, setIsManualMode] = useState(false);
  const hasEverSucceeded = useRef(false);

  const effectiveBaseUrl =
    baseUrl ||
    (modelChoice === "api"
      ? "https://api.deepseek.com"
      : "http://localhost:1234/v1");

  const getProtocolHint = () => {
    if (effectiveBaseUrl.includes("deepseek")) return "DeepSeek 官方接口";
    if (effectiveBaseUrl.includes("openai") || effectiveBaseUrl.includes("localhost"))
      return "OpenAI 兼容协议";
    return "自定义接口";
  };

  const fetchModelsList = useCallback(async () => {
    if (modelChoice === "api" && !apiKey) {
      setAvailableModels([]);
      setFetchError(null);
      return;
    }
    if (!effectiveBaseUrl) return;

    setIsRefreshingModels(true);
    setFetchError(null);
    try {
      const { models } = await getModels(modelChoice, effectiveBaseUrl, apiKey);
      if (!models || models.length === 0) throw new Error("接口未返回任何可用模型");
      setAvailableModels(models);
      setFetchError(null);
      hasEverSucceeded.current = true;
      const currentModel = modelName;
      if (!currentModel || !models.includes(currentModel)) {
        setModelName(models[0]);
      }
    } catch (e: unknown) {
      const errorMsg = e instanceof Error ? e.message : "连接失败";
      console.error("加载模型列表失败", errorMsg);
      setAvailableModels([]);
      if (hasEverSucceeded.current) {
        setFetchError(errorMsg);
        setIsManualMode(true);
      }
    } finally {
      setIsRefreshingModels(false);
    }
  }, [modelChoice, effectiveBaseUrl, apiKey, modelName, setModelName]);

  useEffect(() => {
    const timer = setTimeout(() => fetchModelsList(), 600);
    return () => clearTimeout(timer);
  }, [modelChoice, baseUrl, apiKey, fetchModelsList]);

  const handleProviderChange = (choice: "local" | "api") => {
    setModelChoice(choice);
    setAvailableModels([]);
    setFetchError(null);
    setIsManualMode(false);
    if (choice === "local") {
      setBaseUrl("http://localhost:1234/v1");
    } else {
      if (baseUrl.includes("localhost") || baseUrl.includes("127.0.0.1")) {
        setBaseUrl("");
      }
    }
  };

  return (
    <aside className="flex w-80 shrink-0 flex-col border-l border-border bg-card">
      <div className="flex flex-col gap-6 overflow-y-auto p-5">
        {/* Section: 推理配置 */}
        <section className="flex flex-col gap-4">
          <div className="flex items-center gap-2.5">
            <Sliders size={18} />
            <h3 className="text-sm font-semibold tracking-wide">推理配置</h3>
          </div>

          {/* 供应商切换 */}
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">供应商</label>
            <div className="flex rounded-md bg-muted p-0.5">
              <button
                className={`flex-1 rounded-sm px-3 py-1.5 text-xs font-medium transition-all ${
                  modelChoice === "api"
                    ? "bg-card text-foreground shadow-sm"
                    : "text-muted-foreground"
                }`}
                onClick={() => handleProviderChange("api")}
              >
                API 接口
              </button>
              <button
                className={`flex-1 rounded-sm px-3 py-1.5 text-xs font-medium transition-all ${
                  modelChoice === "local"
                    ? "bg-card text-foreground shadow-sm"
                    : "text-muted-foreground"
                }`}
                onClick={() => handleProviderChange("local")}
              >
                LM Studio
              </button>
            </div>
          </div>

          {/* 接口 URL */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <Globe size={15} className="text-muted-foreground" />
              <label htmlFor="base-url" className="text-sm font-medium">
                接口 URL
              </label>
            </div>
            <Input
              id="base-url"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://api.deepseek.com"
            />
            <p className="text-[11px] text-muted-foreground">{getProtocolHint()}</p>
          </div>

          {/* API 密钥 */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <Key size={15} className="text-muted-foreground" />
              <label htmlFor="api-key" className="text-sm font-medium">
                API 密钥
              </label>
            </div>
            <Input
              id="api-key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="填写密钥后加载模型"
            />
          </div>

          {/* 模型选择 */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <ListFilter size={15} className="text-muted-foreground" />
              <label className="text-sm font-medium">模型选择</label>
              <div className="ml-auto flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon-xs"
                  onClick={() => setIsManualMode(!isManualMode)}
                  className={isManualMode ? "text-foreground" : "text-muted-foreground"}
                >
                  <Edit3 size={13} />
                </Button>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  onClick={fetchModelsList}
                  disabled={isManualMode || (modelChoice === "api" && !apiKey)}
                >
                  <RefreshCw
                    size={14}
                    className={isRefreshingModels ? "animate-spin" : ""}
                  />
                </Button>
              </div>
            </div>

            {isManualMode ? (
              <Input
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="手动输入模型 ID"
              />
            ) : (
              <Select
                value={modelName}
                onValueChange={(v) => setModelName(v || "")}
                disabled={
                  isRefreshingModels ||
                  (modelChoice === "api" && !apiKey) ||
                  availableModels.length === 0
                }
              >
                <SelectTrigger className="w-full">
                  <SelectValue>
                    {isRefreshingModels
                      ? "拉取列表中..."
                      : modelChoice === "api" && !apiKey
                        ? "等待输入 API 密钥"
                        : availableModels.length > 0 && modelName
                          ? modelName
                          : fetchError || "未找到可用模型"}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {availableModels.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>

          {/* 回答精度 */}
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">回答精度</label>
            <div className="flex rounded-md bg-muted p-0.5">
              {["简洁", "标准", "详细"].map((d) => (
                <button
                  key={d}
                  onClick={() => setAnswerDetail(d)}
                  className={`flex-1 rounded-sm px-3 py-1.5 text-xs font-medium transition-all ${
                    answerDetail === d
                      ? "bg-card text-foreground shadow-sm"
                      : "text-muted-foreground"
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          {/* 专业模式 */}
          <div className="flex items-center justify-between">
            <div className="flex flex-col gap-0.5">
              <label className="text-sm font-medium">专业模式</label>
              <p className="text-[11px] text-muted-foreground">
                开启后在回答下方展示全链路数据
              </p>
            </div>
            <Switch
              checked={professionalMode}
              onCheckedChange={setProfessionalMode}
            />
          </div>
        </section>
      </div>
    </aside>
  );
}
