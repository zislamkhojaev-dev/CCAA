"use client";

import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { backendWsUrl } from "@/lib/ws";

type Source = {
  document_id: string;
  chunk_id: string;
  text: string;
  score: number;
  title?: string;
};

type Suggestion = {
  answer: string;
  intent: { intent: string; confidence: number; requires_human: boolean };
  sources: Source[];
  latency_ms: Record<string, number>;
  warning?: string | null;
  customer_text: string;
  ts: number;
};

export default function AgentAssistPage() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [locale, setLocale] = useState<"ru" | "uz">("ru");
  const [draft, setDraft] = useState("");
  const [latest, setLatest] = useState<Suggestion | null>(null);
  const [history, setHistory] = useState<Suggestion[]>([]);
  const [error, setError] = useState<string | null>(null);
  const lastSentRef = useRef<string>("");

  function connect() {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(backendWsUrl(`/ws/agent-assist?locale=${locale}`));
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setError("Не удалось подключиться к серверу");
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === "suggestion") {
          const sug: Suggestion = {
            answer: data.answer,
            intent: data.intent,
            sources: data.sources || [],
            latency_ms: data.latency_ms || {},
            warning: data.warning,
            customer_text: lastSentRef.current,
            ts: Date.now(),
          };
          setLatest(sug);
          setHistory((h) => [sug, ...h].slice(0, 12));
        }
      } catch {
        // ignore malformed
      }
    };
  }

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [locale]);

  function send(speaker: "customer" | "agent", text: string) {
    if (!text.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (speaker === "customer") lastSentRef.current = text.trim();
    wsRef.current.send(JSON.stringify({ type: "utterance", speaker, text }));
  }

  function reset() {
    wsRef.current?.send(JSON.stringify({ type: "reset" }));
    setHistory([]);
    setLatest(null);
  }

  return (
    <div className="space-y-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-3xl font-semibold">Agent Assist (Суфлёр)</h2>
          <p className="text-mutedForeground mt-1">
            Симуляция рабочего места оператора. Имитируйте реплики клиента — система мгновенно
            подбирает подсказку из базы знаний.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone={connected ? "success" : "danger"}>
            {connected ? "WS подключен" : "Нет соединения"}
          </Badge>
          <select
            value={locale}
            onChange={(e) => setLocale(e.target.value as "ru" | "uz")}
            className="rounded-md border border-border bg-background px-3 py-1.5 text-sm"
          >
            <option value="ru">Русский</option>
            <option value="uz">O‘zbekcha</option>
          </select>
          <Button variant="outline" size="sm" onClick={reset}>
            Сброс
          </Button>
        </div>
      </header>

      {error && <Badge tone="danger">{error}</Badge>}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Реплика клиента</CardTitle>
            <CardDescription>
              Как только нажмёте «Передать клиенту», подсказка появится справа.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Textarea
              rows={4}
              placeholder="Например: добрый день, какая комиссия на международные переводы?"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
            />
            <div className="flex justify-end gap-2">
              <Button
                variant="outline"
                onClick={() => {
                  send("agent", draft);
                  setDraft("");
                }}
              >
                Записать как реплику оператора
              </Button>
              <Button
                onClick={() => {
                  send("customer", draft);
                  setDraft("");
                }}
              >
                Передать клиенту
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Текущая подсказка</CardTitle>
            <CardDescription>
              Если intent операционный — здесь появится предупреждение.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!latest && (
              <p className="text-sm text-mutedForeground">
                Жду первую реплику клиента…
              </p>
            )}
            {latest && <SuggestionView item={latest} highlight />}
          </CardContent>
        </Card>
      </div>

      {history.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle>История реплик</CardTitle>
            <CardDescription>Последние 12 запросов клиента в этой сессии.</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {history.slice(1).map((s) => (
                <li key={s.ts}>
                  <SuggestionView item={s} />
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function SuggestionView({ item, highlight = false }: { item: Suggestion; highlight?: boolean }) {
  return (
    <div
      className={`rounded-md border p-4 space-y-3 ${
        highlight ? "border-primary bg-accent/40" : "border-border"
      }`}
    >
      <div className="text-xs text-mutedForeground">
        <span className="font-medium">Клиент:</span> {item.customer_text}
      </div>
      <div className="flex items-center gap-2 flex-wrap">
        <Badge tone={item.intent.requires_human ? "warning" : "success"}>
          {item.intent.intent}
        </Badge>
        <Badge>{(item.intent.confidence * 100).toFixed(0)}%</Badge>
        {Object.entries(item.latency_ms).map(([k, v]) => (
          <Badge key={k}>{`${k}: ${v.toFixed(0)} мс`}</Badge>
        ))}
      </div>
      {item.warning && <Badge tone="warning">{item.warning}</Badge>}
      {item.answer ? (
        <p className="text-base leading-relaxed">{item.answer}</p>
      ) : (
        <p className="text-sm text-mutedForeground">Подсказка не сгенерирована.</p>
      )}
      {item.sources.length > 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer text-mutedForeground">
            Источники ({item.sources.length})
          </summary>
          <ul className="mt-2 space-y-1">
            {item.sources.map((s) => (
              <li key={s.chunk_id}>
                <span className="font-medium">{s.title ?? s.chunk_id}</span> —{" "}
                <span className="text-mutedForeground">score {s.score.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
