"use client";

import { useState } from "react";
import { API_BASE } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

type Source = {
  document_id: string;
  chunk_id: string;
  text: string;
  score: number;
  title?: string;
};

type Meta = {
  intent: { intent: string; confidence: number; requires_human: boolean };
  sources: Source[];
  latency_ms?: Record<string, number>;
};

export default function PlaygroundPage() {
  const [text, setText] = useState("");
  const [locale, setLocale] = useState<"ru" | "uz">("ru");
  const [stream, setStream] = useState(true);

  const [answer, setAnswer] = useState("");
  const [meta, setMeta] = useState<Meta | null>(null);
  const [latency, setLatency] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setAnswer("");
    setMeta(null);
    setLatency({});

    try {
      if (stream) {
        await sendStream();
      } else {
        await sendOnce();
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function sendOnce() {
    const r = await fetch(`${API_BASE}/playground/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, locale, history: [] }),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    setAnswer(data.answer);
    setMeta({ intent: data.intent, sources: data.sources, latency_ms: data.latency_ms });
    setLatency(data.latency_ms || {});
  }

  async function sendStream() {
    const t0 = performance.now();
    let firstTokenAt: number | null = null;

    const r = await fetch(`${API_BASE}/playground/chat/stream`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, locale, history: [] }),
    });
    if (!r.ok || !r.body) throw new Error(await r.text());

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      const frames = pending.split("\n\n");
      pending = frames.pop() ?? "";
      for (const frame of frames) {
        const evMatch = frame.match(/^event:\s*(\S+)/m);
        const dataMatch = frame.match(/^data:\s*(.*)$/ms);
        if (!evMatch || !dataMatch) continue;
        const event = evMatch[1];
        const raw = dataMatch[1];
        if (event === "meta") {
          try {
            setMeta(JSON.parse(raw));
          } catch {
            // ignore
          }
        } else if (event === "token") {
          if (!firstTokenAt) {
            firstTokenAt = performance.now();
            setLatency((l) => ({ ...l, ttfb_ms: firstTokenAt! - t0 }));
          }
          setAnswer((a) => a + raw);
        } else if (event === "error") {
          try {
            setError(JSON.parse(raw).message);
          } catch {
            setError(raw);
          }
        } else if (event === "done") {
          setLatency((l) => ({ ...l, total_ms: performance.now() - t0 }));
        }
      }
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-3xl font-semibold">Плейграунд</h2>
        <p className="text-mutedForeground mt-1">
          Текстовая отладка диалога: intent → RAG → ответ.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Запрос</CardTitle>
          <CardDescription>Введите вопрос так, как его задаёт клиент.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2 items-center">
            <select
              value={locale}
              onChange={(e) => setLocale(e.target.value as "ru" | "uz")}
              className="rounded-md border border-border bg-background px-3 py-2 text-sm"
            >
              <option value="ru">Русский</option>
              <option value="uz">O‘zbekcha</option>
            </select>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={stream}
                onChange={(e) => setStream(e.target.checked)}
              />
              SSE-стриминг ответа
            </label>
          </div>
          <Textarea
            placeholder="Например: какие у вас тарифы по вкладам?"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={4}
          />
          <div className="flex justify-end">
            <Button onClick={send} disabled={loading}>
              {loading ? "Отвечаю…" : "Отправить"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card>
          <CardContent className="pt-6">
            <Badge tone="danger">Ошибка: {error}</Badge>
          </CardContent>
        </Card>
      )}

      {(answer || meta) && (
        <Card>
          <CardHeader>
            <CardTitle>Ответ</CardTitle>
            {meta && (
              <CardDescription>
                Intent:{" "}
                <Badge tone={meta.intent.requires_human ? "warning" : "success"}>
                  {meta.intent.intent}
                </Badge>{" "}
                · уверенность {(meta.intent.confidence * 100).toFixed(0)}%
              </CardDescription>
            )}
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-base leading-relaxed whitespace-pre-wrap">
              {answer || (loading ? "…" : "")}
            </p>

            {Object.keys(latency).length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-2">Латентности</h4>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                  {Object.entries(latency).map(([k, v]) => (
                    <div key={k} className="rounded-md border border-border p-2">
                      <div className="text-mutedForeground">{k}</div>
                      <div className="font-medium">{v.toFixed(1)} мс</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {meta && meta.sources.length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-2">Источники из базы знаний</h4>
                <ul className="space-y-2">
                  {meta.sources.map((s) => (
                    <li
                      key={s.chunk_id}
                      className="rounded-md border border-border p-3 text-sm"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium">{s.title ?? "(без заголовка)"}</span>
                        <Badge>{s.score.toFixed(2)}</Badge>
                      </div>
                      <p className="text-mutedForeground line-clamp-3">{s.text}</p>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
