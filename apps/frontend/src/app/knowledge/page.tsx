"use client";

import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

type Pool = "voice" | "agent_assist";

type Doc = {
  id: string;
  title: string;
  locale: string;
  knowledge_pool: Pool;
  chunk_count: number;
  created_at: string;
};

export default function KnowledgePage() {
  const [pool, setPool] = useState<Pool>("voice");
  const [docs, setDocs] = useState<Doc[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setError(null);
      const r = await fetch(`${API_BASE}/documents?pool=${pool}`);
      if (!r.ok) {
        throw new Error(await r.text());
      }
      setDocs(await r.json());
    } catch (e) {
      setError(String(e));
    }
  }, [pool]);

  useEffect(() => {
    load();
  }, [load]);

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("title", file.name);
      fd.append("locale", "ru");
      fd.append("knowledge_pool", pool);
      const r = await fetch(`${API_BASE}/documents`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      await load();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function onDelete(id: string) {
    if (!confirm("Удалить документ из выбранного пула?")) return;
    setError(null);
    const r = await fetch(`${API_BASE}/documents/${id}`, { method: "DELETE" });
    if (!r.ok) {
      setError(await r.text());
      return;
    }
    await load();
  }

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-3xl font-semibold">База знаний</h2>
        <p className="text-mutedForeground mt-1">
          Два независимых пула в Qdrant: <strong>voice</strong> — голосовой бот и плейграунд;
          <strong> agent_assist</strong> — суфлёр оператора (может быть короче и по другим
          правилам).
        </p>
      </header>

      <div className="flex gap-2 border-b border-border pb-2">
        <Button
          variant={pool === "voice" ? "default" : "outline"}
          size="sm"
          onClick={() => setPool("voice")}
        >
          Voice (бот)
        </Button>
        <Button
          variant={pool === "agent_assist" ? "default" : "outline"}
          size="sm"
          onClick={() => setPool("agent_assist")}
        >
          Agent Assist (суфлёр)
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Загрузка в пул «{pool === "voice" ? "voice" : "agent_assist"}»</CardTitle>
          <CardDescription>Поддерживаются .pdf, .docx, .txt, .md</CardDescription>
        </CardHeader>
        <CardContent>
          <label className="inline-flex h-10 items-center justify-center rounded-md bg-primary px-4 text-sm font-medium text-primaryForeground hover:opacity-90 cursor-pointer">
            <input
              type="file"
              className="hidden"
              accept=".pdf,.docx,.txt,.md"
              onChange={onUpload}
              disabled={uploading}
            />
            {uploading ? "Загружаю…" : "Выбрать файл"}
          </label>
          {error && (
            <p className="mt-2">
              <Badge tone="danger">Ошибка: {error}</Badge>
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Документы ({docs.length})</CardTitle>
        </CardHeader>
        <CardContent>
          {docs.length === 0 ? (
            <p className="text-mutedForeground text-sm">В этом пуле пока нет документов.</p>
          ) : (
            <ul className="divide-y divide-border">
              {docs.map((d) => (
                <li key={d.id} className="py-3 flex items-center justify-between gap-4">
                  <div>
                    <div className="font-medium">{d.title}</div>
                    <div className="text-xs text-mutedForeground">
                      {d.chunk_count} чанков · {d.locale} ·{" "}
                      {new Date(d.created_at).toLocaleString()}
                    </div>
                  </div>
                  <Button variant="ghost" size="sm" onClick={() => onDelete(d.id)}>
                    Удалить
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
