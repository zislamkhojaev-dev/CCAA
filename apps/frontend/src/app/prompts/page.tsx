"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";

type Prompt = {
  id: string;
  name: string;
  locale: string;
  content: string;
  is_active: boolean;
};

export default function PromptsPage() {
  const [items, setItems] = useState<Prompt[]>([]);
  const [draft, setDraft] = useState({ name: "", locale: "ru", content: "", is_active: true });

  async function load() {
    const r = await fetch(`${API_BASE}/prompts`);
    if (r.ok) setItems(await r.json());
  }
  useEffect(() => {
    load();
  }, []);

  async function create() {
    if (!draft.name.trim() || !draft.content.trim()) return;
    await fetch(`${API_BASE}/prompts`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(draft),
    });
    setDraft({ name: "", locale: "ru", content: "", is_active: true });
    await load();
  }

  async function remove(id: string) {
    await fetch(`${API_BASE}/prompts/${id}`, { method: "DELETE" });
    await load();
  }

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-3xl font-semibold">Промпты</h2>
        <p className="text-mutedForeground mt-1">
          Версионируемые системные инструкции для LLM. Активный промпт автоматически подставляется в
          оркестратор.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Создать промпт</CardTitle>
          <CardDescription>Имя должно быть уникальным.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <input
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            placeholder="Имя (например, default-ru)"
            value={draft.name}
            onChange={(e) => setDraft({ ...draft, name: e.target.value })}
          />
          <select
            value={draft.locale}
            onChange={(e) => setDraft({ ...draft, locale: e.target.value })}
            className="rounded-md border border-border bg-background px-3 py-2 text-sm"
          >
            <option value="ru">Русский</option>
            <option value="uz">O‘zbekcha</option>
          </select>
          <Textarea
            placeholder="Текст системной инструкции…"
            rows={6}
            value={draft.content}
            onChange={(e) => setDraft({ ...draft, content: e.target.value })}
          />
          <div className="flex justify-end">
            <Button onClick={create}>Сохранить</Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Все промпты ({items.length})</CardTitle>
        </CardHeader>
        <CardContent>
          {items.length === 0 ? (
            <p className="text-sm text-mutedForeground">Пока нет ни одного промпта.</p>
          ) : (
            <ul className="space-y-3">
              {items.map((p) => (
                <li key={p.id} className="rounded-md border border-border p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-medium">{p.name}</div>
                    <div className="flex items-center gap-2">
                      <Badge>{p.locale}</Badge>
                      <Badge tone={p.is_active ? "success" : "default"}>
                        {p.is_active ? "active" : "draft"}
                      </Badge>
                      <Button variant="ghost" size="sm" onClick={() => remove(p.id)}>
                        Удалить
                      </Button>
                    </div>
                  </div>
                  <pre className="text-xs text-mutedForeground whitespace-pre-wrap">
                    {p.content}
                  </pre>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
