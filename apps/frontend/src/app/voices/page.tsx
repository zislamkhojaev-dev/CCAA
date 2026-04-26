"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

type Voice = {
  id: string;
  name: string;
  provider: string;
  provider_voice_id: string;
  locale: string;
  is_default: boolean;
  tts_params?: Record<string, unknown>;
};

export default function VoicesPage() {
  const [items, setItems] = useState<Voice[]>([]);
  const [editingTts, setEditingTts] = useState<Record<string, string>>({});
  const [draft, setDraft] = useState({
    name: "",
    provider: "elevenlabs",
    provider_voice_id: "",
    locale: "ru",
    is_default: false,
    tts_params: {} as Record<string, unknown>,
  });

  async function load() {
    const r = await fetch(`${API_BASE}/voices`);
    if (r.ok) {
      const list: Voice[] = await r.json();
      setItems(list);
      const next: Record<string, string> = {};
      for (const v of list) {
        next[v.id] = JSON.stringify(v.tts_params ?? {}, null, 2);
      }
      setEditingTts(next);
    }
  }
  useEffect(() => {
    load();
  }, []);

  async function create() {
    if (!draft.name.trim() || !draft.provider_voice_id.trim()) return;
    await fetch(`${API_BASE}/voices`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(draft),
    });
    setDraft({
      name: "",
      provider: "elevenlabs",
      provider_voice_id: "",
      locale: "ru",
      is_default: false,
      tts_params: {},
    });
    await load();
  }

  async function remove(id: string) {
    await fetch(`${API_BASE}/voices/${id}`, { method: "DELETE" });
    await load();
  }

  async function saveTts(id: string) {
    let tts_params: Record<string, unknown>;
    try {
      tts_params = JSON.parse(editingTts[id] || "{}") as Record<string, unknown>;
    } catch {
      return;
    }
    await fetch(`${API_BASE}/voices/${id}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ tts_params }),
    });
    await load();
  }

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-3xl font-semibold">Голоса</h2>
        <p className="text-mutedForeground mt-1">
          Управление профилями TTS. По умолчанию используется ElevenLabs (Turbo v2.5).
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Новый голос</CardTitle>
          <CardDescription>
          provider_voice_id — ID у провайдера. Для OpenAI в «Голосах» можно задать{" "}
          <code className="bg-muted px-1 rounded">tts_params</code> как{" "}
          <code className="bg-muted px-1 rounded">{`{"speed":1.1}`}</code>, для ElevenLabs —{" "}
          <code className="bg-muted px-1 rounded">{`{"stability":0.4,"similarity_boost":0.8}`}</code>
          (ниже у каждого профиля).
        </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <input
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            placeholder="Название (например, Female RU)"
            value={draft.name}
            onChange={(e) => setDraft({ ...draft, name: e.target.value })}
          />
          <input
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            placeholder="provider_voice_id"
            value={draft.provider_voice_id}
            onChange={(e) => setDraft({ ...draft, provider_voice_id: e.target.value })}
          />
          <div className="flex gap-3">
            <select
              value={draft.locale}
              onChange={(e) => setDraft({ ...draft, locale: e.target.value })}
              className="rounded-md border border-border bg-background px-3 py-2 text-sm"
            >
              <option value="ru">ru</option>
              <option value="uz">uz</option>
            </select>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={draft.is_default}
                onChange={(e) => setDraft({ ...draft, is_default: e.target.checked })}
              />
              По умолчанию
            </label>
          </div>
          <div className="flex justify-end">
            <Button onClick={create}>Сохранить</Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Профили ({items.length})</CardTitle>
        </CardHeader>
        <CardContent>
          {items.length === 0 ? (
            <p className="text-sm text-mutedForeground">Пока нет ни одного голоса.</p>
          ) : (
            <ul className="divide-y divide-border">
              {items.map((v) => (
                <li key={v.id} className="py-4 space-y-2 border-b border-border last:border-0">
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <div className="font-medium">{v.name}</div>
                      <div className="text-xs text-mutedForeground">
                        {v.provider} · {v.provider_voice_id} · {v.locale}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {v.is_default && <Badge tone="success">default</Badge>}
                      <Button variant="ghost" size="sm" onClick={() => remove(v.id)}>
                        Удалить
                      </Button>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-mutedForeground">
                      tts_params (переопределяет глобальные настройки из «Поведение бота»)
                    </p>
                    <textarea
                      className="w-full min-h-[72px] rounded-md border border-border bg-background px-2 py-1.5 text-xs font-mono"
                      value={editingTts[v.id] ?? "{}"}
                      onChange={(e) =>
                        setEditingTts((m) => ({ ...m, [v.id]: e.target.value }))
                      }
                      spellCheck={false}
                    />
                    <Button size="sm" variant="outline" onClick={() => saveTts(v.id)}>
                      Сохранить TTS-параметры
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
