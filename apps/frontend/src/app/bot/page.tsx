"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const DEFAULT_HINT = `Ключи JSON (частичное сохранение объединяется с текущими):

llm_temperature, llm_max_tokens, history_max_messages,
system_prompt_suffix, suppress_repeated_greeting,
openai_tts_speed, elevenlabs_stability, elevenlabs_similarity_boost`;

export default function BotSettingsPage() {
  const [raw, setRaw] = useState("{}");
  const [msg, setMsg] = useState<string | null>(null);

  async function load() {
    setMsg(null);
    const r = await fetch(`${API_BASE}/bot-settings`);
    if (!r.ok) {
      setMsg("Не удалось загрузить настройки");
      return;
    }
    const data = await r.json();
    setRaw(JSON.stringify(data, null, 2));
  }

  useEffect(() => {
    load();
  }, []);

  async function save() {
    setMsg(null);
    let body: Record<string, unknown>;
    try {
      body = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      setMsg("Некорректный JSON");
      return;
    }
    const r = await fetch(`${API_BASE}/bot-settings`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      setMsg("Ошибка сохранения");
      return;
    }
    const data = await r.json();
    setRaw(JSON.stringify(data, null, 2));
    setMsg("Сохранено");
  }

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-3xl font-semibold">Поведение бота</h2>
        <p className="text-mutedForeground mt-1">
          Температура LLM, длина истории, правило без повторных приветствий, суффикс системного
          промпта и значения по умолчанию для TTS (скорость OpenAI, stability ElevenLabs).
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Параметры (JSON)</CardTitle>
          <CardDescription className="whitespace-pre-wrap">{DEFAULT_HINT}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <textarea
            className="w-full min-h-[320px] rounded-md border border-border bg-background px-3 py-2 text-sm font-mono"
            value={raw}
            onChange={(e) => setRaw(e.target.value)}
            spellCheck={false}
          />
          <div className="flex gap-2 justify-end">
            <Button variant="ghost" onClick={load}>
              Сбросить с сервера
            </Button>
            <Button onClick={save}>Сохранить</Button>
          </div>
          {msg && <p className="text-sm text-mutedForeground">{msg}</p>}
        </CardContent>
      </Card>
    </div>
  );
}
