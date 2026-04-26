"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const DEFAULT_HINT = `Ключи JSON (частичное сохранение объединяется с текущими):

llm_temperature, llm_max_tokens, history_max_messages,
system_prompt_suffix, suppress_repeated_greeting,
openai_tts_speed, elevenlabs_stability, elevenlabs_similarity_boost,
barge_in_rms_threshold, barge_in_cooldown_ms, barge_in_hold_frames,
stt_voice_rms_threshold, stt_min_voiced_seconds, stt_duplicate_cooldown_sec`;

type BotSettings = {
  llm_temperature?: number;
  llm_max_tokens?: number;
  history_max_messages?: number;
  system_prompt_suffix?: string;
  suppress_repeated_greeting?: boolean;
  openai_tts_speed?: number;
  elevenlabs_stability?: number;
  elevenlabs_similarity_boost?: number;
  barge_in_rms_threshold?: number;
  barge_in_cooldown_ms?: number;
  barge_in_hold_frames?: number;
  stt_voice_rms_threshold?: number;
  stt_min_voiced_seconds?: number;
  stt_duplicate_cooldown_sec?: number;
};

export default function BotSettingsPage() {
  const [form, setForm] = useState<BotSettings>({});
  const [raw, setRaw] = useState("{}");
  const [msg, setMsg] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  async function load() {
    setMsg(null);
    const r = await fetch(`${API_BASE}/bot-settings`);
    if (!r.ok) {
      setMsg("Не удалось загрузить настройки");
      return;
    }
    const data = await r.json();
    setForm(data);
    setRaw(JSON.stringify(data, null, 2));
  }

  useEffect(() => {
    load();
  }, []);

  async function saveForm() {
    setMsg(null);
    setSaving(true);
    const r = await fetch(`${API_BASE}/bot-settings`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(form),
    });
    setSaving(false);
    if (!r.ok) {
      setMsg("Ошибка сохранения");
      return;
    }
    const data = await r.json();
    setForm(data);
    setRaw(JSON.stringify(data, null, 2));
    setMsg("Сохранено");
  }

  async function save() {
    setMsg(null);
    setSaving(true);
    let body: Record<string, unknown>;
    try {
      body = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      setSaving(false);
      setMsg("Некорректный JSON");
      return;
    }
    const r = await fetch(`${API_BASE}/bot-settings`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    setSaving(false);
    if (!r.ok) {
      setMsg("Ошибка сохранения");
      return;
    }
    const data = await r.json();
    setForm(data);
    setRaw(JSON.stringify(data, null, 2));
    setMsg("Сохранено");
  }

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-3xl font-semibold">Поведение бота</h2>
        <p className="text-mutedForeground mt-1">
          Температура LLM, длина истории, правила диалога, TTS, а также чувствительность
          barge-in и пороги шумоподавления STT (настройки голосового канала).
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Быстрая настройка (UI)</CardTitle>
          <CardDescription>
            Рекомендуется для тюнинга вживую: барж-ин и шумоподавление STT.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid md:grid-cols-2 gap-4">
            <LabeledNumber
              label="Barge-in RMS threshold"
              hint="Чем выше, тем реже бот прерывается от фонового шума."
              value={form.barge_in_rms_threshold ?? 0.12}
              step={0.005}
              min={0.01}
              max={0.5}
              onChange={(v) => setForm((f) => ({ ...f, barge_in_rms_threshold: v }))}
            />
            <LabeledNumber
              label="Barge-in cooldown (ms)"
              hint="Минимальный интервал между прерываниями."
              value={form.barge_in_cooldown_ms ?? 700}
              step={50}
              min={100}
              max={4000}
              onChange={(v) => setForm((f) => ({ ...f, barge_in_cooldown_ms: v }))}
            />
            <LabeledNumber
              label="Barge-in hold frames"
              hint="Сколько подряд громких фреймов нужно для interrupt."
              value={form.barge_in_hold_frames ?? 3}
              step={1}
              min={1}
              max={20}
              onChange={(v) => setForm((f) => ({ ...f, barge_in_hold_frames: Math.round(v) }))}
            />
            <LabeledNumber
              label="STT voice RMS threshold"
              hint="Порог отсечки тишины/шума перед отправкой в STT."
              value={form.stt_voice_rms_threshold ?? 0.008}
              step={0.001}
              min={0.001}
              max={0.1}
              onChange={(v) => setForm((f) => ({ ...f, stt_voice_rms_threshold: v }))}
            />
            <LabeledNumber
              label="STT min voiced seconds"
              hint="Минимальная длительность речи, чтобы флашить буфер."
              value={form.stt_min_voiced_seconds ?? 0.2}
              step={0.05}
              min={0.05}
              max={3}
              onChange={(v) => setForm((f) => ({ ...f, stt_min_voiced_seconds: v }))}
            />
            <LabeledNumber
              label="STT duplicate cooldown (sec)"
              hint="Подавление одинаковых распознанных реплик."
              value={form.stt_duplicate_cooldown_sec ?? 4}
              step={0.5}
              min={0}
              max={30}
              onChange={(v) => setForm((f) => ({ ...f, stt_duplicate_cooldown_sec: v }))}
            />
          </div>
          <div className="flex gap-2 justify-end">
            <Button variant="ghost" onClick={load} disabled={saving}>
              Сбросить с сервера
            </Button>
            <Button onClick={saveForm} disabled={saving}>
              {saving ? "Сохраняю…" : "Сохранить UI-настройки"}
            </Button>
          </div>
        </CardContent>
      </Card>

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
            <Button variant="ghost" onClick={load} disabled={saving}>
              Сбросить с сервера
            </Button>
            <Button onClick={save} disabled={saving}>
              {saving ? "Сохраняю…" : "Сохранить JSON"}
            </Button>
          </div>
          {msg && <p className="text-sm text-mutedForeground">{msg}</p>}
        </CardContent>
      </Card>
    </div>
  );
}

function LabeledNumber({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="space-y-1">
      <div className="text-sm font-medium">{label}</div>
      <div className="text-xs text-mutedForeground">{hint}</div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
      />
    </label>
  );
}
