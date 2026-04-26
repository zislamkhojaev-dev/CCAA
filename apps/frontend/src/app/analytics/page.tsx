"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";

type Criterion = {
  id: string;
  set_id: string;
  code: string;
  title: string;
  description: string;
  weight: number;
  max_score: number;
  rubric: string;
};

type SetRow = { id: string; name: string; locale: string; is_default: boolean };

type Analysis = {
  transcript: string;
  diarization: { speaker: string; text: string; start_sec?: number | null; end_sec?: number | null }[];
  criterion_scores: {
    code: string;
    title: string;
    score: number;
    max_score: number;
    weight: number;
    weighted: number;
    comment: string;
  }[];
  total_weighted: number;
  notes: string;
};

export default function AnalyticsPage() {
  const [sets, setSets] = useState<SetRow[]>([]);
  const [selectedSet, setSelectedSet] = useState<string | null>(null);
  const [criteria, setCriteria] = useState<Criterion[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [newCrit, setNewCrit] = useState({
    code: "",
    title: "",
    description: "",
    weight: 1,
    max_score: 10,
    rubric: "",
  });

  async function loadSets() {
    const r = await fetch(`${API_BASE}/analytics/criteria-sets`);
    if (r.ok) {
      const data = await r.json();
      setSets(data);
      if (data.length && !selectedSet) setSelectedSet(data[0].id);
    }
  }

  async function loadCriteria(setId: string) {
    const r = await fetch(`${API_BASE}/analytics/criteria-sets/${setId}/criteria`);
    if (r.ok) setCriteria(await r.json());
  }

  useEffect(() => {
    loadSets();
  }, []);

  useEffect(() => {
    if (selectedSet) loadCriteria(selectedSet);
  }, [selectedSet]);

  async function onAnalyze(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file || !selectedSet) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("criteria_set_id", selectedSet);
      fd.append("locale", "ru");
      const r = await fetch(`${API_BASE}/analytics/analyze`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      setResult(await r.json());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
      e.target.value = "";
    }
  }

  async function addCriterion() {
    if (!selectedSet || !newCrit.code.trim() || !newCrit.title.trim()) return;
    const r = await fetch(`${API_BASE}/analytics/criteria-sets/${selectedSet}/criteria`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        code: newCrit.code.trim(),
        title: newCrit.title.trim(),
        description: newCrit.description,
        weight: newCrit.weight,
        max_score: newCrit.max_score,
        rubric: newCrit.rubric,
      }),
    });
    if (!r.ok) {
      setError(await r.text());
      return;
    }
    setNewCrit({ code: "", title: "", description: "", weight: 1, max_score: 10, rubric: "" });
    await loadCriteria(selectedSet);
  }

  async function deleteCriterion(id: string) {
    await fetch(`${API_BASE}/analytics/criteria/${id}`, { method: "DELETE" });
    if (selectedSet) await loadCriteria(selectedSet);
  }

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-3xl font-semibold">Речевая аналитика</h2>
        <p className="text-mutedForeground mt-1">
          Загрузка записи звонка: транскрибация (Whisper), разметка ролей (LLM по сегментам), оценка
          по вашим критериям с весами. Диаризация как у отдельного ASR-провайдера — следующий шаг
          (Deepgram/pyannote); сейчас — эвристика через LLM.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Набор критериев</CardTitle>
          <CardDescription>Выберите набор, который будет передан в LLM-оценщик.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <select
            className="w-full max-w-md rounded-md border border-border bg-background px-3 py-2 text-sm"
            value={selectedSet ?? ""}
            onChange={(e) => setSelectedSet(e.target.value || null)}
          >
            {sets.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
                {s.is_default ? " (по умолчанию)" : ""}
              </option>
            ))}
          </select>
          {sets.length === 0 && (
            <p className="text-sm text-mutedForeground">
              Нет наборов — выполните <code className="bg-muted px-1">python -m apps.backend.scripts.seed</code>
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Критерии (веса и шкала)</CardTitle>
          <CardDescription>
            Итоговый балл: взвешенное среднее нормализованных оценок (score/max × weight).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <ul className="space-y-2 text-sm">
            {criteria.map((c) => (
              <li
                key={c.id}
                className="flex items-start justify-between gap-2 rounded-md border border-border p-3"
              >
                <div>
                  <span className="font-medium">{c.title}</span>{" "}
                  <Badge>
                    {c.code} · w={c.weight} · max {c.max_score}
                  </Badge>
                  {c.rubric && (
                    <p className="text-xs text-mutedForeground mt-1 line-clamp-2">{c.rubric}</p>
                  )}
                </div>
                <Button variant="ghost" size="sm" onClick={() => deleteCriterion(c.id)}>
                  Удалить
                </Button>
              </li>
            ))}
          </ul>
          <div className="grid gap-2 sm:grid-cols-2 max-w-3xl">
            <input
              className="rounded-md border border-border px-3 py-2 text-sm"
              placeholder="code (латиница)"
              value={newCrit.code}
              onChange={(e) => setNewCrit({ ...newCrit, code: e.target.value })}
            />
            <input
              className="rounded-md border border-border px-3 py-2 text-sm"
              placeholder="Название"
              value={newCrit.title}
              onChange={(e) => setNewCrit({ ...newCrit, title: e.target.value })}
            />
            <input
              type="number"
              className="rounded-md border border-border px-3 py-2 text-sm"
              placeholder="Вес"
              value={newCrit.weight}
              onChange={(e) => setNewCrit({ ...newCrit, weight: Number(e.target.value) })}
            />
            <input
              type="number"
              className="rounded-md border border-border px-3 py-2 text-sm"
              placeholder="Макс. балл"
              value={newCrit.max_score}
              onChange={(e) => setNewCrit({ ...newCrit, max_score: Number(e.target.value) })}
            />
            <div className="sm:col-span-2">
              <Textarea
                placeholder="Рубрика для LLM (как оценивать)"
                rows={2}
                value={newCrit.rubric}
                onChange={(e) => setNewCrit({ ...newCrit, rubric: e.target.value })}
              />
            </div>
          </div>
          <Button onClick={addCriterion} disabled={!selectedSet}>
            Добавить критерий
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Анализ записи</CardTitle>
          <CardDescription>wav / mp3 / m4a — через OpenAI Whisper.</CardDescription>
        </CardHeader>
        <CardContent>
          <label className="inline-flex h-10 items-center justify-center rounded-md bg-primary px-4 text-sm font-medium text-primaryForeground hover:opacity-90 cursor-pointer">
            <input
              type="file"
              className="hidden"
              accept="audio/*,.wav,.mp3,.m4a,.webm"
              onChange={onAnalyze}
              disabled={loading || !selectedSet}
            />
            {loading ? "Анализ…" : "Выбрать аудиофайл"}
          </label>
          {error && (
            <p className="mt-2">
              <Badge tone="danger">{error}</Badge>
            </p>
          )}
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>Результат</CardTitle>
            <CardDescription>
              Сводный индекс (взвеш. нормализация):{" "}
              <Badge tone="success">{(result.total_weighted * 100).toFixed(1)}%</Badge> от
              идеала при текущих весах
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <h4 className="font-medium mb-1">Транскрипт</h4>
              <pre className="whitespace-pre-wrap rounded-md bg-muted p-3 text-xs max-h-48 overflow-auto">
                {result.transcript}
              </pre>
            </div>
            <div>
              <h4 className="font-medium mb-1">Диаризация (LLM)</h4>
              <ul className="space-y-1">
                {result.diarization.map((t, i) => (
                  <li key={i}>
                    <Badge>{t.speaker}</Badge> {t.text}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Оценки</h4>
              <ul className="space-y-2">
                {result.criterion_scores.map((s) => (
                  <li key={s.code} className="rounded-md border border-border p-2">
                    <div className="flex justify-between">
                      <span className="font-medium">{s.title}</span>
                      <span>
                        {s.score.toFixed(1)} / {s.max_score} (w={s.weight})
                      </span>
                    </div>
                    {s.comment && <p className="text-xs text-mutedForeground mt-1">{s.comment}</p>}
                  </li>
                ))}
              </ul>
            </div>
            {result.notes && (
              <p>
                <span className="font-medium">Резюме LLM:</span> {result.notes}
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
