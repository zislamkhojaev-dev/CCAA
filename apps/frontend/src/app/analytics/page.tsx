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
  const [newSet, setNewSet] = useState({
    name: "",
    locale: "ru" as "ru" | "uz",
    is_default: false,
  });

  async function loadSets(): Promise<SetRow[]> {
    const r = await fetch(`${API_BASE}/analytics/criteria-sets`);
    if (r.ok) {
      const data = await r.json();
      setSets(data);
      if (data.length && !selectedSet) setSelectedSet(data[0].id);
      return data;
    }
    return [];
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

  async function createSet() {
    if (!newSet.name.trim()) return;
    setError(null);
    const r = await fetch(`${API_BASE}/analytics/criteria-sets`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        name: newSet.name.trim(),
        locale: newSet.locale,
        is_default: newSet.is_default,
      }),
    });
    if (!r.ok) {
      setError(await r.text());
      return;
    }
    const created = await r.json();
    setNewSet({ name: "", locale: "ru", is_default: false });
    await loadSets();
    setSelectedSet(created.id);
  }

  async function deleteSelectedSet() {
    if (!selectedSet) return;
    const current = sets.find((s) => s.id === selectedSet);
    if (!current) return;
    if (!confirm(`Удалить шаблон "${current.name}" со всеми критериями?`)) return;
    setError(null);
    const r = await fetch(`${API_BASE}/analytics/criteria-sets/${selectedSet}`, {
      method: "DELETE",
    });
    if (!r.ok) {
      setError(await r.text());
      return;
    }
    const prev = selectedSet;
    const refreshed = await loadSets();
    const next = refreshed.find((s) => s.id !== prev)?.id ?? null;
    setSelectedSet(next);
    setCriteria([]);
  }

  async function cloneSelectedSet() {
    if (!selectedSet) return;
    const currentSet = sets.find((s) => s.id === selectedSet);
    if (!currentSet) return;
    setError(null);
    const createResp = await fetch(`${API_BASE}/analytics/criteria-sets`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        name: `${currentSet.name} (копия)`,
        locale: currentSet.locale,
        is_default: false,
      }),
    });
    if (!createResp.ok) {
      setError(await createResp.text());
      return;
    }
    const cloneSet = await createResp.json();
    for (const c of criteria) {
      const r = await fetch(`${API_BASE}/analytics/criteria-sets/${cloneSet.id}/criteria`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          code: c.code,
          title: c.title,
          description: c.description,
          weight: c.weight,
          max_score: c.max_score,
          rubric: c.rubric,
        }),
      });
      if (!r.ok) {
        setError(`Шаблон создан, но копирование критерия "${c.code}" не удалось: ${await r.text()}`);
        break;
      }
    }
    await loadSets();
    setSelectedSet(cloneSet.id);
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
          <CardDescription>
            Шаблон = набор критериев для оценки звонка. Выберите активный шаблон или создайте новый.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
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
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" onClick={cloneSelectedSet} disabled={!selectedSet}>
              Клонировать шаблон
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={deleteSelectedSet}
              disabled={!selectedSet || sets.length <= 1}
            >
              Удалить шаблон
            </Button>
          </div>
          <div className="rounded-md border border-border p-3 space-y-2 max-w-3xl">
            <p className="text-sm font-medium">Создать новый шаблон</p>
            <div className="grid gap-2 sm:grid-cols-3">
              <input
                className="rounded-md border border-border px-3 py-2 text-sm sm:col-span-2"
                placeholder="Название шаблона (например, QA входящая линия)"
                value={newSet.name}
                onChange={(e) => setNewSet((v) => ({ ...v, name: e.target.value }))}
              />
              <select
                className="rounded-md border border-border bg-background px-3 py-2 text-sm"
                value={newSet.locale}
                onChange={(e) => setNewSet((v) => ({ ...v, locale: e.target.value as "ru" | "uz" }))}
              >
                <option value="ru">ru</option>
                <option value="uz">uz</option>
              </select>
            </div>
            <label className="inline-flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={newSet.is_default}
                onChange={(e) => setNewSet((v) => ({ ...v, is_default: e.target.checked }))}
              />
              Сделать шаблоном по умолчанию
            </label>
            <div>
              <Button size="sm" onClick={createSet} disabled={!newSet.name.trim()}>
                Создать шаблон
              </Button>
            </div>
          </div>
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
          <div className="rounded-md border border-border bg-muted/40 p-3 text-xs text-mutedForeground space-y-1">
            <p>
              <strong>Как настраивать поля:</strong>
            </p>
            <p>
              <strong>code</strong> — короткий уникальный ID критерия (латиница/цифры), например
              <code className="mx-1 bg-muted px-1 rounded">greeting</code>,
              <code className="mx-1 bg-muted px-1 rounded">compliance</code>.
            </p>
            <p>
              <strong>title</strong> — как критерий будет называться в отчёте.
            </p>
            <p>
              <strong>description</strong> — кратко, что именно проверяем (для команды/читабельности).
            </p>
            <p>
              <strong>weight</strong> — важность критерия в итоге. Обычно 0.5-3.0 (чем больше, тем
              сильнее влияет на общий балл).
            </p>
            <p>
              <strong>max_score</strong> — верхняя граница оценки по критерию (обычно 5 или 10).
            </p>
            <p>
              <strong>rubric</strong> — инструкция для LLM, как ставить балл (что такое 0, средний и
              максимум + примеры).
            </p>
          </div>
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
            <label className="space-y-1">
              <span className="text-xs text-mutedForeground">code (уникальный ID)</span>
            <input
              className="rounded-md border border-border px-3 py-2 text-sm"
              placeholder="Например: compliance"
              value={newCrit.code}
              onChange={(e) => setNewCrit({ ...newCrit, code: e.target.value })}
            />
            </label>
            <label className="space-y-1">
              <span className="text-xs text-mutedForeground">Название критерия</span>
            <input
              className="rounded-md border border-border px-3 py-2 text-sm"
              placeholder="Например: Соблюдение регламентов"
              value={newCrit.title}
              onChange={(e) => setNewCrit({ ...newCrit, title: e.target.value })}
            />
            </label>
            <label className="space-y-1">
              <span className="text-xs text-mutedForeground">Описание (что проверяем)</span>
              <Textarea
                placeholder="Кратко: какие сигналы в речи считаем хорошими/плохими."
                rows={2}
                value={newCrit.description}
                onChange={(e) => setNewCrit({ ...newCrit, description: e.target.value })}
              />
            </label>
            <div />
            <label className="space-y-1">
              <span className="text-xs text-mutedForeground">Вес (важность, обычно 0.5-3)</span>
            <input
              type="number"
              className="rounded-md border border-border px-3 py-2 text-sm"
              placeholder="Вес"
              value={newCrit.weight}
              onChange={(e) => setNewCrit({ ...newCrit, weight: Number(e.target.value) })}
            />
            </label>
            <label className="space-y-1">
              <span className="text-xs text-mutedForeground">Макс. балл (обычно 5 или 10)</span>
            <input
              type="number"
              className="rounded-md border border-border px-3 py-2 text-sm"
              placeholder="Макс. балл"
              value={newCrit.max_score}
              onChange={(e) => setNewCrit({ ...newCrit, max_score: Number(e.target.value) })}
            />
            </label>
            <div className="sm:col-span-2">
              <p className="text-xs text-mutedForeground mb-1">
                Rubric (как LLM должен оценивать критерий)
              </p>
              <Textarea
                placeholder="Пример: 0 — нет приветствия; 5 — есть формальное приветствие; 10 — приветствие + имя + вежливое уточнение запроса."
                rows={3}
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
