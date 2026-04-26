"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Health = {
  status: string;
  app: string;
  env: string;
  providers: { stt: string; llm: string; tts: string };
  latency_budget_ms: number;
};

export default function DashboardPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then(setHealth)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-3xl font-semibold">Дашборд</h2>
        <p className="text-mutedForeground mt-1">
          Состояние системы голосового ИИ-агента и активных интеграций.
        </p>
      </header>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Бэкенд</CardTitle>
            <CardDescription>FastAPI · async-only</CardDescription>
          </CardHeader>
          <CardContent>
            {error && <Badge tone="danger">Недоступен: {error}</Badge>}
            {health && (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-mutedForeground">Окружение</span>
                  <Badge>{health.env}</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-mutedForeground">Бюджет ответа</span>
                  <span className="font-medium">{health.latency_budget_ms} мс</span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Провайдеры</CardTitle>
            <CardDescription>Strategy: легко заменить на on-prem</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {health ? (
              <>
                <ProviderRow label="STT" value={health.providers.stt} />
                <ProviderRow label="LLM" value={health.providers.llm} />
                <ProviderRow label="TTS" value={health.providers.tts} />
              </>
            ) : (
              <p className="text-mutedForeground">Загрузка…</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Метрики ФТ</CardTitle>
            <CardDescription>Жёсткие требования прототипа</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <Metric label="Задержка ответа" value="≤ 1.5 c" />
            <Metric label="Точность интента" value="≥ 80%" />
            <Metric label="Локали" value="ru, uz" />
            <Metric label="Безопасность" value="контур компании" />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function ProviderRow({ label, value }: { label: string; value: string }) {
  const tone: "warning" | "success" | "default" =
    value === "mock" ? "warning" : value === "openai" ? "default" : "success";
  return (
    <div className="flex items-center justify-between">
      <span className="text-mutedForeground">{label}</span>
      <Badge tone={tone}>{value}</Badge>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-mutedForeground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}
