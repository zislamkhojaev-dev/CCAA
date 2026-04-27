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
vad_silero_speech_threshold, vad_webrtc_aggressiveness,
stt_voice_rms_threshold, stt_min_voiced_seconds, stt_duplicate_cooldown_sec, stt_final_debounce_ms,
silence_nudge_enabled, silence_timeout_sec, silence_nudge_cooldown_sec,
semantic_cache_ttl_sec, semantic_cache_max_entries, agent_tool_loop_max_steps,
native_function_calling_enabled, rag_hybrid_enabled, rag_hybrid_alpha, rag_candidate_pool_size,
semantic_router_fast_enabled, semantic_router_embed_enabled,
router_embed_escalation_cos, router_embed_smalltalk_cos, router_embed_knowledge_cos, router_embed_noise_max_cos,
semantic_intent_embed_enabled, semantic_intent_min_cos, semantic_intent_op_min_cos,
low_signal_reply_enabled, tts_micro_batch_chars, fallback_cooldown_sec,
router_noise_max_words, router_noise_max_chars, router_simple_max_words,
router_simple_min_confidence, router_escalation_min_confidence,
router_policy_noise_run_intent, router_policy_noise_run_rag,
router_policy_simple_run_intent, router_policy_simple_run_rag,
router_policy_complex_run_intent, router_policy_complex_run_rag,
case_clarification_max_attempts, case_fallback_handoff_limit,
case_low_confidence_threshold, case_confirm_intent_min_confidence,
case_use_dynamic_phrasing, case_strict_template_intents, case_strict_template_slots,
case_intent_confirmation_templates, case_slot_question_templates`;

type LocaleTemplate = { ru?: string; uz?: string };
type TemplateMap = Record<string, LocaleTemplate>;
type TemplateRow = { key: string; ru: string; uz: string };

const INTENT_TEMPLATE_KEYS = ["default", "block_card", "transfer_money", "complaint", "products", "tariffs"];
const SLOT_TEMPLATE_KEYS = [
  "default",
  "card_last4",
  "incident_time",
  "channel",
  "source_account",
  "target_account",
  "amount",
  "customer_id",
  "field_to_change",
  "reason",
  "topic",
  "details",
  "product_name",
];

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
  vad_silero_speech_threshold?: number;
  vad_webrtc_aggressiveness?: number;
  stt_voice_rms_threshold?: number;
  stt_min_voiced_seconds?: number;
  stt_duplicate_cooldown_sec?: number;
  stt_final_debounce_ms?: number;
  rag_hybrid_enabled?: boolean;
  rag_hybrid_alpha?: number;
  rag_candidate_pool_size?: number;
  semantic_router_fast_enabled?: boolean;
  semantic_router_embed_enabled?: boolean;
  router_embed_escalation_cos?: number;
  router_embed_smalltalk_cos?: number;
  router_embed_knowledge_cos?: number;
  router_embed_noise_max_cos?: number;
  semantic_intent_embed_enabled?: boolean;
  semantic_intent_min_cos?: number;
  semantic_intent_op_min_cos?: number;
  low_signal_reply_enabled?: boolean;
  tts_micro_batch_chars?: number;
  fallback_cooldown_sec?: number;
  case_clarification_max_attempts?: number;
  case_fallback_handoff_limit?: number;
  case_low_confidence_threshold?: number;
  case_confirm_intent_min_confidence?: number;
  case_use_dynamic_phrasing?: boolean;
  case_strict_template_intents?: string[];
  case_strict_template_slots?: string[];
  case_intent_confirmation_templates?: TemplateMap;
  case_slot_question_templates?: TemplateMap;
};

export default function BotSettingsPage() {
  const [form, setForm] = useState<BotSettings>({});
  const [raw, setRaw] = useState("{}");
  const [intentTemplatesRaw, setIntentTemplatesRaw] = useState("{}");
  const [slotTemplatesRaw, setSlotTemplatesRaw] = useState("{}");
  const [intentTemplateRows, setIntentTemplateRows] = useState<TemplateRow[]>([]);
  const [slotTemplateRows, setSlotTemplateRows] = useState<TemplateRow[]>([]);
  const [strictIntentsRaw, setStrictIntentsRaw] = useState("");
  const [strictSlotsRaw, setStrictSlotsRaw] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const intentDefaultMissing = !intentTemplateRows.find((r) => r.key === "default")?.ru?.trim();
  const slotDefaultMissing = !slotTemplateRows.find((r) => r.key === "default")?.ru?.trim();

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
    const intentMap = normalizeTemplateMap(data.case_intent_confirmation_templates ?? {}, INTENT_TEMPLATE_KEYS);
    const slotMap = normalizeTemplateMap(data.case_slot_question_templates ?? {}, SLOT_TEMPLATE_KEYS);
    setIntentTemplatesRaw(JSON.stringify(intentMap, null, 2));
    setSlotTemplatesRaw(JSON.stringify(slotMap, null, 2));
    setIntentTemplateRows(mapToRows(intentMap, INTENT_TEMPLATE_KEYS));
    setSlotTemplateRows(mapToRows(slotMap, SLOT_TEMPLATE_KEYS));
    setStrictIntentsRaw(((data.case_strict_template_intents ?? []) as string[]).join(", "));
    setStrictSlotsRaw(((data.case_strict_template_slots ?? []) as string[]).join(", "));
  }

  useEffect(() => {
    load();
  }, []);

  async function saveForm() {
    setMsg(null);
    if (intentDefaultMissing || slotDefaultMissing) {
      setMsg("Заполните обязательные default.ru для intent/slot шаблонов");
      return;
    }
    let intentTemplates: TemplateMap;
    let slotTemplates: TemplateMap;
    const strictIntents = splitCsv(strictIntentsRaw);
    const strictSlots = splitCsv(strictSlotsRaw);
    try {
      intentTemplates = JSON.parse(intentTemplatesRaw) as TemplateMap;
      slotTemplates = JSON.parse(slotTemplatesRaw) as TemplateMap;
    } catch {
      setMsg("Некорректный JSON в шаблонах case-manager");
      return;
    }
    setSaving(true);
    const r = await fetch(`${API_BASE}/bot-settings`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        ...form,
        case_strict_template_intents: strictIntents,
        case_strict_template_slots: strictSlots,
        case_intent_confirmation_templates: intentTemplates,
        case_slot_question_templates: slotTemplates,
      }),
    });
    setSaving(false);
    if (!r.ok) {
      setMsg("Ошибка сохранения");
      return;
    }
    const data = await r.json();
    setForm(data);
    setRaw(JSON.stringify(data, null, 2));
    const intentMap = normalizeTemplateMap(data.case_intent_confirmation_templates ?? {}, INTENT_TEMPLATE_KEYS);
    const slotMap = normalizeTemplateMap(data.case_slot_question_templates ?? {}, SLOT_TEMPLATE_KEYS);
    setIntentTemplatesRaw(JSON.stringify(intentMap, null, 2));
    setSlotTemplatesRaw(JSON.stringify(slotMap, null, 2));
    setIntentTemplateRows(mapToRows(intentMap, INTENT_TEMPLATE_KEYS));
    setSlotTemplateRows(mapToRows(slotMap, SLOT_TEMPLATE_KEYS));
    setStrictIntentsRaw(((data.case_strict_template_intents ?? []) as string[]).join(", "));
    setStrictSlotsRaw(((data.case_strict_template_slots ?? []) as string[]).join(", "));
    setMsg("Сохранено");
  }

  function updateIntentRow(key: string, locale: "ru" | "uz", value: string) {
    const next = intentTemplateRows.map((row) =>
      row.key === key ? { ...row, [locale]: value } : row,
    );
    setIntentTemplateRows(next);
    setIntentTemplatesRaw(JSON.stringify(rowsToMap(next), null, 2));
  }

  function updateSlotRow(key: string, locale: "ru" | "uz", value: string) {
    const next = slotTemplateRows.map((row) =>
      row.key === key ? { ...row, [locale]: value } : row,
    );
    setSlotTemplateRows(next);
    setSlotTemplatesRaw(JSON.stringify(rowsToMap(next), null, 2));
  }

  function addIntentRow(key: string) {
    const clean = key.trim();
    if (!clean || intentTemplateRows.some((r) => r.key === clean)) return;
    const next = [...intentTemplateRows, { key: clean, ru: "", uz: "" }];
    setIntentTemplateRows(next);
    setIntentTemplatesRaw(JSON.stringify(rowsToMap(next), null, 2));
  }

  function removeIntentRow(key: string) {
    if (key === "default") return;
    const next = intentTemplateRows.filter((r) => r.key !== key);
    setIntentTemplateRows(next);
    setIntentTemplatesRaw(JSON.stringify(rowsToMap(next), null, 2));
  }

  function addSlotRow(key: string) {
    const clean = key.trim();
    if (!clean || slotTemplateRows.some((r) => r.key === clean)) return;
    const next = [...slotTemplateRows, { key: clean, ru: "", uz: "" }];
    setSlotTemplateRows(next);
    setSlotTemplatesRaw(JSON.stringify(rowsToMap(next), null, 2));
  }

  function removeSlotRow(key: string) {
    if (key === "default") return;
    const next = slotTemplateRows.filter((r) => r.key !== key);
    setSlotTemplateRows(next);
    setSlotTemplatesRaw(JSON.stringify(rowsToMap(next), null, 2));
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
          Температура LLM, длина истории, правила диалога, TTS, barge-in, VAD перед STT (Silero ONNX /
          WebRTC — переключение бэкенда задаётся переменной окружения{" "}
          <code className="text-xs bg-muted px-1 rounded">VOICE_VAD_BACKEND</code>
          : silero_onnx | webrtc), локальный семантический роутинг (MiniLM) и прочие параметры голосового
          канала.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Быстрая настройка (UI)</CardTitle>
          <CardDescription>
            Тюнинг вживую: barge-in, VAD (порог Silero / агрессивность WebRTC), дебаунс STT и семантический
            роутер (эмбеддинги без LLM).
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
              label="VAD Silero: порог вероятности речи"
              hint="Для бэкенда silero_onnx: 0..1, ниже — меньше ложных срабатываний, выше — чувствительнее к тихой речи."
              value={form.vad_silero_speech_threshold ?? 0.45}
              step={0.02}
              min={0.1}
              max={0.95}
              onChange={(v) => setForm((f) => ({ ...f, vad_silero_speech_threshold: v }))}
            />
            <LabeledNumber
              label="VAD WebRTC: агрессивность"
              hint="Для бэкенда webrtc: 0 (мягко) … 3 (жёстче отсекает не-речь)."
              value={form.vad_webrtc_aggressiveness ?? 2}
              step={1}
              min={0}
              max={3}
              onChange={(v) => setForm((f) => ({ ...f, vad_webrtc_aggressiveness: Math.round(v) }))}
            />
            <LabeledNumber
              label="STT legacy RMS threshold"
              hint="Используется в отдельных STT-путях с RMS-гейтом; основной голосовой VAD — Silero/WebRTC."
              value={form.stt_voice_rms_threshold ?? 0.008}
              step={0.001}
              min={0.001}
              max={0.1}
              onChange={(v) => setForm((f) => ({ ...f, stt_voice_rms_threshold: v }))}
            />
            <LabeledNumber
              label="STT min voiced seconds"
              hint="Минимальная длительность речи, чтобы открыть гейт перед STT (VAD)."
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
            <LabeledNumber
              label="STT final debounce (ms)"
              hint="Пауза после финального фрагмента STT перед стартом ответа; 0 — без ожидания."
              value={form.stt_final_debounce_ms ?? 450}
              step={50}
              min={0}
              max={2000}
              onChange={(v) => setForm((f) => ({ ...f, stt_final_debounce_ms: Math.round(v) }))}
            />
            <LabeledToggle
              label="Semantic router fast path"
              hint="Быстрый роутинг приветствий/коротких реплик без тяжелого пайплайна."
              checked={form.semantic_router_fast_enabled ?? true}
              onChange={(v) => setForm((f) => ({ ...f, semantic_router_fast_enabled: v }))}
            />
            <LabeledToggle
              label="Semantic router (локальные эмбеддинги)"
              hint="Маршрут noise/simple/complex и эскалация по косинусу к эталонным фразам (MiniLM), без ключевых слов."
              checked={form.semantic_router_embed_enabled ?? true}
              onChange={(v) => setForm((f) => ({ ...f, semantic_router_embed_enabled: v }))}
            />
            <LabeledToggle
              label="Semantic intent (локальные эмбеддинги)"
              hint="Быстрый выбор интента по MiniLM до ключевых слов и LLM."
              checked={form.semantic_intent_embed_enabled ?? true}
              onChange={(v) => setForm((f) => ({ ...f, semantic_intent_embed_enabled: v }))}
            />
            <LabeledNumber
              label="Router embed: порог эскалации (cos)"
              hint="Сходство с кластером «оператор / человек»."
              value={form.router_embed_escalation_cos ?? 0.42}
              step={0.02}
              min={0.15}
              max={0.95}
              onChange={(v) => setForm((f) => ({ ...f, router_embed_escalation_cos: v }))}
            />
            <LabeledNumber
              label="Router embed: порог smalltalk (cos)"
              hint="Приветствие / спасибо / прощание."
              value={form.router_embed_smalltalk_cos ?? 0.4}
              step={0.02}
              min={0.15}
              max={0.95}
              onChange={(v) => setForm((f) => ({ ...f, router_embed_smalltalk_cos: v }))}
            />
            <LabeledNumber
              label="Router embed: порог «справка» (cos)"
              hint="Короткий консультативный запрос к базе знаний."
              value={form.router_embed_knowledge_cos ?? 0.36}
              step={0.02}
              min={0.15}
              max={0.95}
              onChange={(v) => setForm((f) => ({ ...f, router_embed_knowledge_cos: v }))}
            />
            <LabeledNumber
              label="Router embed: шум max (cos)"
              hint="Если макс. сходство ниже порога на очень короткой фразе — класс noise."
              value={form.router_embed_noise_max_cos ?? 0.34}
              step={0.02}
              min={0.1}
              max={0.6}
              onChange={(v) => setForm((f) => ({ ...f, router_embed_noise_max_cos: v }))}
            />
            <LabeledNumber
              label="Intent embed: min cos"
              hint="Минимальная уверенность, чтобы принять интент из локальных эмбеддингов."
              value={form.semantic_intent_min_cos ?? 0.38}
              step={0.02}
              min={0.2}
              max={0.9}
              onChange={(v) => setForm((f) => ({ ...f, semantic_intent_min_cos: v }))}
            />
            <LabeledNumber
              label="Intent embed: min cos (операционные)"
              hint="Порог для block_card / перевод / персональные данные / жалоба."
              value={form.semantic_intent_op_min_cos ?? 0.4}
              step={0.02}
              min={0.2}
              max={0.9}
              onChange={(v) => setForm((f) => ({ ...f, semantic_intent_op_min_cos: v }))}
            />
            <LabeledToggle
              label="Low-signal reply"
              hint="Короткие нейтральные ответы вместо fallback про KB на 'подожди/угу'."
              checked={form.low_signal_reply_enabled ?? true}
              onChange={(v) => setForm((f) => ({ ...f, low_signal_reply_enabled: v }))}
            />
            <LabeledToggle
              label="RAG hybrid enabled"
              hint="Гибридный поиск: векторный + лексический (BM25-like) rerank."
              checked={form.rag_hybrid_enabled ?? true}
              onChange={(v) => setForm((f) => ({ ...f, rag_hybrid_enabled: v }))}
            />
            <LabeledNumber
              label="RAG hybrid alpha"
              hint="Вес векторной части (0..1). 0.65 — баланс."
              value={form.rag_hybrid_alpha ?? 0.65}
              step={0.05}
              min={0}
              max={1}
              onChange={(v) => setForm((f) => ({ ...f, rag_hybrid_alpha: v }))}
            />
            <LabeledNumber
              label="RAG candidate pool size"
              hint="Сколько кандидатов брать до fusion rerank."
              value={form.rag_candidate_pool_size ?? 30}
              step={1}
              min={5}
              max={100}
              onChange={(v) => setForm((f) => ({ ...f, rag_candidate_pool_size: Math.round(v) }))}
            />
            <LabeledNumber
              label="TTS micro-batch chars"
              hint="Размер текстового микробатча для более раннего старта синтеза."
              value={form.tts_micro_batch_chars ?? 40}
              step={1}
              min={20}
              max={120}
              onChange={(v) => setForm((f) => ({ ...f, tts_micro_batch_chars: Math.round(v) }))}
            />
            <LabeledNumber
              label="Fallback cooldown (sec)"
              hint="Антиспам интервал между fallback-репликами."
              value={form.fallback_cooldown_sec ?? 2.5}
              step={0.1}
              min={0}
              max={10}
              onChange={(v) => setForm((f) => ({ ...f, fallback_cooldown_sec: v }))}
            />
            <LabeledNumber
              label="Case: max clarification attempts"
              hint="Сколько уточняющих вопросов задать до перевода на оператора."
              value={form.case_clarification_max_attempts ?? 2}
              step={1}
              min={1}
              max={8}
              onChange={(v) => setForm((f) => ({ ...f, case_clarification_max_attempts: Math.round(v) }))}
            />
            <LabeledNumber
              label="Case: fallback handoff limit"
              hint="После скольких fallback-ответов эскалировать на оператора."
              value={form.case_fallback_handoff_limit ?? 2}
              step={1}
              min={1}
              max={8}
              onChange={(v) => setForm((f) => ({ ...f, case_fallback_handoff_limit: Math.round(v) }))}
            />
            <LabeledNumber
              label="Case: low confidence threshold"
              hint="Если confidence ниже порога, кейс быстрее уходит в handoff."
              value={form.case_low_confidence_threshold ?? 0.45}
              step={0.05}
              min={0}
              max={1}
              onChange={(v) => setForm((f) => ({ ...f, case_low_confidence_threshold: v }))}
            />
            <LabeledNumber
              label="Case: confirm intent min confidence"
              hint="Ниже порога бот сначала подтверждает интент у клиента."
              value={form.case_confirm_intent_min_confidence ?? 0.72}
              step={0.05}
              min={0}
              max={1}
              onChange={(v) => setForm((f) => ({ ...f, case_confirm_intent_min_confidence: v }))}
            />
            <LabeledToggle
              label="Case: dynamic phrasing (LLM)"
              hint="Если включено, формулировки подтверждения/уточнения генерируются LLM."
              checked={form.case_use_dynamic_phrasing ?? true}
              onChange={(v) => setForm((f) => ({ ...f, case_use_dynamic_phrasing: v }))}
            />
            <LabeledText
              label="Case: strict template intents"
              hint="Список intent через запятую, где всегда использовать шаблоны."
              value={strictIntentsRaw}
              onChange={setStrictIntentsRaw}
              placeholder="block_card, transfer_money, personal_data"
            />
            <LabeledText
              label="Case: strict template slots"
              hint="Список slot через запятую, где всегда использовать шаблоны."
              value={strictSlotsRaw}
              onChange={setStrictSlotsRaw}
              placeholder="card_last4, customer_id, source_account, target_account"
            />
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            <LabeledJson
              label="Шаблоны подтверждения интента"
              hint="JSON map: intent -> { ru, uz }. Обязателен ключ default."
              value={intentTemplatesRaw}
              onChange={setIntentTemplatesRaw}
            />
            <LabeledJson
              label="Шаблоны вопросов по слотам"
              hint="JSON map: slot -> { ru, uz }. Обязателен ключ default."
              value={slotTemplatesRaw}
              onChange={setSlotTemplatesRaw}
            />
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            <TemplateTable
              title="Визуальный редактор: intent templates"
              description="Редактируй формулировки подтверждения интента в ru/uz."
              kind="intent"
              rows={intentTemplateRows}
              onChange={updateIntentRow}
              onAdd={addIntentRow}
              onRemove={removeIntentRow}
            />
            <TemplateTable
              title="Визуальный редактор: slot templates"
              description="Редактируй уточняющие вопросы по слотам в ru/uz."
              kind="slot"
              rows={slotTemplateRows}
              onChange={updateSlotRow}
              onAdd={addSlotRow}
              onRemove={removeSlotRow}
            />
          </div>
          {(intentDefaultMissing || slotDefaultMissing) && (
            <p className="text-sm text-red-500">
              Для корректной работы case-manager обязательно заполнить `default.ru` в обоих наборах
              шаблонов.
            </p>
          )}
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

function LabeledToggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="space-y-2">
      <div className="text-sm font-medium">{label}</div>
      <div className="text-xs text-mutedForeground">{hint}</div>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="h-4 w-4"
      />
    </label>
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

function LabeledJson({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="space-y-1">
      <div className="text-sm font-medium">{label}</div>
      <div className="text-xs text-mutedForeground">{hint}</div>
      <textarea
        className="w-full min-h-[220px] rounded-md border border-border bg-background px-3 py-2 text-sm font-mono"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
      />
    </label>
  );
}

function LabeledText({
  label,
  hint,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  hint: string;
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
}) {
  return (
    <label className="space-y-1">
      <div className="text-sm font-medium">{label}</div>
      <div className="text-xs text-mutedForeground">{hint}</div>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
      />
    </label>
  );
}

function TemplateTable({
  title,
  description,
  kind,
  rows,
  onChange,
  onAdd,
  onRemove,
}: {
  title: string;
  description: string;
  kind: "intent" | "slot";
  rows: TemplateRow[];
  onChange: (key: string, locale: "ru" | "uz", value: string) => void;
  onAdd: (key: string) => void;
  onRemove: (key: string) => void;
}) {
  const [newKey, setNewKey] = useState("");

  return (
    <div className="space-y-2">
      <div className="text-sm font-medium">{title}</div>
      <div className="text-xs text-mutedForeground">{description}</div>
      <div className="flex gap-2">
        <input
          type="text"
          value={newKey}
          onChange={(e) => setNewKey(e.target.value)}
          placeholder="Новый ключ (например: refund_issue)"
          className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs"
        />
        <Button
          variant="outline"
          onClick={() => {
            onAdd(newKey);
            setNewKey("");
          }}
        >
          + Ключ
        </Button>
      </div>
      <div className="space-y-2 rounded-md border border-border p-3 max-h-[360px] overflow-y-auto">
        {rows.map((row) => (
          <div key={row.key} className="space-y-1 rounded-md border border-border/60 p-2">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold">{row.key}</div>
              <Button
                variant="ghost"
                onClick={() => onRemove(row.key)}
                disabled={row.key === "default"}
              >
                Удалить
              </Button>
            </div>
            <input
              type="text"
              value={row.ru}
              onChange={(e) => onChange(row.key, "ru", e.target.value)}
              placeholder="ru шаблон"
              className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs"
            />
            <input
              type="text"
              value={row.uz}
              onChange={(e) => onChange(row.key, "uz", e.target.value)}
              placeholder="uz шаблон"
              className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs"
            />
            <div className="text-[11px] text-mutedForeground">
              preview: {renderTemplatePreview(kind, row)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function normalizeTemplateMap(raw: unknown, keys: string[]): TemplateMap {
  const out: TemplateMap = {};
  for (const key of keys) {
    out[key] = { ru: "", uz: "" };
  }
  if (raw && typeof raw === "object") {
    for (const [key, val] of Object.entries(raw as Record<string, unknown>)) {
      const row = typeof val === "object" && val ? (val as Record<string, unknown>) : {};
      out[key] = {
        ru: typeof row.ru === "string" ? row.ru : "",
        uz: typeof row.uz === "string" ? row.uz : "",
      };
    }
  }
  return out;
}

function mapToRows(map: TemplateMap, keys: string[]): TemplateRow[] {
  const seen = new Set<string>();
  const rows: TemplateRow[] = [];
  for (const key of keys) {
    const row = map[key] ?? {};
    rows.push({ key, ru: row.ru ?? "", uz: row.uz ?? "" });
    seen.add(key);
  }
  for (const [key, row] of Object.entries(map)) {
    if (seen.has(key)) continue;
    rows.push({ key, ru: row.ru ?? "", uz: row.uz ?? "" });
  }
  return rows;
}

function rowsToMap(rows: TemplateRow[]): TemplateMap {
  const out: TemplateMap = {};
  for (const row of rows) {
    out[row.key] = { ru: row.ru, uz: row.uz };
  }
  return out;
}

function renderTemplatePreview(kind: "intent" | "slot", row: TemplateRow): string {
  const base = row.ru || row.uz || "";
  if (!base) return "—";
  if (kind === "intent") {
    return base.replaceAll("{intent}", "products");
  }
  return base;
}

function splitCsv(raw: string): string[] {
  return raw
    .split(",")
    .map((v) => v.trim())
    .filter((v) => Boolean(v));
}
