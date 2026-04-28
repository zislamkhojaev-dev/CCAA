"use client";

import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { API_BASE } from "@/lib/utils";
import { backendWsUrl } from "@/lib/ws";

const TARGET_SAMPLE_RATE = 16_000;
const DEFAULT_BARGE_IN_RMS = 0.12;
const DEFAULT_BARGE_IN_COOLDOWN_MS = 700;
const DEFAULT_BARGE_IN_HOLD_FRAMES = 3;

const WORKLET_SRC = `
class DownsamplerWorklet extends AudioWorkletProcessor {
  constructor(opts) {
    super();
    this.targetRate = opts.processorOptions.targetRate;
    this.ratio = sampleRate / this.targetRate;
    this.acc = 0;
    this.buf = [];
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];
    for (let i = 0; i < ch.length; i++) {
      this.acc += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        const s = Math.max(-1, Math.min(1, ch[i]));
        this.buf.push(s < 0 ? s * 0x8000 : s * 0x7fff);
      }
    }
    if (this.buf.length >= 1024) {
      const out = new Int16Array(this.buf);
      this.buf = [];
      this.port.postMessage(out, [out.buffer]);
    }
    return true;
  }
}
registerProcessor("downsampler", DownsamplerWorklet);
`;

type Status = "idle" | "connecting" | "recording" | "stopping";

type VoiceOpt = { id: string; name: string; is_default: boolean };
type BotRuntime = {
  barge_in_rms_threshold?: number;
  barge_in_cooldown_ms?: number;
  barge_in_hold_frames?: number;
};

const MSE_MIME = "audio/mpeg";

function mseSupported(): boolean {
  if (typeof window === "undefined") return false;
  if (!("MediaSource" in window)) return false;
  try {
    return MediaSource.isTypeSupported(MSE_MIME);
  } catch {
    return false;
  }
}

class MseAudioPlayer {
  private audio: HTMLAudioElement;
  private onSegmentDone: (idx: number) => void;
  private mediaSource: MediaSource | null = null;
  private sourceBuffer: SourceBuffer | null = null;
  private pending: ArrayBuffer[] = [];
  private ready = false;
  private segEnds = new Map<number, number>();
  private pendingSegEnd: number | null = null;
  private rafId: number | null = null;
  private objectUrl: string | null = null;

  constructor(audio: HTMLAudioElement, onSegmentDone: (idx: number) => void) {
    this.audio = audio;
    this.onSegmentDone = onSegmentDone;
  }

  start(): boolean {
    try {
      this.mediaSource = new MediaSource();
      this.objectUrl = URL.createObjectURL(this.mediaSource);
      this.audio.src = this.objectUrl;
      this.mediaSource.addEventListener("sourceopen", () => this.onSourceOpen(), { once: true });
      this.audio.play().catch(() => {});
      this.rafId = requestAnimationFrame(this.tick);
      return true;
    } catch {
      this.destroy();
      return false;
    }
  }

  private onSourceOpen() {
    if (!this.mediaSource) return;
    try {
      const sb = this.mediaSource.addSourceBuffer(MSE_MIME);
      sb.mode = "sequence";
      sb.addEventListener("updateend", () => this.onUpdateEnd());
      this.sourceBuffer = sb;
      this.ready = true;
      this.drain();
    } catch {
      // Ignore: fallback will take over via push() returning false.
    }
  }

  push(buf: ArrayBuffer) {
    this.pending.push(buf);
    this.drain();
  }

  markSegmentEnd(idx: number) {
    this.pendingSegEnd = idx;
    if (this.sourceBuffer && !this.sourceBuffer.updating && this.pending.length === 0) {
      this.captureSegEnd();
    }
  }

  private onUpdateEnd() {
    if (this.pendingSegEnd !== null && this.pending.length === 0) {
      this.captureSegEnd();
    }
    this.drain();
  }

  private captureSegEnd() {
    if (this.pendingSegEnd === null || !this.sourceBuffer) return;
    const ranges = this.sourceBuffer.buffered;
    const endPos = ranges.length > 0 ? ranges.end(ranges.length - 1) : 0;
    this.segEnds.set(this.pendingSegEnd, endPos);
    this.pendingSegEnd = null;
  }

  private tick = () => {
    this.rafId = requestAnimationFrame(this.tick);
    if (this.segEnds.size === 0) return;
    const t = this.audio.currentTime;
    for (const [idx, endPos] of [...this.segEnds.entries()]) {
      if (t >= endPos - 0.05) {
        this.segEnds.delete(idx);
        this.onSegmentDone(idx);
      }
    }
  };

  private drain() {
    if (!this.ready || !this.sourceBuffer || this.sourceBuffer.updating) return;
    const next = this.pending.shift();
    if (!next) return;
    try {
      this.sourceBuffer.appendBuffer(next);
    } catch {
      // QuotaExceededError or invalid state: drop and continue.
    }
  }

  flush() {
    this.pending = [];
    this.segEnds.clear();
    this.pendingSegEnd = null;
    if (this.sourceBuffer) {
      try {
        if (this.sourceBuffer.updating) this.sourceBuffer.abort();
      } catch {
        /* ignore */
      }
      try {
        const buf = this.sourceBuffer.buffered;
        if (buf.length > 0) {
          this.sourceBuffer.remove(0, buf.end(buf.length - 1));
        }
      } catch {
        /* ignore */
      }
    }
    try {
      this.audio.pause();
      this.audio.currentTime = 0;
    } catch {
      /* ignore */
    }
  }

  destroy() {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    try {
      if (this.mediaSource && this.mediaSource.readyState === "open") {
        this.mediaSource.endOfStream();
      }
    } catch {
      /* ignore */
    }
    try {
      this.audio.pause();
      this.audio.removeAttribute("src");
      this.audio.load();
    } catch {
      /* ignore */
    }
    if (this.objectUrl) {
      try {
        URL.revokeObjectURL(this.objectUrl);
      } catch {
        /* ignore */
      }
      this.objectUrl = null;
    }
    this.sourceBuffer = null;
    this.mediaSource = null;
    this.ready = false;
  }
}

export default function VoicePage() {
  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const vadRafRef = useRef<number | null>(null);
  const vadBufRef = useRef<Uint8Array | null>(null);
  const lastBargeRef = useRef(0);
  const loudFramesRef = useRef(0);
  const botSpeakingRef = useRef(false);
  const bargeInRmsRef = useRef(DEFAULT_BARGE_IN_RMS);
  const bargeInCooldownMsRef = useRef(DEFAULT_BARGE_IN_COOLDOWN_MS);
  const bargeInHoldFramesRef = useRef(DEFAULT_BARGE_IN_HOLD_FRAMES);

  const audioElRef = useRef<HTMLAudioElement | null>(null);
  const incomingSegmentRef = useRef<number | null>(null);
  const playerRef = useRef<MseAudioPlayer | null>(null);
  const fallbackQueueRef = useRef<Array<{ blob: Blob; segmentIndex: number | null }>>([]);
  const fallbackPlayingRef = useRef(false);

  const [status, setStatus] = useState<Status>("idle");
  const [locale, setLocale] = useState<"ru" | "uz">("ru");
  const [error, setError] = useState<string | null>(null);
  const [bytesSent, setBytesSent] = useState(0);
  const [bytesRecv, setBytesRecv] = useState(0);
  const [bargeInCount, setBargeInCount] = useState(0);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [lastEscalation, setLastEscalation] = useState<Record<string, unknown> | null>(null);
  const [voiceOpts, setVoiceOpts] = useState<VoiceOpt[]>([]);
  const [voiceProfileId, setVoiceProfileId] = useState<string>("");
  const [bargeInHint, setBargeInHint] = useState("");
  const [callState, setCallState] = useState<string>("listening");
  const [segmentCount, setSegmentCount] = useState(0);

  useEffect(() => () => stop(), []);

  useEffect(() => {
    let cancel = false;
    fetch(`${API_BASE}/voices`)
      .then((r) => r.json())
      .then((list: VoiceOpt[]) => {
        if (cancel || !Array.isArray(list)) return;
        setVoiceOpts(list);
        const d = list.find((x) => x.is_default) ?? list[0];
        if (d) setVoiceProfileId((cur) => (cur ? cur : d.id));
      })
      .catch(() => {});
    return () => {
      cancel = true;
    };
  }, []);

  useEffect(() => {
    let cancel = false;
    fetch(`${API_BASE}/bot-settings`)
      .then((r) => (r.ok ? r.json() : null))
      .then((cfg: BotRuntime | null) => {
        if (cancel || !cfg) return;
        const rms = Number(cfg.barge_in_rms_threshold ?? DEFAULT_BARGE_IN_RMS);
        const cd = Number(cfg.barge_in_cooldown_ms ?? DEFAULT_BARGE_IN_COOLDOWN_MS);
        const hold = Number(cfg.barge_in_hold_frames ?? DEFAULT_BARGE_IN_HOLD_FRAMES);
        bargeInRmsRef.current = Number.isFinite(rms) ? Math.max(0.01, rms) : DEFAULT_BARGE_IN_RMS;
        bargeInCooldownMsRef.current = Number.isFinite(cd)
          ? Math.max(100, Math.round(cd))
          : DEFAULT_BARGE_IN_COOLDOWN_MS;
        bargeInHoldFramesRef.current = Number.isFinite(hold)
          ? Math.max(1, Math.round(hold))
          : DEFAULT_BARGE_IN_HOLD_FRAMES;
        setBargeInHint(
          `barge-in: rms>${bargeInRmsRef.current.toFixed(3)}, cooldown ${bargeInCooldownMsRef.current}ms, hold ${bargeInHoldFramesRef.current}`
        );
      })
      .catch(() => {});
    return () => {
      cancel = true;
    };
  }, []);

  function ensureAudioEl(): HTMLAudioElement {
    let a = audioElRef.current;
    if (!a) {
      a = new Audio();
      audioElRef.current = a;
    }
    return a;
  }

  function ackSegment(idx: number) {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "audio_played_ack", index: idx }));
    }
  }

  function pushAudioChunk(buf: ArrayBuffer) {
    botSpeakingRef.current = true;
    const player = playerRef.current;
    if (player) {
      player.push(buf);
      return;
    }
    const blob = new Blob([buf], { type: MSE_MIME });
    fallbackQueueRef.current.push({
      blob,
      segmentIndex: incomingSegmentRef.current,
    });
    fallbackPlayNext();
  }

  function markSegmentEndForPlayback(idx: number) {
    const player = playerRef.current;
    if (player) {
      player.markSegmentEnd(idx);
    } else {
      ackSegment(idx);
    }
  }

  function flushPlayback() {
    botSpeakingRef.current = false;
    const player = playerRef.current;
    if (player) {
      player.flush();
      return;
    }
    fallbackQueueRef.current = [];
    fallbackPlayingRef.current = false;
    const a = audioElRef.current;
    if (a) {
      a.pause();
      a.removeAttribute("src");
      a.load();
    }
  }

  function fallbackPlayNext() {
    if (fallbackPlayingRef.current) return;
    const item = fallbackQueueRef.current.shift();
    if (!item) {
      botSpeakingRef.current = false;
      return;
    }
    const { blob, segmentIndex } = item;
    fallbackPlayingRef.current = true;
    botSpeakingRef.current = true;
    const url = URL.createObjectURL(blob);
    const audio = ensureAudioEl();
    audio.src = url;
    audio.onended = () => {
      if (segmentIndex != null) ackSegment(segmentIndex);
      URL.revokeObjectURL(url);
      fallbackPlayingRef.current = false;
      fallbackPlayNext();
    };
    audio.onerror = () => {
      URL.revokeObjectURL(url);
      fallbackPlayingRef.current = false;
      fallbackPlayNext();
    };
    audio.play().catch(() => {
      fallbackPlayingRef.current = false;
    });
  }

  function startVadLoop(ws: WebSocket) {
    const tick = () => {
      vadRafRef.current = requestAnimationFrame(tick);
      const analyser = analyserRef.current;
      if (ws.readyState !== WebSocket.OPEN || !analyser) return;
      if (!vadBufRef.current || vadBufRef.current.length !== analyser.frequencyBinCount) {
        vadBufRef.current = new Uint8Array(analyser.frequencyBinCount);
      }
      const buf = vadBufRef.current;
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) {
        const v = (buf[i]! - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / buf.length);
      const now = performance.now();
      if (rms > bargeInRmsRef.current) loudFramesRef.current += 1;
      else loudFramesRef.current = 0;
      if (
        botSpeakingRef.current &&
        loudFramesRef.current >= bargeInHoldFramesRef.current &&
        now - lastBargeRef.current > bargeInCooldownMsRef.current
      ) {
        loudFramesRef.current = 0;
        lastBargeRef.current = now;
        ws.send(JSON.stringify({ type: "interrupt" }));
        flushPlayback();
        setBargeInCount((n) => n + 1);
      }
    };
    vadRafRef.current = requestAnimationFrame(tick);
  }

  async function start() {
    setError(null);
    setBytesSent(0);
    setBytesRecv(0);
    setBargeInCount(0);
    setConversationId(null);
    setLastEscalation(null);
    setStatus("connecting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48_000,
          echoCancellation: true,
          noiseSuppression: true,
        },
        video: false,
      });
      streamRef.current = stream;

      const ctx = new AudioContext();
      ctxRef.current = ctx;
      const moduleUrl = URL.createObjectURL(
        new Blob([WORKLET_SRC], { type: "application/javascript" })
      );
      await ctx.audioWorklet.addModule(moduleUrl);
      URL.revokeObjectURL(moduleUrl);

      const audio = ensureAudioEl();
      if (mseSupported()) {
        const player = new MseAudioPlayer(audio, ackSegment);
        if (player.start()) {
          playerRef.current = player;
        } else {
          playerRef.current = null;
        }
      } else {
        playerRef.current = null;
      }

      const sp = new URLSearchParams({
        locale,
        sample_rate: String(TARGET_SAMPLE_RATE),
      });
      if (voiceProfileId) sp.set("voice_id", voiceProfileId);
      const ws = new WebSocket(backendWsUrl(`/ws/voice?${sp.toString()}`));
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          try {
            const j = JSON.parse(ev.data as string);
            const evtType = String(j.event || j.type || "");
            const evtData =
              j && typeof j.data === "object" && j.data !== null
                ? (j.data as Record<string, unknown>)
                : (j as Record<string, unknown>);
            if (evtType === "interrupted") flushPlayback();
            if (evtType === "state" && evtData.state) setCallState(String(evtData.state));
            if (evtType === "audio_segment_start") {
              setSegmentCount((n) => n + 1);
              incomingSegmentRef.current = Number(evtData.index ?? -1);
            }
            if (evtType === "audio_segment_end") {
              const idx = Number(evtData.index ?? incomingSegmentRef.current ?? -1);
              if (idx >= 0) markSegmentEndForPlayback(idx);
              incomingSegmentRef.current = null;
            }
            if (evtType === "session" && evtData.conversation_id)
              setConversationId(String(evtData.conversation_id));
            if (evtType === "escalation_packet") {
              flushPlayback();
              setLastEscalation(evtData);
            }
          } catch {
            /* ignore */
          }
          return;
        }
        const buf = ev.data as ArrayBuffer;
        setBytesRecv((n) => n + buf.byteLength);
        pushAudioChunk(buf);
      };
      ws.onerror = () => setError("WebSocket error");
      ws.onclose = () => {
        if (vadRafRef.current != null) {
          cancelAnimationFrame(vadRafRef.current);
          vadRafRef.current = null;
        }
        setStatus("idle");
      };

      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => resolve();
        ws.addEventListener("error", () => reject(new Error("ws_open_failed")), { once: true });
      });

      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyserRef.current = analyser;
      source.connect(analyser);

      const node = new AudioWorkletNode(ctx, "downsampler", {
        processorOptions: { targetRate: TARGET_SAMPLE_RATE },
      });
      workletRef.current = node;
      node.port.onmessage = (msg) => {
        const pcm = msg.data as Int16Array;
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(pcm.buffer);
          setBytesSent((n) => n + pcm.byteLength);
        }
      };
      source.connect(node);

      startVadLoop(ws);
      setStatus("recording");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      stop();
    }
  }

  function stop() {
    setStatus("stopping");
    if (vadRafRef.current != null) {
      cancelAnimationFrame(vadRafRef.current);
      vadRafRef.current = null;
    }
    flushPlayback();
    try {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "end" }));
        ws.close();
      }
    } catch {
      // noop
    }
    workletRef.current?.disconnect();
    analyserRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    ctxRef.current?.close().catch(() => undefined);
    if (playerRef.current) {
      playerRef.current.destroy();
      playerRef.current = null;
    }
    workletRef.current = null;
    analyserRef.current = null;
    streamRef.current = null;
    ctxRef.current = null;
    wsRef.current = null;
    setStatus("idle");
  }

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-3xl font-semibold">Голосовой плейграунд</h2>
        <p className="text-mutedForeground mt-1">
          Непрерывный разговор: микрофон всегда в эфире, ответ бота можно{" "}
          <strong>перебить</strong> — во время воспроизведения клиент по громкости (RMS) шлёт{" "}
          <code className="mx-1 bg-muted px-1 rounded">interrupt</code>, на сервере перед STT
          отдельно включён VAD (Silero ONNX или WebRTC по настройке бэкенда).
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Управление</CardTitle>
          <CardDescription>
            Режим имитирует линию: аудио на сервере проходит VAD, затем STT; новая финальная фраза
            отменяет текущий ответ бота.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <select
              value={locale}
              onChange={(e) => setLocale(e.target.value as "ru" | "uz")}
              disabled={status !== "idle"}
              className="rounded-md border border-border bg-background px-3 py-2 text-sm"
            >
              <option value="ru">Русский</option>
              <option value="uz">O‘zbekcha</option>
            </select>
            {voiceOpts.length > 0 && (
              <select
                value={voiceProfileId}
                onChange={(e) => setVoiceProfileId(e.target.value)}
                disabled={status !== "idle"}
                className="rounded-md border border-border bg-background px-3 py-2 text-sm min-w-[12rem]"
              >
                {voiceOpts.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.name}
                    {v.is_default ? " ★" : ""}
                  </option>
                ))}
              </select>
            )}
            {status === "idle" && <Button onClick={start}>Начать разговор</Button>}
            {status === "recording" && (
              <Button variant="destructive" onClick={stop}>
                Завершить
              </Button>
            )}
            {status === "connecting" && <Badge>Подключаюсь…</Badge>}
            {status === "stopping" && <Badge>Завершаю…</Badge>}
            <Badge tone={status === "recording" ? "success" : "default"}>
              {status === "recording" ? "В эфире" : "Не активно"}
            </Badge>
            {status === "recording" && <Badge>State: {callState}</Badge>}
            {segmentCount > 0 && <Badge>Сегментов: {segmentCount}</Badge>}
            {bargeInCount > 0 && <Badge tone="warning">Перебиваний: {bargeInCount}</Badge>}
            <Button
              variant="outline"
              size="sm"
              disabled={status !== "recording"}
              onClick={() => {
                wsRef.current?.send(
                  JSON.stringify({ type: "escalate", reason: "manual_operator_request" })
                );
              }}
            >
              Эскалация на оператора
            </Button>
          </div>
          {conversationId && (
            <p className="text-xs text-mutedForeground">
              ID сессии: <code className="bg-muted px-1 rounded">{conversationId}</code> —{" "}
              <a
                className="text-primary underline"
                href={`${API_BASE}/conversations/${conversationId}`}
                target="_blank"
                rel="noreferrer"
              >
                JSON транскрипта
              </a>
            </p>
          )}
          <div className="grid grid-cols-2 gap-3 max-w-sm text-sm">
            <Stat label="Отправлено (PCM)" value={`${(bytesSent / 1024).toFixed(1)} KB`} />
            <Stat label="Получено (MP3)" value={`${(bytesRecv / 1024).toFixed(1)} KB`} />
          </div>
          {bargeInHint && <p className="text-xs text-mutedForeground">{bargeInHint}</p>}
          {error && <Badge tone="danger">Ошибка: {error}</Badge>}
        </CardContent>
      </Card>

      {lastEscalation && (
        <Card>
          <CardHeader>
            <CardTitle>Последняя эскалация</CardTitle>
            <CardDescription>
              Пакет для CRM / софтфона: краткое саммари + полная история реплик (роли customer /
              assistant).
            </CardDescription>
          </CardHeader>
          <CardContent className="text-xs space-y-2">
            <div>
              <span className="text-mutedForeground">Причина:</span>{" "}
              {String(lastEscalation.reason ?? "")}
            </div>
            <div>
              <span className="text-mutedForeground">Саммари:</span>{" "}
              {String(lastEscalation.summary ?? "")}
            </div>
            <details>
              <summary className="cursor-pointer text-mutedForeground">История (turns)</summary>
              <pre className="mt-2 max-h-48 overflow-auto rounded-md bg-muted p-2 whitespace-pre-wrap">
                {JSON.stringify(lastEscalation.turns, null, 2)}
              </pre>
            </details>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Как это работает</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-mutedForeground space-y-2">
          <p>
            Пока бот говорит, анализатор уровня с микрофона ловит ваш голос и шлёт{" "}
            <code className="bg-muted px-1 rounded">interrupt</code> — сервер отменяет LLM/TTS.
            Дополнительно любая <strong>новая</strong> распознанная фраза тоже прерывает предыдущий
            ответ (логика в Voice Engine).
          </p>
          <p>
            База знаний для этого канала — пул <code className="bg-muted px-1">voice</code> (см.
            раздел «База знаний»). Суфлёр оператора использует отдельный пул{" "}
            <code className="bg-muted px-1">agent_assist</code>.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border p-2">
      <div className="text-xs text-mutedForeground">{label}</div>
      <div className="font-medium">{value}</div>
    </div>
  );
}
