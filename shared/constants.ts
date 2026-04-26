/**
 * Shared constants between backend and frontend.
 * Keep this file dependency-free so it can be consumed by either side.
 */

export const SUPPORTED_LOCALES = ["ru", "uz"] as const;
export type Locale = (typeof SUPPORTED_LOCALES)[number];

export const LATENCY_BUDGET_MS = 1500;
export const DEFAULT_AUDIO_SAMPLE_RATE = 16_000;

export const INTENT_GROUPS = {
  consultative: ["tariffs", "products", "office_hours", "branches"] as const,
  operational: [
    "block_card",
    "transfer_money",
    "personal_data",
    "complaint",
    "human_agent",
  ] as const,
} as const;
