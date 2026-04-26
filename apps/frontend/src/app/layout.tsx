import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "VoiceAgent — Контрольная панель",
  description: "Голосовой ИИ-агент для контакт-центра",
};

const NAV = [
  { href: "/", label: "Дашборд" },
  { href: "/playground", label: "Плейграунд" },
  { href: "/voice", label: "Голосовой канал" },
  { href: "/agent-assist", label: "Суфлёр оператора" },
  { href: "/knowledge", label: "База знаний" },
  { href: "/prompts", label: "Промпты" },
  { href: "/voices", label: "Голоса" },
  { href: "/bot", label: "Поведение бота" },
  { href: "/analytics", label: "Речевая аналитика" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru">
      <body>
        <div className="min-h-screen flex">
          <aside className="hidden md:flex w-64 flex-col border-r border-border bg-muted/40 p-6 gap-6">
            <div>
              <p className="text-xs uppercase tracking-wider text-mutedForeground">VoiceAgent</p>
              <h1 className="text-xl font-semibold">Контрольная панель</h1>
            </div>
            <nav className="flex flex-col gap-1">
              {NAV.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="rounded-md px-3 py-2 text-sm hover:bg-accent transition-colors"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
            <div className="mt-auto text-xs text-mutedForeground">
              Прототип · v0.1.0
            </div>
          </aside>
          <main className="flex-1 p-6 md:p-10 max-w-6xl mx-auto w-full">{children}</main>
        </div>
      </body>
    </html>
  );
}
