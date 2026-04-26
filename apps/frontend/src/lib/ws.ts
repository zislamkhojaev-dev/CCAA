/** Resolve the backend WebSocket URL from the current page origin. */
export function backendWsUrl(path: string): string {
  if (typeof window === "undefined") return "";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = process.env.NEXT_PUBLIC_BACKEND_WS_HOST || window.location.host;

  // When the frontend is served on :3000 in dev, point to the backend on :8000.
  // In production both share the same host (rewrite or reverse proxy).
  const isLocalDev = host.endsWith(":3000");
  const target = isLocalDev ? host.replace(":3000", ":8000") : host;
  return `${proto}//${target}/api/v1${path}`;
}
