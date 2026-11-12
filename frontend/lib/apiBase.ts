  export function getApiBase(): string {
    const env = process.env.NEXT_PUBLIC_API_URL;
    if (env) return env;
    if (typeof window !== "undefined") return window.location.origin;
    return "http://localhost:8000";
  }