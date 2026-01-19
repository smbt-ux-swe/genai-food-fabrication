"use client";

import { useMemo, useState } from "react";

export default function Page() {
  const [prompt, setPrompt] = useState("Kids snack, rabbit shape, low sugar");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const pretty = useMemo(() => (result ? JSON.stringify(result, null, 2) : ""), [result]);

  async function run() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("http://localhost:8000/parse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const data = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e?.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 960, margin: "0 auto", padding: 24, fontFamily: "sans-serif" }}>
      <h1 style={{ fontSize: 22, fontWeight: 700 }}>Prompt to requirement.json</h1>

      <p style={{ opacity: 0.8, marginTop: 8 }}>
        Next.js frontend calls FastAPI backend to generate requirement.json
      </p>

      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={5}
        style={{ width: "100%", marginTop: 16, padding: 12, borderRadius: 10, border: "1px solid #ccc" }}
      />

      <button
        onClick={run}
        disabled={loading || prompt.trim().length === 0}
        style={{ marginTop: 12, padding: "10px 14px", borderRadius: 10, border: "1px solid #333" }}
      >
        {loading ? "Running..." : "Run"}
      </button>

      {error && <p style={{ color: "crimson", marginTop: 12 }}>{error}</p>}

      {result && (
        <pre style={{ marginTop: 16, padding: 12, borderRadius: 10, border: "1px solid #ddd", background: "#fafafa", color: "black" }}>
          {pretty}
        </pre>
      )}
    </main>
  );
}
