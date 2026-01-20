"use client";

import { useMemo, useState } from "react";

type ApiError = { message: string };

export default function Page() {
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

  const [prompt, setPrompt] = useState(
    "I want a low-sugar snack for kids, shaped like a rabbit."
  );
  const [mode, setMode] = useState<"parse" | "compare">("parse");

  const [loading, setLoading] = useState(false);
  const [resultParse, setResultParse] = useState<any>(null);
  const [resultExtract, setResultExtract] = useState<any>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const prettyParse = useMemo(
    () => (resultParse ? JSON.stringify(resultParse, null, 2) : ""),
    [resultParse]
  );

  const prettyExtract = useMemo(
    () => (resultExtract ? JSON.stringify(resultExtract, null, 2) : ""),
    [resultExtract]
  );

  async function postJson(path: string, body: any) {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Request failed (${res.status}): ${text}`);
    }

    return res.json();
  }

  async function run() {
    setLoading(true);
    setError(null);
    setResultParse(null);
    setResultExtract(null);

    try {
      // Always run /parse for the main pipeline output
      const parsed = await postJson("/parse", { prompt });
      setResultParse(parsed);

      // Optionally also run /debug/extract for comparison
      if (mode === "compare") {
        const extracted = await postJson("/debug/extract", { prompt });
        setResultExtract(extracted);
      }
    } catch (e: any) {
      setError({ message: e?.message ?? "Unknown error" });
    } finally {
      setLoading(false);
    }
  }

  const clarificationItems: Array<{ field: string; question: string }> =
    resultParse?.needs_clarification ?? [];

  const defaultsApplied: Array<{ field: string; reason: string; default: any }> =
    resultParse?.defaults_applied ?? [];

  return (
    <main style={{ maxWidth: 1100, margin: "0 auto", padding: 24, fontFamily: "system-ui, sans-serif", color: "#111" }}>
      <header style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 16, color: "#fff" }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 750, margin: 0 }}>GenAI Food Fabrication</h1>
          <p style={{ margin: "6px 0 0", opacity: 0.75, color: "#fff" }}>
            Prompt → LLM raw intent → deterministic normalization → requirement.json
          </p>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 14 }}>
            <input
              type="radio"
              name="mode"
              checked={mode === "parse"}
              onChange={() => setMode("parse")}
            />
            Parse
          </label>
          <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 14 }}>
            <input
              type="radio"
              name="mode"
              checked={mode === "compare"}
              onChange={() => setMode("compare")}
            />
            Compare (also call /debug/extract)
          </label>
        </div>
      </header>

      <section style={{ marginTop: 18 }}>
        <label style={{ fontSize: 14, fontWeight: 650, color: "#fff" }}>Prompt</label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={5}
          style={{
            width: "100%",
            marginTop: 8,
            padding: 12,
            borderRadius: 12,
            color: "#fff", 
            border: "1px solid #d0d0d0",
            lineHeight: 1.4,
          }}
          placeholder="Describe your snack request in English..."
        />

        <div style={{ display: "flex", gap: 10, marginTop: 10 }}>
          <button
            onClick={run}
            disabled={loading || prompt.trim().length === 0}
            style={{
              padding: "10px 14px",
              borderRadius: 12,
              border: "1px solid #222",
              background: "white",
              color: "#111",
              cursor: loading ? "not-allowed" : "pointer",
              fontWeight: 650,
            }}
          >
            {loading ? "Running..." : "Run"}
          </button>

          <button
            onClick={() => {
              mentions();
            }}
            style={{
              padding: "10px 14px",
              borderRadius: 12,
              border: "1px solid #d0d0d0",
              background: "white",
              color: "#111",
              cursor: "pointer",
            }}
            title="Fill example prompts"
          >
            Examples
          </button>
        </div>

        {error && (
          <div style={{ marginTop: 12, padding: 12, borderRadius: 12, border: "1px solid #ffd0d0", background: "#fff7f7" }}>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>Error</div>
            <div style={{ whiteSpace: "pre-wrap" }}>{error.message}</div>
          </div>
        )}
      </section>

      <section style={{ marginTop: 18, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card title="Clarifications">
          {clarificationItems.length === 0 ? (
            <p style={{ margin: 0, opacity: 0.75 }}>No clarification needed.</p>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {clarificationItems.map((c, idx) => (
                <li key={`${c.field}-${idx}`} style={{ marginBottom: 8 }}>
                  <div style={{ fontWeight: 650 }}>{c.field}</div>
                  <div style={{ opacity: 0.85 }}>{c.question}</div>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="Defaults applied">
          {defaultsApplied.length === 0 ? (
            <p style={{ margin: 0, opacity: 0.75 }}>No defaults applied.</p>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {defaultsApplied.map((d, idx) => (
                <li key={`${d.field}-${idx}`} style={{ marginBottom: 8 }}>
                  <div style={{ fontWeight: 650 }}>{d.field}</div>
                  <div style={{ opacity: 0.85 }}>{d.reason}</div>
                  <pre style={{ margin: "6px 0 0", padding: 10, borderRadius: 10, border: "1px solid #eee", background: "#fafafa" }}>
                    {JSON.stringify(d.default, null, 2)}
                  </pre>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </section>

      <section style={{ marginTop: 14, display: "grid", gridTemplateColumns: mode === "compare" ? "1fr 1fr" : "1fr", gap: 14 }}>
        <Card title="/parse result (requirement.json)">
          <pre style={preStyle()}>{prettyParse || "Run the pipeline to see output."}</pre>
        </Card>

        {mode === "compare" && (
          <Card title="/debug/extract result (raw intent only)">
            <pre style={preStyle()}>{prettyExtract || "Run in compare mode to see raw intent output."}</pre>
          </Card>
        )}
      </section>

      <footer style={{ marginTop: 18, opacity: 0.6, fontSize: 13 }}>
        API Base URL: <code>{API_BASE}</code>
      </footer>
    </main>
  );

  function mentions() {
    // Example prompts used for quick testing.
    const examples = [
      "I want a low-sugar snack for kids, shaped like a rabbit.",
      "Make a snack for kids shaped like a star. No nuts.",
      "I want a snack shaped like a circle. 200-300 kcal. sugar 5g.",
      "Snack for kids, rabbit shape, soft texture, low sugar.",
      "Make a crunchy snack shaped like a dog. sugar-free.",
    ];
    const next = examples[Math.floor(Math.random() * examples.length)];
    setPrompt(next);
  }
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ border: "1px solid #e6e6e6", borderRadius: 16, padding: 14, background: "white" }}>
      <div style={{ fontWeight: 750, marginBottom: 10 }}>{title}</div>
      {children}
    </div>
  );
}

function preStyle(): React.CSSProperties {
  return {
    margin: 0,
    padding: 12,
    borderRadius: 12,
    border: "1px solid #eee",
    background: "#fafafa",
    overflowX: "auto",
    fontSize: 12,
    color: "#111",
    lineHeight: 1.4,
    maxHeight: 520,
  };
}
