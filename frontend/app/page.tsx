"use client";

import { useEffect, useMemo, useState } from "react";

type ApiError = { message: string };

type Mode = "parse" | "compare-raw" | "compare-dietitian";

type Sex = "female" | "male" | "other";
type ActivityLevel =
  | "sedentary"
  | "light"
  | "moderate"
  | "active"
  | "very_active";

type UserProfile = {
  id: string;
  profileName: string;
  sex: Sex;
  weightKg: number;
  heightCm: number;
  age: number;
  activityLevel: ActivityLevel;
  illnessCondition: string;
  createdAtIso: string;
};

const STORAGE_KEY = "genai_food_fabrication_user_profiles_v1";
const STORAGE_SELECTED_KEY = "genai_food_fabrication_selected_profile_id_v1";

function uid() {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function clampNumber(v: number, min: number, max: number) {
  if (Number.isNaN(v)) return min;
  return Math.max(min, Math.min(max, v));
}

function formatProfileForPrompt(p: UserProfile) {
  return [
    "User profile for nutrition calculation",
    `Profile name: ${p.profileName}`,
    `Sex: ${p.sex}`,
    `Age: ${p.age}`,
    `Height: ${p.heightCm} cm`,
    `Weight: ${p.weightKg} kg`,
    `Activity level: ${p.activityLevel}`,
    `Illness condition: ${p.illnessCondition || "none"}`,
  ].join("\n");
}

export default function Page() {
  const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

  const [prompt, setPrompt] = useState(
    "I want a low-sugar snack for kids, shaped like a rabbit."
  );
  const [mode, setMode] = useState<Mode>("parse");

  const [loading, setLoading] = useState(false);
  const [resultParse, setResultParse] = useState<any>(null);
  const [resultExtract, setResultExtract] = useState<any>(null);
  const [resultDietitianCompare, setResultDietitianCompare] = useState<any>(null);
  const [error, setError] = useState<ApiError | null>(null);

  // Profiles
  const [profiles, setProfiles] = useState<UserProfile[]>([]);
  const [selectedProfileId, setSelectedProfileId] = useState<string>("");

  const selectedProfile = useMemo(() => {
    return profiles.find((p) => p.id === selectedProfileId) ?? null;
  }, [profiles, selectedProfileId]);

  const [showCreate, setShowCreate] = useState(false);

  const [draftName, setDraftName] = useState("");
  const [draftSex, setDraftSex] = useState<Sex>("female");
  const [draftWeightKg, setDraftWeightKg] = useState<number>(60);
  const [draftHeightCm, setDraftHeightCm] = useState<number>(160);
  const [draftAge, setDraftAge] = useState<number>(75);
  const [draftActivity, setDraftActivity] = useState<ActivityLevel>("sedentary");
  const [draftIllness, setDraftIllness] = useState<string>("");

  // Load profiles from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      const parsed: UserProfile[] = raw ? JSON.parse(raw) : [];
      setProfiles(Array.isArray(parsed) ? parsed : []);

      const savedSelected = localStorage.getItem(STORAGE_SELECTED_KEY) || "";
      if (savedSelected) {
        setSelectedProfileId(savedSelected);
      } else if (parsed?.length) {
        setSelectedProfileId(parsed[0].id);
      }
    } catch {
      setProfiles([]);
    }
  }, []);

  // Persist profiles
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(profiles));
    } catch {}
  }, [profiles]);

  // Persist selection
  useEffect(() => {
    try {
      if (selectedProfileId) {
        localStorage.setItem(STORAGE_SELECTED_KEY, selectedProfileId);
      }
    } catch {}
  }, [selectedProfileId]);

  const prettyParse = useMemo(
    () => (resultParse ? JSON.stringify(resultParse, null, 2) : ""),
    [resultParse]
  );

  const prettyExtract = useMemo(
    () => (resultExtract ? JSON.stringify(resultExtract, null, 2) : ""),
    [resultExtract]
  );

  const prettyDietitianCompare = useMemo(
    () =>
      resultDietitianCompare
        ? JSON.stringify(resultDietitianCompare, null, 2)
        : "",
    [resultDietitianCompare]
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

  function buildEffectivePrompt() {
    if (!selectedProfile) return prompt;

    return [
      prompt.trim(),
      "",
      "Context",
      formatProfileForPrompt(selectedProfile),
    ].join("\n");
  }

  async function run() {
    setLoading(true);
    setError(null);
    setResultParse(null);
    setResultExtract(null);
    setResultDietitianCompare(null);

    try {
      const effectivePrompt = buildEffectivePrompt();

      const parsed = await postJson("/parse", { prompt: effectivePrompt });
      setResultParse(parsed);

      if (mode === "compare-raw") {
        const extracted = await postJson("/debug/extract", { prompt: effectivePrompt });
        setResultExtract(extracted);
      }

      if (mode === "compare-dietitian") {
        const requirement =
          parsed?.normalized_requirement ?? parsed?.result?.normalized_requirement;

        if (!requirement) {
          throw new Error(
            "Could not find normalized_requirement in /parse response. Check backend response shape."
          );
        }

        const dietitianCompare = await postJson("/agents/dietitian/compare", {
          requirement,
          use_kb: true,
          kb_path: "knowledgebases/dietitian_kb.md",
        });

        setResultDietitianCompare(dietitianCompare);
      }
    } catch (e: any) {
      setError({ message: e?.message ?? "Unknown error" });
    } finally {
      setLoading(false);
    }
  }

  function saveProfile() {
    const name = draftName.trim();
    if (!name) {
      setError({ message: "Profile name is required." });
      return;
    }

    const next: UserProfile = {
      id: uid(),
      profileName: name,
      sex: draftSex,
      weightKg: clampNumber(Number(draftWeightKg), 1, 400),
      heightCm: clampNumber(Number(draftHeightCm), 30, 250),
      age: clampNumber(Number(draftAge), 0, 130),
      activityLevel: draftActivity,
      illnessCondition: draftIllness.trim(),
      createdAtIso: new Date().toISOString(),
    };

    setProfiles((prev) => [next, ...prev]);
    setSelectedProfileId(next.id);
    setShowCreate(false);
    setDraftName("");
    setDraftIllness("");
    setError(null);
  }

  function deleteSelectedProfile() {
    if (!selectedProfile) return;
    const deletingId = selectedProfile.id;

    setProfiles((prev) => prev.filter((p) => p.id !== deletingId));

    const remaining = profiles.filter((p) => p.id !== deletingId);
    if (remaining.length) setSelectedProfileId(remaining[0].id);
    else setSelectedProfileId("");
  }

  const clarificationItems: Array<{ field: string; question: string }> =
    resultParse?.clarifications ?? [];

  const defaultsApplied: Array<{ field: string; reason: string; default: any }> =
    resultParse?.defaults_applied ?? [];

  return (
    <main
      style={{
        maxWidth: 1100,
        margin: "0 auto",
        padding: 24,
        fontFamily: "system-ui, sans-serif",
        color: "#111",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          gap: 16,
        }}
      >
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 750, margin: 0 }}>
            GenAI Food Fabrication
          </h1>
          <p style={{ margin: "6px 0 0", opacity: 0.75 }}>
            Prompt → LLM raw intent → deterministic normalization → requirement.json
          </p>
        </div>

        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
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
              checked={mode === "compare-raw"}
              onChange={() => setMode("compare-raw")}
            />
            Compare Raw (also call /debug/extract)
          </label>

          <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 14 }}>
            <input
              type="radio"
              name="mode"
              checked={mode === "compare-dietitian"}
              onChange={() => setMode("compare-dietitian")}
            />
            Compare Dietitian (KB vs no KB)
          </label>
        </div>
      </header>

      {/* Profile selector */}
      <section style={{ marginTop: 18 }}>
        <label style={{ fontSize: 14, fontWeight: 650 }}>User profile</label>

        <div style={{ display: "flex", gap: 10, marginTop: 8, alignItems: "center" }}>
          <select
            value={selectedProfileId}
            onChange={(e) => setSelectedProfileId(e.target.value)}
            style={{
              padding: "10px 12px",
              borderRadius: 12,
              border: "1px solid #d0d0d0",
              minWidth: 320,
              background: "white",
            }}
          >
            <option value="">No profile selected</option>
            {profiles.map((p) => (
              <option key={p.id} value={p.id}>
                {p.profileName}
              </option>
            ))}
          </select>

          <button
            onClick={() => setShowCreate((v) => !v)}
            style={{
              padding: "10px 14px",
              borderRadius: 12,
              border: "1px solid #d0d0d0",
              background: "white",
              color: "#111",
              cursor: "pointer",
              fontWeight: 650,
            }}
          >
            {showCreate ? "Close" : "Create"}
          </button>

          <button
            onClick={deleteSelectedProfile}
            disabled={!selectedProfile}
            style={{
              padding: "10px 14px",
              borderRadius: 12,
              border: "1px solid #d0d0d0",
              background: "white",
              color: "#111",
              cursor: selectedProfile ? "pointer" : "not-allowed",
              opacity: selectedProfile ? 1 : 0.5,
            }}
            title="Delete selected profile"
          >
            Delete
          </button>
        </div>

        {selectedProfile && (
          <div
            style={{
              marginTop: 10,
              padding: 12,
              borderRadius: 12,
              border: "1px solid #eee",
              background: "#fafafa",
              fontSize: 13,
              whiteSpace: "pre-wrap",
              lineHeight: 1.4,
            }}
          >
            {formatProfileForPrompt(selectedProfile)}
          </div>
        )}

        {showCreate && (
          <div
            style={{
              marginTop: 12,
              padding: 14,
              borderRadius: 16,
              border: "1px solid #e6e6e6",
              background: "white",
            }}
          >
            <div style={{ fontWeight: 750, marginBottom: 10 }}>Create profile</div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 12,
              }}
            >
              <div>
                <label style={{ fontSize: 13, fontWeight: 650 }}>Profile name</label>
                <input
                  value={draftName}
                  onChange={(e) => setDraftName(e.target.value)}
                  style={{
                    width: "100%",
                    marginTop: 6,
                    padding: "10px 12px",
                    borderRadius: 12,
                    border: "1px solid #d0d0d0",
                  }}
                  placeholder="e.g. Grandma 75"
                />
              </div>

              <div>
                <label style={{ fontSize: 13, fontWeight: 650 }}>Sex</label>
                <select
                  value={draftSex}
                  onChange={(e) => setDraftSex(e.target.value as Sex)}
                  style={{
                    width: "100%",
                    marginTop: 6,
                    padding: "10px 12px",
                    borderRadius: 12,
                    border: "1px solid #d0d0d0",
                    background: "white",
                  }}
                >
                  <option value="female">female</option>
                  <option value="male">male</option>
                  <option value="other">other</option>
                </select>
              </div>

              <div>
                <label style={{ fontSize: 13, fontWeight: 650 }}>Weight (kg)</label>
                <input
                  type="number"
                  value={draftWeightKg}
                  onChange={(e) => setDraftWeightKg(Number(e.target.value))}
                  style={{
                    width: "100%",
                    marginTop: 6,
                    padding: "10px 12px",
                    borderRadius: 12,
                    border: "1px solid #d0d0d0",
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: 13, fontWeight: 650 }}>Height (cm)</label>
                <input
                  type="number"
                  value={draftHeightCm}
                  onChange={(e) => setDraftHeightCm(Number(e.target.value))}
                  style={{
                    width: "100%",
                    marginTop: 6,
                    padding: "10px 12px",
                    borderRadius: 12,
                    border: "1px solid #d0d0d0",
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: 13, fontWeight: 650 }}>Age</label>
                <input
                  type="number"
                  value={draftAge}
                  onChange={(e) => setDraftAge(Number(e.target.value))}
                  style={{
                    width: "100%",
                    marginTop: 6,
                    padding: "10px 12px",
                    borderRadius: 12,
                    border: "1px solid #d0d0d0",
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: 13, fontWeight: 650 }}>Activity level</label>
                <select
                  value={draftActivity}
                  onChange={(e) => setDraftActivity(e.target.value as ActivityLevel)}
                  style={{
                    width: "100%",
                    marginTop: 6,
                    padding: "10px 12px",
                    borderRadius: 12,
                    border: "1px solid #d0d0d0",
                    background: "white",
                  }}
                >
                  <option value="sedentary">sedentary</option>
                  <option value="light">light</option>
                  <option value="moderate">moderate</option>
                  <option value="active">active</option>
                  <option value="very_active">very_active</option>
                </select>
              </div>

              <div style={{ gridColumn: "1 / -1" }}>
                <label style={{ fontSize: 13, fontWeight: 650 }}>
                  Illness condition
                </label>
                <input
                  value={draftIllness}
                  onChange={(e) => setDraftIllness(e.target.value)}
                  style={{
                    width: "100%",
                    marginTop: 6,
                    padding: "10px 12px",
                    borderRadius: 12,
                    border: "1px solid #d0d0d0",
                  }}
                  placeholder="e.g. diabetes, hypertension, kidney disease"
                />
              </div>
            </div>

            <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
              <button
                onClick={saveProfile}
                style={{
                  padding: "10px 14px",
                  borderRadius: 12,
                  border: "1px solid #222",
                  background: "white",
                  color: "#111",
                  cursor: "pointer",
                  fontWeight: 650,
                }}
              >
                Save
              </button>

              <button
                onClick={() => setShowCreate(false)}
                style={{
                  padding: "10px 14px",
                  borderRadius: 12,
                  border: "1px solid #d0d0d0",
                  background: "white",
                  color: "#111",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </section>

      <section style={{ marginTop: 18 }}>
        <label style={{ fontSize: 14, fontWeight: 650 }}>Prompt</label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={5}
          style={{
            width: "100%",
            marginTop: 8,
            padding: 12,
            borderRadius: 12,
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
          <div
            style={{
              marginTop: 12,
              padding: 12,
              borderRadius: 12,
              border: "1px solid #ffd0d0",
              background: "#fff7f7",
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 6 }}>Error</div>
            <div style={{ whiteSpace: "pre-wrap" }}>{error.message}</div>
          </div>
        )}
      </section>

      <section
        style={{
          marginTop: 18,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 14,
        }}
      >
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
                  <pre
                    style={{
                      margin: "6px 0 0",
                      padding: 10,
                      border: "1px solid #eee",
                      background: "#fafafa",
                      borderRadius: 10,
                    }}
                  >
                    {JSON.stringify(d.default, null, 2)}
                  </pre>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </section>

      <section
        style={{
          marginTop: 14,
          display: "grid",
          gridTemplateColumns:
            mode === "compare-raw"
              ? "1fr 1fr"
              : mode === "compare-dietitian"
              ? "1fr 1fr"
              : "1fr",
          gap: 14,
        }}
      >
        <Card title="/parse result (includes requirement.json)">
          <pre style={preStyle()}>{prettyParse || "Run the pipeline to see output."}</pre>
        </Card>

        {mode === "compare-raw" && (
          <Card title="/debug/extract result (raw intent only)">
            <pre style={preStyle()}>
              {prettyExtract || "Run in Compare Raw mode to see raw intent output."}
            </pre>
          </Card>
        )}

        {mode === "compare-dietitian" && (
          <Card title="/agents/dietitian/compare result (KB vs no KB)">
            <pre style={preStyle()}>
              {prettyDietitianCompare || "Run in Compare Dietitian mode to see output."}
            </pre>
          </Card>
        )}
      </section>

      {mode === "compare-dietitian" && resultDietitianCompare && (
        <section
          style={{
            marginTop: 14,
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 14,
          }}
        >
          <Card title="Dietitian without KB">
            <pre style={preStyle()}>
              {JSON.stringify(resultDietitianCompare?.without_kb ?? {}, null, 2)}
            </pre>
          </Card>

          <Card title="Dietitian with KB">
            <pre style={preStyle()}>
              {JSON.stringify(resultDietitianCompare?.with_kb ?? {}, null, 2)}
            </pre>
          </Card>
        </section>
      )}

      <footer style={{ marginTop: 18, opacity: 0.6, fontSize: 13 }}>
        API Base URL: <code>{API_BASE}</code>
      </footer>
    </main>
  );

  function mentions() {
    const examples = [
      "I want a low-sugar snack for kids, shaped like a rabbit.",
      "Make a snack for kids shaped like a star. No nuts.",
      "I want a snack shaped like a circle. 200-300 kcal. sugar 5g.",
      "Snack for kids, rabbit shape, soft texture, low sugar.",
      "Make a crunchy snack shaped like a dog. sugar-free.",
      "Make a snack for grandma, low sugar, soft texture, rabbit shape.",
      "Make a meal-regular for grandpa, low sugar, square shape, no nuts.",
      "Make a snack for a 75-year-old grandma, low activity, low sugar.",
    ];
    const next = examples[Math.floor(Math.random() * examples.length)];
    setPrompt(next);
  }
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div
      style={{
        border: "1px solid #e6e6e6",
        borderRadius: 16,
        padding: 14,
        background: "white",
      }}
    >
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
