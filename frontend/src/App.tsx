import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import { fetchExample, listExamples, runAudit } from "./api";
import type { AuditResult, Example, Settings } from "./types";
import Requisition from "./components/Requisition";
import Report from "./components/Report";
import EmptyReport from "./components/EmptyReport";

export interface Loaded {
  file: File;
  name: string;
  columns: string[];
  rows: number;
  numericCols: Set<string>;
  probabilityCols: Set<string>;
}

const LIKELY_PROTECTED = ["sex", "gender", "race", "ethnicity", "age_band", "age"];

function inspect(file: File, name: string): Promise<Loaded> {
  return new Promise((resolve, reject) => {
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      preview: 2000,
      complete: (res) => {
        const columns = res.meta.fields ?? [];
        if (!columns.length) return reject(new Error("The file has no header row."));
        const numericCols = new Set<string>();
        const probabilityCols = new Set<string>();
        for (const c of columns) {
          const vals = res.data.map((r) => r[c]).filter((v) => v !== "" && v != null);
          if (vals.length && vals.every((v) => !Number.isNaN(Number(v)))) {
            numericCols.add(c);
            const nums = vals.map(Number);
            if (nums.every((v) => v >= 0 && v <= 1) && nums.some((v) => v !== 0 && v !== 1)) probabilityCols.add(c);
          }
        }
        resolve({ file, name, columns, rows: res.data.length, numericCols, probabilityCols });
      },
      error: (err) => reject(err),
    });
  });
}

function defaults(d: Loaded, example?: Example): Settings {
  const cols = d.columns;
  const labelCol = example?.columns.label ?? cols.find((c) => /label|outcome|readmit|target|^y$/i.test(c)) ?? cols[0];
  const scoreCol = example?.columns.score ?? [...d.probabilityCols][0]
    ?? cols.find((c) => /score|risk|prob|pred/i.test(c) && c !== labelCol) ?? cols[1] ?? cols[0];
  const protectedCols = example?.columns.protected
    ?? cols.filter((c) => LIKELY_PROTECTED.includes(c.toLowerCase())).slice(0, 3);
  return { labelCol, scoreCol, protectedCols, metric: "fnr", capacity: 20, maxOrder: 3, minSize: 30, useLlm: true };
}

export default function App() {
  const [examples, setExamples] = useState<Example[]>([]);
  const [data, setData] = useState<Loaded | null>(null);
  const [exampleId, setExampleId] = useState<string | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [result, setResult] = useState<AuditResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listExamples().then(setExamples).catch(() => setExamples([]));
  }, []);

  const scoreIsProbability = useMemo(
    () => !!(data && settings && data.probabilityCols.has(settings.scoreCol)),
    [data, settings],
  );

  async function load(file: File, name: string, example?: Example) {
    setError(null);
    try {
      const d = await inspect(file, name);
      setData(d);
      setExampleId(example?.id ?? null);
      setSettings(defaults(d, example));
      setResult(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function loadExample(ex: Example) {
    setBusy(true);
    try {
      await load(await fetchExample(ex.id), ex.title, ex);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function submit() {
    if (!data || !settings) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await runAudit(data.file, settings, scoreIsProbability));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header className="masthead">
        <div className="brand">
          <svg viewBox="0 0 32 32" aria-hidden="true" className="brand-mark">
            <rect width="32" height="32" rx="6" />
            <path d="M4 17h7l3-8 4 15 3-7h7" />
          </svg>
          <span>Clinical Fairness Audit</span>
        </div>
        <p className="masthead-note">
          Finds the patient groups a risk model under-serves, including combinations of attributes that
          one-at-a-time checks miss.
        </p>
      </header>
      <main className="layout">
        <Requisition
          examples={examples}
          exampleId={exampleId}
          data={data}
          settings={settings}
          scoreIsProbability={scoreIsProbability}
          busy={busy}
          onExample={loadExample}
          onFile={(f) => load(f, f.name)}
          onChange={(s) => setSettings(s)}
          onSubmit={submit}
        />
        <section className="report-column" aria-live="polite">
          {error && <div className="alert" role="alert">{error}</div>}
          {result && data ? (
            <Report result={result} dataName={data.name} />
          ) : (
            <EmptyReport busy={busy} hasData={!!data} examples={examples} onExample={loadExample} />
          )}
        </section>
      </main>
    </div>
  );
}
