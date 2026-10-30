import { useRef, useState } from "react";
import type { Loaded } from "../App";
import type { Example, Settings } from "../types";
import { METRICS, count } from "../format";

interface Props {
  examples: Example[];
  exampleId: string | null;
  data: Loaded | null;
  settings: Settings | null;
  scoreIsProbability: boolean;
  busy: boolean;
  onExample: (ex: Example) => void;
  onFile: (f: File) => void;
  onChange: (s: Settings) => void;
  onSubmit: () => void;
}

export default function Requisition(p: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const s = p.settings;
  const set = (patch: Partial<Settings>) => s && p.onChange({ ...s, ...patch });
  const others = p.data?.columns.filter((c) => s && c !== s.labelCol && c !== s.scoreCol) ?? [];
  const canRun = !!(p.data && s && s.protectedCols.length && s.labelCol !== s.scoreCol) && !p.busy;

  return (
    <aside className="requisition" aria-label="Audit settings">
      <h2 className="req-title">Request an audit</h2>

      <fieldset className="req-block">
        <legend>Patient data</legend>
        {p.examples.length > 0 && (
          <div className="examples" role="radiogroup" aria-label="Example datasets">
            {p.examples.map((ex) => (
              <button
                key={ex.id}
                type="button"
                role="radio"
                aria-checked={p.exampleId === ex.id}
                className="example"
                onClick={() => p.onExample(ex)}
              >
                <strong>{ex.title}</strong>
                <span>{ex.description}</span>
              </button>
            ))}
          </div>
        )}
        <div
          className={`dropzone${dragging ? " is-dragging" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const f = e.dataTransfer.files[0];
            if (f) p.onFile(f);
          }}
        >
          <button type="button" className="link-button" onClick={() => input.current?.click()}>
            Upload your own CSV
          </button>
          <span> or drop it here. One row per patient, de-identified. Files are analysed in memory and not stored.</span>
          <input
            ref={input}
            type="file"
            accept=".csv,text/csv"
            hidden
            onChange={(e) => e.target.files?.[0] && p.onFile(e.target.files[0])}
          />
        </div>
        {p.data && (
          <p className="loaded">
            Loaded <strong>{p.data.name}</strong>: {p.data.columns.length} columns
            {p.data.rows >= 2000 ? ", previewed first 2,000 rows" : `, ${count(p.data.rows)} rows`}.
          </p>
        )}
      </fieldset>

      {p.data && s && (
        <>
          <fieldset className="req-block">
            <legend>Columns</legend>
            <label className="field">
              <span>Observed outcome (0/1)</span>
              <select value={s.labelCol} onChange={(e) => set({ labelCol: e.target.value })}>
                {p.data.columns.map((c) => <option key={c}>{c}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Risk score or 0/1 prediction</span>
              <select value={s.scoreCol} onChange={(e) => set({ scoreCol: e.target.value })}>
                {p.data.columns.map((c) => <option key={c}>{c}</option>)}
              </select>
            </label>
            <div className="field">
              <span id="protected-label">Attributes to check</span>
              <div className="checks" role="group" aria-labelledby="protected-label">
                {others.map((c) => (
                  <label key={c} className="check">
                    <input
                      type="checkbox"
                      checked={s.protectedCols.includes(c)}
                      onChange={(e) =>
                        set({
                          protectedCols: e.target.checked
                            ? [...s.protectedCols, c]
                            : s.protectedCols.filter((x) => x !== c),
                        })
                      }
                    />
                    {c}
                  </label>
                ))}
              </div>
            </div>
          </fieldset>

          <fieldset className="req-block">
            <legend>What to measure</legend>
            <div className="segmented" role="radiogroup" aria-label="Measure">
              {METRICS.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  role="radio"
                  aria-checked={s.metric === m.id}
                  onClick={() => set({ metric: m.id })}
                >
                  {m.name}
                </button>
              ))}
            </div>
            <p className="hint">{METRICS.find((m) => m.id === s.metric)?.help}</p>

            {p.scoreIsProbability && s.metric !== "calibration" && (
              <label className="field">
                <span>
                  Programme capacity: flag the top <output>{s.capacity}%</output> by risk
                </span>
                <input
                  type="range"
                  min={1}
                  max={50}
                  value={s.capacity}
                  onChange={(e) => set({ capacity: Number(e.target.value) })}
                />
              </label>
            )}

            <div className="field-row">
              <label className="field">
                <span>Combine up to</span>
                <select
                  value={s.maxOrder}
                  onChange={(e) => set({ maxOrder: Number(e.target.value) as 1 | 2 | 3 })}
                >
                  <option value={1}>1 attribute</option>
                  <option value={2}>2 attributes</option>
                  <option value={3}>3 attributes</option>
                </select>
              </label>
              <label className="field">
                <span>Smallest group reported</span>
                <input
                  type="number"
                  min={10}
                  max={1000}
                  step={5}
                  value={s.minSize}
                  onChange={(e) => set({ minSize: Number(e.target.value) || 30 })}
                />
              </label>
            </div>
            <label className="check">
              <input type="checkbox" checked={s.useLlm} onChange={(e) => set({ useLlm: e.target.checked })} />
              Write a plain-English interpretation with an AI model
            </label>
          </fieldset>

          <button type="button" className="run" disabled={!canRun} onClick={p.onSubmit}>
            {p.busy ? "Running audit…" : "Run audit"}
          </button>
          {!s.protectedCols.length && <p className="hint">Choose at least one attribute to check.</p>}
        </>
      )}
    </aside>
  );
}
