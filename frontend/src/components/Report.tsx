import { useState } from "react";
import type { AuditResult } from "../types";
import { downloadPdf } from "../api";
import { METRICS, count, pct, pp } from "../format";
import Interpretation from "./Interpretation";
import ResultTable from "./ResultTable";
import { EcgDefs } from "./Strip";

/** Natural order within an attribute: "<40" before "40-59" before "80+"; words alphabetically. */
function naturalKey(subgroup: string): [string, number, string] {
  const [attr, value = ""] = subgroup.split("=");
  const n = value.startsWith("<") ? -1 : parseInt(value, 10);
  return [attr, Number.isNaN(n) ? Infinity : n, value];
}

function bySubgroup(a: { subgroup: string }, b: { subgroup: string }, attrs: string[]) {
  const [aa, an, av] = naturalKey(a.subgroup);
  const [ba, bn, bv] = naturalKey(b.subgroup);
  return attrs.indexOf(aa) - attrs.indexOf(ba) || an - bn || av.localeCompare(bv);
}

export default function Report({ result, dataName }: { result: AuditResult; dataName: string }) {
  const [pdfError, setPdfError] = useState<string | null>(null);
  const r = result;
  const singles = r.single_attribute.filter((x) => x.reliable).sort((a, b) => bySubgroup(a, b, r.protected_attributes));
  const combos = r.top_raw_gap.filter((x) => x.order >= 2).slice(0, 8);
  const all = [...singles, ...combos];
  const extent = Math.max(...all.map((x) => Math.abs(x.gap) + 1.96 * x.std_err), 0.05);
  const domain = Math.ceil(extent * 10) / 10;
  const measure = METRICS.find((m) => m.id === r.metric)?.name ?? r.metric_label;
  const nSig = all.filter((x) => x.significant).length;
  const signed = r.metric === "calibration";

  return (
    <article className="report" key={JSON.stringify([r.metric, r.n_rows, r.flag_top_pct, r.max_order])}>
      <EcgDefs />
      <header className="report-head">
        <h2>Audit report</h2>
        <dl className="specimen">
          <div><dt>Data</dt><dd>{dataName}</dd></div>
          <div><dt>Patients</dt><dd>{count(r.n_rows)}</dd></div>
          <div><dt>Measure</dt><dd>{measure} ({r.metric_label})</dd></div>
          <div>
            <dt>Flagging</dt>
            <dd>{r.flag_top_pct ? `Top ${r.flag_top_pct}% by risk score` : signed ? "Raw risk scores" : "Predictions as given"}</dd>
          </div>
          <div><dt>Groups checked</dt><dd>{r.n_subgroups_checked}, up to {r.max_order} attributes combined</dd></div>
          <div><dt>Smallest group</dt><dd>{r.min_size} patients</dd></div>
        </dl>
      </header>

      {r.warnings.map((w) => <div key={w} className="alert alert-soft">{w}</div>)}

      <section className="key" aria-label="Key results">
        <div>
          <span className="key-value">{signed ? pp(r.overall_rate) : pct(r.overall_rate)}</span>
          <span className="key-label">overall, all patients</span>
        </div>
        {r.worst_single && (
          <div>
            <span className="key-value">{pp(r.worst_single.gap)}</span>
            <span className="key-label">largest single-attribute gap: {r.worst_single.label}</span>
          </div>
        )}
        {r.worst_intersection && (
          <div>
            <span className="key-value key-combo">{pp(r.worst_intersection.gap)}</span>
            <span className="key-label">
              largest combination gap: {r.worst_intersection.label}
              {r.gerrymandering_ratio ? `, ${r.gerrymandering_ratio.toFixed(1)}× the single-attribute gap` : ""}
            </span>
          </div>
        )}
      </section>

      <Interpretation summary={r.summary} />

      <section className="tables" aria-label="Results by group">
        <ResultTable caption="One attribute at a time" rows={singles} metric={r.metric}
          overall={r.overall_rate} domain={domain} tone="single" />
        <ResultTable caption="Combinations with the largest gaps" rows={combos} metric={r.metric}
          overall={r.overall_rate} domain={domain} tone="combo" />
        <p className="legend">
          Bars show each group's difference from the overall rate; whiskers are 95% intervals. H and L mark
          differences that remain significant after correcting for all {r.n_subgroups_checked} groups tested
          (Benjamini–Hochberg, 5% false discovery rate). {nSig} of the {all.length} groups shown are flagged.
          Only groups with at least {r.min_size} patients are shown.
        </p>
      </section>

      <footer className="report-foot">
        <button type="button" className="secondary" onClick={() => downloadPdf(r).catch((e) => setPdfError(e.message))}>
          Download PDF report
        </button>
        {pdfError && <span className="alert-inline">{pdfError}</span>}
        <p className="method">
          Method: Kearns, Neel, Roth and Wu (2018), Preventing Fairness Gerrymandering. A difference is evidence to
          investigate, not proof of discrimination. Recorded outcomes such as visits or readmissions are proxies for
          need and can carry their own bias.
        </p>
      </footer>
    </article>
  );
}
