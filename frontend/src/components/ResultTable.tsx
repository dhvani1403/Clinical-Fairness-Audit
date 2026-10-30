import type { Metric, Row } from "../types";
import { BASE_NOUN, RATE_NOUN, count, pct, pp } from "../format";
import Strip from "./Strip";

interface Props {
  caption: string;
  rows: Row[];
  metric: Metric;
  overall: number;
  domain: number;
  tone: "single" | "combo";
}

function flag(r: Row) {
  if (!r.significant) return null;
  const high = r.gap > 0;
  return (
    <abbr className={`flag ${high ? "flag-high" : "flag-low"}`} title={`Significantly ${high ? "higher" : "lower"} than overall (q = ${r.q_value.toFixed(3)})`}>
      {high ? "H" : "L"}
    </abbr>
  );
}

export default function ResultTable({ caption, rows, metric, overall, domain, tone }: Props) {
  const signed = metric === "calibration";
  return (
    <table className="results">
      <caption>
        {caption}
        <span className="caption-key">
          {RATE_NOUN[metric]}, then difference from overall ({signed ? pp(overall) : pct(overall)}),
          scale ±{(100 * domain).toFixed(0)} pp
        </span>
      </caption>
      <thead>
        <tr>
          <th scope="col">Group</th>
          <th scope="col" className="num">{BASE_NOUN[metric]}</th>
          <th scope="col" className="num">{RATE_NOUN[metric]}</th>
          <th scope="col" className="diff-head">
            Difference from overall ({signed ? pp(overall) : pct(overall)})
            <span className="scale">
              <span>−{(100 * domain).toFixed(0)}</span>
              <span>0</span>
              <span>+{(100 * domain).toFixed(0)} pp</span>
            </span>
          </th>
          <th scope="col" className="flag-col"><span className="visually-hidden">Flag</span></th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.subgroup} className={r.reliable ? "" : "is-small"}>
            <th scope="row" title={r.subgroup}>{r.label.charAt(0).toUpperCase() + r.label.slice(1)}</th>
            <td className="num">{count(r.base_size)}</td>
            <td className="num">{signed ? pp(r.rate) : pct(r.rate)}</td>
            <td className="diff">
              <Strip gap={r.gap} se={r.std_err} domain={domain} tone={tone} />
              <span className="diff-value">
                {pp(r.gap)} <span className="ci">± {(196 * r.std_err).toFixed(1)}</span>
              </span>
            </td>
            <td className="flag-col">{flag(r)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
