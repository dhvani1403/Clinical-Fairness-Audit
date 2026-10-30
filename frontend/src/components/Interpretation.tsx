import type { Summary } from "../types";
import { factName } from "../format";

export default function Interpretation({ summary }: { summary: Summary }) {
  const text = summary.text;
  const matches = [...(summary.matches ?? [])].sort((a, b) => a.start - b.start);
  const parts: (string | JSX.Element)[] = [];
  let at = 0;
  matches.forEach((m, i) => {
    parts.push(text.slice(at, m.start));
    parts.push(
      <mark key={i} className="checked" title={`Checked: ${factName(m.fact)}`}>
        {text.slice(m.start, m.end)}
      </mark>,
    );
    at = m.end;
  });
  parts.push(text.slice(at));

  const model = summary.model?.split("/").pop()?.replace(":free", "");
  return (
    <section className="interpretation" aria-labelledby="interp-h">
      <h3 id="interp-h">Interpretation</h3>
      <p className="interp-text">{parts}</p>
      <p className="provenance">
        {summary.source === "llm" ? (
          <>
            Written by {model}. All {matches.length} figures were checked against the computed results and
            the group they describe; hover an underlined figure to see its source.
          </>
        ) : (
          <>
            Written from a fixed template{summary.note ? `. ${summary.note}` : ""}. Underlined figures are taken
            directly from the results.
          </>
        )}
      </p>
    </section>
  );
}
