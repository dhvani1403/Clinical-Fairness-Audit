import type { Example } from "../types";

interface Props {
  busy: boolean;
  hasData: boolean;
  examples: Example[];
  onExample: (ex: Example) => void;
}

export default function EmptyReport({ busy, hasData, examples, onExample }: Props) {
  if (busy)
    return (
      <div className="empty">
        <p className="empty-lead">Searching every group and combination…</p>
        <p className="hint">The audit itself takes a second or two. Free AI models can take up to a minute to write
          the interpretation; untick that option for an instant report.</p>
      </div>
    );
  return (
    <div className="empty">
      <p className="empty-lead">
        A risk score can look fair for women, for Black patients and for older patients taken one at a time, and
        still miss young Black women. This audit checks every combination.
      </p>
      <ol className="empty-steps">
        <li>Give it one row per patient: the outcome, the model's risk score, and attributes such as sex, race and age.</li>
        <li>Set the programme capacity: how many patients the score is used to flag.</li>
        <li>Read which groups are missed, falsely flagged or mis-calibrated, with intervals and a check for chance findings.</li>
      </ol>
      {!hasData && examples[0] && (
        <button type="button" className="secondary" onClick={() => onExample(examples[0])}>
          Load the {examples[0].title.split(":")[0]} example
        </button>
      )}
    </div>
  );
}
