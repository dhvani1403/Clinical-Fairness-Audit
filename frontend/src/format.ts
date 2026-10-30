import type { Metric } from "./types";

export const pct = (x: number, digits = 1) => `${(100 * x).toFixed(digits)}%`;
export const pp = (x: number) => {
  const v = Math.abs(100 * x).toFixed(1);
  return v === "0.0" ? "0.0 pp" : `${x > 0 ? "+" : "−"}${v} pp`;
};
export const count = (n: number) => n.toLocaleString("en-US");

export const METRICS: { id: Metric; name: string; help: string }[] = [
  { id: "fnr", name: "Missed patients", help: "Of the patients who had the outcome, the share the model did not flag." },
  { id: "fpr", name: "False alarms", help: "Of the patients who did not have the outcome, the share the model flagged." },
  { id: "calibration", name: "Calibration", help: "Average predicted risk minus the observed rate. Needs risk scores." },
];

/** What the base population of each metric is called in a table header. */
export const BASE_NOUN: Record<Metric, string> = {
  fnr: "Patients with outcome",
  fpr: "Patients without outcome",
  calibration: "Patients",
  positive_rate: "Patients",
  error_rate: "Patients",
};

export const RATE_NOUN: Record<Metric, string> = {
  fnr: "Missed",
  fpr: "False alarms",
  calibration: "Predicted − observed",
  positive_rate: "Flagged",
  error_rate: "Errors",
};

/** Plain-English name for a fact the summary's numbers were matched to. */
export function factName(path: string): string {
  const where = path.startsWith("worst_single") ? "worst single attribute"
    : path.startsWith("worst_intersectional") ? "worst combination"
    : path.startsWith("top_intersectional") ? `combination ranked ${Number(path.match(/\[(\d+)\]/)?.[1] ?? 0) + 1}`
    : "";
  const what: Record<string, string> = {
    rate_pct: "rate", gap_pct_points: "difference from overall", rows_in_metric_base: "patients",
    share_of_data_pct: "share of patients", overall_metric_pct: "overall rate", overall_accuracy_pct: "accuracy",
    rows: "patients audited", flagged_top_pct: "programme capacity", subgroups_checked: "groups checked",
    min_rows_for_reliable_gap: "minimum group size", intersection_vs_single_gap_ratio: "combination vs single ratio",
    largest_size_weighted_gap_pct_points: "largest size-weighted gap",
  };
  const leaf = path.split(".").pop() ?? path;
  return [where, what[leaf] ?? leaf].filter(Boolean).join(", ");
}
