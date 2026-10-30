export type Metric = "fnr" | "fpr" | "calibration" | "positive_rate" | "error_rate";

export interface Row {
  subgroup: string;
  label: string;
  order: number;
  size: number;
  share: number;
  base_size: number;
  rate: number;
  gap: number;
  weighted_gap: number;
  std_err: number;
  z: number;
  reliable: boolean;
  p_value: number;
  q_value: number;
  significant: boolean;
}

export interface Match {
  text: string;
  start: number;
  end: number;
  fact: string;
}

export interface Summary {
  text: string;
  source: "llm" | "template";
  model: string | null;
  verified: boolean;
  note: string | null;
  matches?: Match[];
  unverified_numbers: string[];
  unbound_numbers: string[];
}

export interface AuditResult {
  metric: Metric;
  metric_label: string;
  n_rows: number;
  max_order: number;
  min_size: number;
  n_subgroups_checked: number;
  overall: Record<string, number>;
  overall_rate: number;
  single_attribute: Row[];
  top_weighted: Row[];
  top_raw_gap: Row[];
  worst_single: Row | null;
  worst_intersection: Row | null;
  gerrymandering_ratio: number | null;
  max_weighted_gap: number;
  protected_attributes: string[];
  flag_top_pct: number | null;
  warnings: string[];
  summary: Summary;
}

export interface Example {
  id: string;
  title: string;
  description: string;
  columns: { label: string; score: string; protected: string[] };
}

export interface Settings {
  labelCol: string;
  scoreCol: string;
  protectedCols: string[];
  metric: Metric;
  capacity: number;
  maxOrder: 1 | 2 | 3;
  minSize: number;
  useLlm: boolean;
}
