import type { AuditResult, Example, Settings } from "./types";

async function errorText(r: Response): Promise<string> {
  try {
    const body = await r.json();
    return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    return `The server returned ${r.status}.`;
  }
}

export async function listExamples(): Promise<Example[]> {
  const r = await fetch("/api/examples");
  return r.ok ? r.json() : [];
}

export async function fetchExample(id: string): Promise<File> {
  const r = await fetch(`/api/examples/${id}.csv`);
  if (!r.ok) throw new Error(await errorText(r));
  return new File([await r.blob()], `${id}.csv`, { type: "text/csv" });
}

export async function runAudit(file: File, s: Settings, scoreIsProbability: boolean): Promise<AuditResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("label_col", s.labelCol);
  form.append("pred_col", s.scoreCol);
  form.append("protected_cols", s.protectedCols.join(","));
  form.append("metric", s.metric);
  form.append("max_order", String(s.maxOrder));
  form.append("min_size", String(s.minSize));
  form.append("use_llm", String(s.useLlm));
  if (scoreIsProbability && s.metric !== "calibration") form.append("flag_top_pct", String(s.capacity));
  const r = await fetch("/api/audit", { method: "POST", body: form });
  if (!r.ok) throw new Error(await errorText(r));
  return r.json();
}

export async function downloadPdf(result: AuditResult): Promise<void> {
  const r = await fetch("/api/report/pdf", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ result }),
  });
  if (!r.ok) throw new Error(await errorText(r));
  const url = URL.createObjectURL(await r.blob());
  const a = Object.assign(document.createElement("a"), { href: url, download: "fairness-audit.pdf" });
  a.click();
  URL.revokeObjectURL(url);
}
