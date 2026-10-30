/** A difference-from-overall bar with its 95% interval, drawn on ECG-paper grid. */
interface Props {
  gap: number;
  se: number;
  domain: number;
  tone: "single" | "combo";
}

const W = 220;
const H = 22;

export default function Strip({ gap, se, domain, tone }: Props) {
  const x = (v: number) => W / 2 + (Math.max(-domain, Math.min(domain, v)) / domain) * (W / 2);
  const lo = x(gap - 1.96 * se);
  const hi = x(gap + 1.96 * se);
  const end = x(gap);
  return (
    <svg className={`strip strip-${tone}`} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
      <rect width={W} height={H} fill="url(#ecg-minor)" />
      <rect width={W} height={H} fill="url(#ecg-major)" />
      <line x1={W / 2} x2={W / 2} y1={0} y2={H} className="strip-zero" />
      <rect
        className="strip-bar"
        x={Math.min(end, W / 2)}
        y={6}
        width={Math.max(Math.abs(end - W / 2), 1)}
        height={10}
        style={{ transformOrigin: `${W / 2}px 11px` }}
      />
      <line x1={lo} x2={hi} y1={11} y2={11} className="strip-ci" />
      <line x1={lo} x2={lo} y1={7} y2={15} className="strip-ci" />
      <line x1={hi} x2={hi} y1={7} y2={15} className="strip-ci" />
    </svg>
  );
}

/** Pattern definitions shared by every strip; rendered once per report. */
export function EcgDefs() {
  return (
    <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden="true">
      <defs>
        <pattern id="ecg-minor" width="4.4" height="4.4" patternUnits="userSpaceOnUse">
          <path d="M4.4 0V4.4H0" fill="none" className="ecg-minor" />
        </pattern>
        <pattern id="ecg-major" width="22" height="22" patternUnits="userSpaceOnUse">
          <path d="M22 0V22H0" fill="none" className="ecg-major" />
        </pattern>
      </defs>
    </svg>
  );
}
