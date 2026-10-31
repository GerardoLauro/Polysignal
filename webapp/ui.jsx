/* PolySignal — shared UI primitives ---------------------------------------- */

const DOMAIN_META = {
  MACRO: { label: 'Macro', dot: '#5b9dff' },
  POLITICS: { label: 'Politics', dot: '#c084fc' },
  CRYPTO: { label: 'Crypto', dot: '#f5a524' },
  AI: { label: 'AI', dot: '#34d399' },
  SPORTS: { label: 'Sports', dot: '#f76d6d' },
  CULTURE: { label: 'Culture', dot: '#f2a0c4' },
};

const STATUS_MAP = {
  ACTIVE: { c: '#34d399', bg: 'rgba(52,211,153,0.1)', label: 'ACTIVE' },
  CLOSING: { c: '#f5a524', bg: 'rgba(245,165,36,0.1)', label: 'CLOSING' },
  RESOLVED: { c: '#7d838a', bg: 'rgba(125,131,138,0.12)', label: 'RESOLVED' },
};

const EVIDENCE_BUCKETS = {
  High: { n: 3, c: '#34d399' },
  Moderate: { n: 2, c: '#f5a524' },
  Sparse: { n: 1, c: '#f76d6d' },
  None: { n: 0, c: '#5f656c' },
};

function DomainTag({ domain }) {
  const m = DOMAIN_META[domain] || { label: domain, dot: '#9aa0a6' };

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 6,
      font: "10.5px 'IBM Plex Mono', monospace",
      color: '#9aa0a6',
      letterSpacing: '0.04em'
    }}>
      <span style={{
        width: 6,
        height: 6,
        borderRadius: '50%',
        background: m.dot
      }} />
      {domain}
    </span>
  );
}

function StatusBadge({ status }) {
  const s = STATUS_MAP[status] || STATUS_MAP.ACTIVE;

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 5,
      font: "10px 'IBM Plex Mono', monospace",
      color: s.c,
      background: s.bg,
      padding: '2px 7px',
      borderRadius: 4,
      letterSpacing: '0.05em'
    }}>
      <span style={{
        width: 5,
        height: 5,
        borderRadius: '50%',
        background: s.c
      }} />
      {s.label}
    </span>
  );
}

function DeltaTag({ delta, accent = '#34d399', size = 'md' }) {
  const v = Math.round(delta * 100);
  const up = v >= 0;
  const c = up ? accent : '#f76d6d';
  const big = size === 'lg';

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 4,
      font: `600 ${big ? '15px' : '12px'} 'IBM Plex Mono', monospace`,
      color: c
    }}>
      <span style={{ fontSize: big ? 12 : 10 }}>
        {up ? '▲' : '▼'}
      </span>
      {up ? '+' : ''}{v}
      <span style={{
        color: '#5f656c',
        fontWeight: 400,
        fontSize: big ? 11 : 9.5
      }}>
        pts
      </span>
    </span>
  );
}

function EvidencePip({ bucket }) {
  const e = EVIDENCE_BUCKETS[bucket] || EVIDENCE_BUCKETS.Moderate;

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 5,
      font: "10.5px 'IBM Plex Mono', monospace",
      color: '#9aa0a6'
    }}>
      <span style={{ display: 'inline-flex', gap: 2 }}>
        {[0, 1, 2].map(i => (
          <span
            key={i}
            style={{
              width: 4,
              height: 11,
              borderRadius: 1,
              background: i < e.n ? e.c : '#26292d'
            }}
          />
        ))}
      </span>
      {bucket}
    </span>
  );
}

function DivergenceBadge({ delta, accent = '#34d399' }) {
  const v = Math.abs(Math.round(delta * 100));

  if (v < 5) return null;

  const high = v >= 10;
  const c = high ? '#f5a524' : accent;

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 5,
      font: "9.5px 'IBM Plex Mono', monospace",
      color: c,
      border: `1px solid ${c}33`,
      background: `${c}14`,
      padding: '2px 6px',
      borderRadius: 4,
      letterSpacing: '0.04em'
    }}>
      <span style={{
        width: 5,
        height: 5,
        borderRadius: '50%',
        background: c,
        boxShadow: `0 0 6px ${c}`
      }} />
      {high ? 'HIGH DIVERGENCE' : 'DIVERGENCE'}
    </span>
  );
}

function PressureChip({ value, dir, accent = '#34d399' }) {
  const toYes = dir === 'Toward YES';
  const c = dir === 'Balanced' ? '#9aa0a6' : toYes ? accent : '#f76d6d';

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 6,
      font: "11px 'IBM Plex Mono', monospace",
      color: '#9aa0a6'
    }}>
      <span style={{
        width: 44,
        height: 5,
        borderRadius: 3,
        background: '#1c1e21',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <span style={{
          position: 'absolute',
          left: 0,
          top: 0,
          bottom: 0,
          width: Math.round(value) + '%',
          background: c,
          opacity: 0.85
        }} />
      </span>
      <span style={{ color: c }}>
        {Math.round(value)}
      </span>
    </span>
  );
}

function OutcomeBadge({ outcome }) {
  const yes = outcome === 'YES';
  const c = yes ? '#34d399' : '#f76d6d';

  return (
    <span style={{
      font: "9.5px 'IBM Plex Mono', monospace",
      color: c,
      border: `1px solid ${c}44`,
      background: `${c}14`,
      padding: '2px 7px',
      borderRadius: 4,
      letterSpacing: '0.05em'
    }}>
      ● {outcome}
    </span>
  );
}

function HitMark({ ok, label }) {
  const c = ok ? '#34d399' : '#f76d6d';

  return (
    <span
      title={ok ? 'called it' : 'missed'}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 4,
        font: "11px 'IBM Plex Mono', monospace",
        color: c
      }}
    >
      <span>{ok ? '✓' : '✗'}</span>
      {label}
    </span>
  );
}

function ScorecardBanner({ scorecard, accent }) {
  if (!scorecard || !scorecard.n) return null;

  const s = scorecard;
  const sigBetter = s.signalBrier < s.marketBrier;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'stretch',
      border: '1px solid var(--border)',
      borderRadius: 10,
      background: 'var(--panel)',
      overflow: 'hidden',
      margin: '14px 22px 0',
      flexWrap: 'wrap'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        padding: '12px 16px',
        borderRight: '1px solid var(--border)'
      }}>
        <span style={{
          font: "9.5px 'IBM Plex Mono', monospace",
          color: '#5f656c',
          letterSpacing: '0.07em',
          textTransform: 'uppercase',
          lineHeight: 1.35
        }}>
          Descriptive<br />in-sample
        </span>
      </div>

      <ScoreCell
        label="Order book"
        value={`${s.marketCorrect}/${s.n}`}
        sub={`Brier ${s.marketBrier.toFixed(2)}`}
        good={!sigBetter}
        accent={accent}
      />
      <ScoreCell
        label="Social est."
        value={`${s.signalCorrect}/${s.n}`}
        sub={`Brier ${s.signalBrier.toFixed(2)}`}
        good={sigBetter}
        accent={accent}
      />
      <ScoreCell
        label="Social closer"
        value={`${s.signalWins}/${s.n}`}
        sub={s.signalWins ? 'markets' : 'no market'}
        accent={accent}
      />

      <div style={{
        flex: 1,
        minWidth: 240,
        display: 'flex',
        alignItems: 'center',
        padding: '10px 16px',
        font: "11px/1.5 'IBM Plex Sans', sans-serif",
        color: '#9aa0a6'
      }}>
        {s.verdict}
      </div>
    </div>
  );
}

/* Out-of-fold predictive benchmark (Task-3 temporal CV). This is the *headline
   predictive claim* — leakage-free, outcome-fitted, held-out — and is deliberately
   kept separate from the in-sample descriptive ScorecardBanner above it. */
function PredictiveBenchmarkBanner({ benchmark, accent }) {
  if (!benchmark || !benchmark.available) return null;

  const b = benchmark;
  const ab = b.ablations || {};
  const bestAbl = b.best && b.best.ablation;
  const rows = [
    ['market_only', 'Market only'],
    ['social_only', 'Social only'],
    ['market_social', 'Market+social'],
  ];

  return (
    <div style={{
      display: 'flex',
      alignItems: 'stretch',
      border: '1px solid var(--border)',
      borderRadius: 10,
      background: 'var(--panel)',
      overflow: 'hidden',
      margin: '14px 22px 0',
      flexWrap: 'wrap'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        padding: '12px 16px',
        borderRight: '1px solid var(--border)'
      }}>
        <span style={{
          font: "9.5px 'IBM Plex Mono', monospace",
          color: accent,
          letterSpacing: '0.07em',
          textTransform: 'uppercase',
          lineHeight: 1.35
        }}>
          Predictive<br />out-of-fold
        </span>
      </div>

      {rows.map(([key, label]) => {
        const m = ab[key];
        if (!m) return null;
        return (
          <ScoreCell
            key={key}
            label={label}
            value={`AUC ${Number(m.auc).toFixed(2)}`}
            sub={`Brier ${Number(m.brier).toFixed(3)}`}
            good={key === bestAbl}
            accent={accent}
          />
        );
      })}

      <div style={{
        flex: 1,
        minWidth: 240,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        padding: '10px 16px',
        font: "11px/1.5 'IBM Plex Sans', sans-serif",
        color: '#9aa0a6'
      }}>
        <span>{b.conclusion}</span>
        <span style={{ font: "9.5px 'IBM Plex Mono', monospace", color: '#5f656c', marginTop: 4 }}>
          {b.scheme} · {b.nFolds} folds · {b.nContracts} contracts ({b.nWithSocialEvidence} with social evidence) · leakage-free headline result
        </span>
      </div>
    </div>
  );
}

function ScoreCell({ label, value, sub, good, accent }) {
  return (
    <div style={{
      padding: '10px 16px',
      borderRight: '1px solid var(--border)',
      minWidth: 92
    }}>
      <div style={{
        font: "9px 'IBM Plex Mono', monospace",
        color: '#5f656c',
        letterSpacing: '0.05em',
        textTransform: 'uppercase',
        marginBottom: 4
      }}>
        {label}
      </div>
      <div style={{
        font: "600 17px 'IBM Plex Mono', monospace",
        color: good ? (accent || '#34d399') : '#c4c8cc'
      }}>
        {value}
      </div>
      <div style={{
        font: "9.5px 'IBM Plex Mono', monospace",
        color: '#5f656c',
        marginTop: 2
      }}>
        {sub}
      </div>
    </div>
  );
}

function Panel({ title, sub, right, children, style, pad = 18 }) {
  return (
    <section style={{
      background: 'var(--panel)',
      border: '1px solid var(--border)',
      borderRadius: 10,
      ...style
    }}>
      {title && (
        <header style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: `13px ${pad}px`,
          borderBottom: '1px solid var(--border)'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'baseline',
            gap: 10
          }}>
            <h3 style={{
              margin: 0,
              font: "600 12px 'IBM Plex Mono', monospace",
              letterSpacing: '0.08em',
              color: '#c4c8cc',
              textTransform: 'uppercase'
            }}>
              {title}
            </h3>
            {sub && (
              <span style={{
                font: "10.5px 'IBM Plex Mono', monospace",
                color: '#5f656c'
              }}>
                {sub}
              </span>
            )}
          </div>
          {right}
        </header>
      )}
      <div style={{ padding: pad }}>
        {children}
      </div>
    </section>
  );
}

function Stat({ label, children, sub, accent }) {
  return (
    <div>
      <div style={{
        font: "10px 'IBM Plex Mono', monospace",
        color: '#5f656c',
        letterSpacing: '0.05em',
        marginBottom: 5,
        textTransform: 'uppercase'
      }}>
        {label}
      </div>
      <div style={{
        font: "600 18px 'IBM Plex Mono', monospace",
        color: accent || '#e8eaed',
        lineHeight: 1.1
      }}>
        {children}
      </div>
      {sub && (
        <div style={{
          font: "10.5px 'IBM Plex Mono', monospace",
          color: '#5f656c',
          marginTop: 3
        }}>
          {sub}
        </div>
      )}
    </div>
  );
}

Object.assign(window, {
  DOMAIN_META,
  DomainTag,
  StatusBadge,
  DeltaTag,
  EvidencePip,
  DivergenceBadge,
  PressureChip,
  Panel,
  Stat,
  OutcomeBadge,
  HitMark,
  ScorecardBanner,
  PredictiveBenchmarkBanner
});
