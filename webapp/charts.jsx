/* PolySignal — chart + viz components -------------------------------------- */
const { useRef, useState, useEffect, useLayoutEffect, useMemo } = React;

/**
 * Hook to measure a container's width responsively
 * @returns {[React.Ref, number]} - Ref to attach and current width
 */
function useMeasure() {
  const ref = useRef(null);
  const [w, setW] = useState(0);

  useLayoutEffect(() => {
    if (!ref.current) return;

    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        setW(e.contentRect.width);
      }
    });

    ro.observe(ref.current);
    setW(ref.current.clientWidth);

    return () => ro.disconnect();
  }, []);

  return [ref, w];
}

function pct(n) {
  return Math.round(n * 100);
}

function fmtSignedPts(n) {
  const v = Math.round(n * 100);
  return (v > 0 ? '+' : '') + v + ' pts';
}

/* ---- Sparkline (used in universe table/cards) --------------------------- */
function Sparkline({ poly, signal, w = 120, h = 30, accent = '#34d399' }) {
  const pad = 2;
  const n = poly.length;
  const all = poly.concat(signal);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const rng = hi - lo || 1;

  const x = (i) => pad + (i / (n - 1)) * (w - pad * 2);
  const y = (v) => pad + (1 - (v - lo) / rng) * (h - pad * 2);

  const path = (arr) => arr.map((v, i) =>
    `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`
  ).join(' ');

  return (
    <svg width={w} height={h} style={{ display: 'block', overflow: 'visible' }}>
      <path
        d={path(poly)}
        fill="none"
        stroke="rgba(255,255,255,0.28)"
        strokeWidth="1.25"
      />
      <path
        d={path(signal)}
        fill="none"
        stroke={accent}
        strokeWidth="1.5"
      />
      <circle cx={x(n - 1)} cy={y(signal[n - 1])} r="2" fill={accent} />
    </svg>
  );
}

/* ---- Main dual-line chart with pressure overlay + events ---------------- */
function DualLineChart({
  market,
  height = 320,
  showPressure = true,
  area = false,
  accent = '#34d399'
}) {
  const [ref, w] = useMeasure();
  const [hover, setHover] = useState(null);
  const { poly, signal, pressure } = market.series;
  const n = poly.length;

  const padL = 44;
  const padR = 16;
  const padT = 16;
  const padB = 28;
  const plotW = Math.max(10, w - padL - padR);
  const pressH = showPressure ? 54 : 0;
  const lineH = height - padT - padB - pressH - (showPressure ? 10 : 0);

  const x = (i) => padL + (i / (n - 1)) * plotW;
  const yProb = (v) => padT + (1 - v) * lineH;
  const yPress = (v) => padT + lineH + 10 + (1 - v / 100) * pressH;

  const linePath = (arr) => arr.map((v, i) =>
    `${i ? 'L' : 'M'}${x(i).toFixed(1)},${yProb(v).toFixed(1)}`
  ).join(' ');

  const areaPath = (arr) =>
    `${linePath(arr)} L${x(n - 1).toFixed(1)},${(padT + lineH).toFixed(1)} L${padL.toFixed(1)},${(padT + lineH).toFixed(1)} Z`;

  const pressArea = `M${padL},${padT + lineH + 10 + pressH} ` +
    pressure.map((v, i) => `L${x(i).toFixed(1)},${yPress(v).toFixed(1)}`).join(' ') +
    ` L${x(n - 1)},${padT + lineH + 10 + pressH} Z`;

  function onMove(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    let i = Math.round(((mx - padL) / plotW) * (n - 1));
    i = Math.max(0, Math.min(n - 1, i));
    setHover(i);
  }

  const gridlines = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div ref={ref} style={{ width: '100%', position: 'relative' }}>
      {w > 0 && (
        <svg
          width={w}
          height={height}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
          style={{ display: 'block' }}
        >
          {/* Horizontal gridlines + probability axis */}
          {gridlines.map((g) => (
            <g key={g}>
              <line
                x1={padL}
                x2={w - padR}
                y1={yProb(g)}
                y2={yProb(g)}
                stroke="rgba(255,255,255,0.05)"
              />
              <text
                x={padL - 8}
                y={yProb(g) + 3}
                textAnchor="end"
                fontSize="10"
                fill="#5f656c"
                fontFamily="'IBM Plex Mono', monospace"
              >
                {pct(g)}
              </text>
            </g>
          ))}

          {/* 50% reference line */}
          <line
            x1={padL}
            x2={w - padR}
            y1={yProb(0.5)}
            y2={yProb(0.5)}
            stroke="rgba(255,255,255,0.1)"
            strokeDasharray="2 3"
          />

          {/* Event markers */}
          {market.eventMarks.map((ev, k) => (
            <g key={k}>
              <line
                x1={x(ev.i)}
                x2={x(ev.i)}
                y1={padT}
                y2={padT + lineH}
                stroke={ev.kind === 'resolve' ? accent : 'rgba(245,165,36,0.35)'}
                strokeWidth="1"
                strokeDasharray="2 2"
              />
              <circle
                cx={x(ev.i)}
                cy={padT}
                r="3"
                fill={ev.kind === 'resolve' ? accent : '#f5a524'}
              />
            </g>
          ))}

          {/* Pressure overlay */}
          {showPressure && (
            <g>
              <path d={pressArea} fill={accent} opacity="0.08" />
              <path
                d={pressure.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${yPress(v)}`).join(' ')}
                fill="none"
                stroke={accent}
                strokeWidth="1"
                opacity="0.45"
              />
              <text
                x={padL - 8}
                y={yPress(100) + 3}
                textAnchor="end"
                fontSize="9"
                fill="#5f656c"
                fontFamily="'IBM Plex Mono', monospace"
              >
                100
              </text>
              <text
                x={padL - 8}
                y={yPress(0) + 3}
                textAnchor="end"
                fontSize="9"
                fill="#5f656c"
                fontFamily="'IBM Plex Mono', monospace"
              >
                0
              </text>
              <text
                x={w - padR}
                y={yPress(100) - 4}
                textAnchor="end"
                fontSize="9"
                fill="#5f656c"
                fontFamily="'IBM Plex Mono', monospace"
                letterSpacing="0.05em"
              >
                SOCIAL PRESSURE
              </text>
            </g>
          )}

          {/* Probability lines */}
          {area && <path d={areaPath(signal)} fill={accent} opacity="0.07" />}
          <path
            d={linePath(poly)}
            fill="none"
            stroke="rgba(255,255,255,0.45)"
            strokeWidth="1.75"
          />
          <path
            d={linePath(signal)}
            fill="none"
            stroke={accent}
            strokeWidth="2.25"
          />
          <circle
            cx={x(n - 1)}
            cy={yProb(signal[n - 1])}
            r="3.5"
            fill={accent}
          />
          <circle
            cx={x(n - 1)}
            cy={yProb(poly[n - 1])}
            r="3"
            fill="#0a0b0c"
            stroke="rgba(255,255,255,0.6)"
            strokeWidth="1.5"
          />

          {/* X-axis labels */}
          {[0, Math.floor(n / 2), n - 1].map((i) => (
            <text
              key={i}
              x={x(i)}
              y={height - 8}
              textAnchor={i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle'}
              fontSize="10"
              fill="#5f656c"
              fontFamily="'IBM Plex Mono', monospace"
            >
              {market.dateLabels[i]}
            </text>
          ))}

          {/* Hover crosshair */}
          {hover !== null && (
            <g>
              <line
                x1={x(hover)}
                x2={x(hover)}
                y1={padT}
                y2={padT + lineH}
                stroke="rgba(255,255,255,0.18)"
              />
              <circle
                cx={x(hover)}
                cy={yProb(signal[hover])}
                r="3.5"
                fill={accent}
                stroke="#0a0b0c"
                strokeWidth="1.5"
              />
              <circle
                cx={x(hover)}
                cy={yProb(poly[hover])}
                r="3.5"
                fill="#e8eaed"
                stroke="#0a0b0c"
                strokeWidth="1.5"
              />
            </g>
          )}
        </svg>
      )}

      {/* Tooltip */}
      {hover !== null && (
        <div style={{
          position: 'absolute',
          top: 6,
          right: 6,
          pointerEvents: 'none',
          background: 'rgba(10,11,12,0.92)',
          border: '1px solid #26292d',
          borderRadius: 6,
          padding: '7px 9px',
          font: "11px/1.5 'IBM Plex Mono', monospace",
          color: '#e8eaed',
          boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
        }}>
          <div style={{ color: '#9aa0a6', marginBottom: 3 }}>
            {market.dateLabels[hover]}
          </div>
          <div>
            <span style={{ color: accent }}>●</span> Signal {pct(signal[hover])}%
          </div>
          <div>
            <span style={{ color: 'rgba(255,255,255,0.6)' }}>●</span> Polymarket {pct(poly[hover])}%
          </div>
          <div style={{ color: '#9aa0a6' }}>
            Pressure {Math.round(pressure[hover])}
          </div>
        </div>
      )}
    </div>
  );
}

/* ---- Stance distribution bar (segmented) -------------------------------- */
function StanceBar({ stance, accent = '#34d399' }) {
  const segs = [
    { k: 'YES', v: stance.yes, c: accent },
    { k: 'NO', v: stance.no, c: '#f76d6d' },
    { k: 'NEUTRAL', v: stance.neutral, c: '#5f656c' },
    { k: 'AMBIG', v: stance.ambiguous, c: '#3a3d42' },
  ];

  return (
    <div>
      <div style={{
        display: 'flex',
        height: 30,
        borderRadius: 5,
        overflow: 'hidden',
        border: '1px solid #1c1e21'
      }}>
        {segs.map((s) => (
          <div
            key={s.k}
            title={`${s.k} ${s.v}%`}
            style={{
              width: s.v + '%',
              background: s.c,
              opacity: s.k === 'YES' ? 1 : 0.8
            }}
          />
        ))}
      </div>
      <div style={{
        display: 'flex',
        gap: 16,
        marginTop: 10,
        flexWrap: 'wrap'
      }}>
        {segs.map((s) => (
          <div
            key={s.k}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              font: "11px 'IBM Plex Mono', monospace",
              color: '#9aa0a6'
            }}
          >
            <span style={{
              width: 8,
              height: 8,
              borderRadius: 2,
              background: s.c,
              display: 'inline-block'
            }} />
            {s.k} <span style={{ color: '#e8eaed' }}>{s.v}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ---- Engagement / pressure mini bars ------------------------------------ */
function MiniBars({ data, w = 220, h = 44, accent = '#34d399' }) {
  const [ref, mw] = useMeasure();
  const width = mw || w;
  const numBars = 24;

  const vals = useMemo(() => {
    const kf = data;
    const out = [];
    const maxD = kf[kf.length - 1][0];

    for (let i = 0; i < numBars; i++) {
      const x = (i / (numBars - 1)) * maxD;
      let v = kf[0][1];

      for (let j = 0; j < kf.length - 1; j++) {
        const [x0, y0] = kf[j];
        const [x1, y1] = kf[j + 1];

        if (x >= x0 && x <= x1) {
          v = y0 + (y1 - y0) * ((x - x0) / (x1 - x0));
          break;
        }
      }
      out.push(v);
    }

    return out;
  }, [data]);

  const max = Math.max(...vals) || 1;
  const barWidth = (width - (numBars - 1) * 2) / numBars;

  return (
    <div ref={ref} style={{ width: '100%' }}>
      {mw > 0 && (
        <svg width={width} height={h} style={{ display: 'block' }}>
          {vals.map((v, i) => {
            const barHeight = (v / max) * (h - 4);
            const opacity = 0.25 + 0.6 * (v / max);

            return (
              <rect
                key={i}
                x={i * (barWidth + 2)}
                y={h - barHeight}
                width={barWidth}
                height={barHeight}
                rx="1"
                fill={accent}
                opacity={opacity}
              />
            );
          })}
        </svg>
      )}
    </div>
  );
}

/* ---- Radial gauge (confidence) ------------------------------------------ */
function Gauge({ value, label, accent = '#34d399', size = 86 }) {
  const r = size / 2 - 6;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - value / 100);

  return (
    <div style={{ position: 'relative', width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="#1c1e21"
          strokeWidth="5"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={accent}
          strokeWidth="5"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <div style={{
          font: "600 19px 'IBM Plex Mono', monospace",
          color: '#e8eaed'
        }}>
          {value}
        </div>
        <div style={{
          font: "9px 'IBM Plex Mono', monospace",
          color: '#5f656c',
          letterSpacing: '0.06em'
        }}>
          {label}
        </div>
      </div>
    </div>
  );
}

/* ---- Narrative graph (layered node-link) -------------------------------- */
function NarrativeGraph({ market, accent = '#34d399' }) {
  const [ref, w] = useMeasure();
  const h = 230;

  const entities = market.entities;
  const platforms = market.platforms;
  const posts = market.posts.slice(0, 4);

  if (!w) {
    return <div ref={ref} style={{ width: '100%', height: h }} />;
  }

  const mx = 56;
  const ex = w * 0.36;
  const px = w * 0.64;
  const plx = w * 0.9;

  const ySpread = (count, i) => 40 + (i + 0.5) * ((h - 80) / count);

  const nodes = {
    market: { x: mx, y: h / 2 },
    entities: entities.map((e, i) => ({
      x: ex,
      y: ySpread(entities.length, i),
      label: e
    })),
    posts: posts.map((p, i) => ({
      x: px,
      y: ySpread(posts.length, i),
      stance: p.stance,
      label: p.platform
    })),
    platforms: platforms.map((p, i) => ({
      x: plx,
      y: ySpread(platforms.length, i),
      label: p
    })),
  };

  const edges = [];
  nodes.entities.forEach((e) => edges.push([nodes.market, e]));
  nodes.posts.forEach((p, i) => {
    const e = nodes.entities[i % nodes.entities.length];
    edges.push([e, p]);
    const plat = nodes.platforms.find(pl => pl.label === p.label) || nodes.platforms[i % nodes.platforms.length];
    edges.push([p, plat]);
  });

  const stanceColor = (s) => {
    if (s === 'YES') return accent;
    if (s === 'NO') return '#f76d6d';
    return '#7d838a';
  };

  return (
    <div ref={ref} style={{ width: '100%' }}>
      <svg width={w} height={h} style={{ display: 'block' }}>
        {/* Column captions */}
        {[['MARKET', mx], ['ENTITIES', ex], ['POSTS', px], ['PLATFORMS', plx]].map(([t, xx]) => (
          <text
            key={t}
            x={xx}
            y={16}
            textAnchor="middle"
            fontSize="9"
            fill="#4a4f55"
            fontFamily="'IBM Plex Mono', monospace"
            letterSpacing="0.08em"
          >
            {t}
          </text>
        ))}

        {/* Edges */}
        {edges.map(([a, b], i) => (
          <path
            key={i}
            d={`M${a.x},${a.y} C${(a.x + b.x) / 2},${a.y} ${(a.x + b.x) / 2},${b.y} ${b.x},${b.y}`}
            fill="none"
            stroke="rgba(255,255,255,0.12)"
            strokeWidth="1"
          />
        ))}

        {/* Market node */}
        <g>
          <circle cx={nodes.market.x} cy={nodes.market.y} r="9" fill={accent} />
          <circle
            cx={nodes.market.x}
            cy={nodes.market.y}
            r="14"
            fill="none"
            stroke={accent}
            opacity="0.3"
          />
        </g>

        {/* Entity nodes */}
        {nodes.entities.map((e, i) => (
          <g key={i}>
            <circle cx={e.x} cy={e.y} r="5" fill="#16181a" stroke="#3a3d42" />
            <text
              x={e.x}
              y={e.y - 10}
              textAnchor="middle"
              fontSize="9.5"
              fill="#9aa0a6"
              fontFamily="'IBM Plex Sans', sans-serif"
            >
              {e.label}
            </text>
          </g>
        ))}

        {/* Post nodes */}
        {nodes.posts.map((p, i) => (
          <g key={i}>
            <rect
              x={p.x - 5}
              y={p.y - 5}
              width="10"
              height="10"
              rx="2"
              fill="#16181a"
              stroke={stanceColor(p.stance)}
            />
            <circle
              cx={p.x + 6}
              cy={p.y - 6}
              r="2.5"
              fill={stanceColor(p.stance)}
            />
          </g>
        ))}

        {/* Platform nodes */}
        {nodes.platforms.map((p, i) => (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r="4" fill="#26292d" />
            <text
              x={p.x}
              y={p.y + 3}
              dx="10"
              textAnchor="start"
              fontSize="9.5"
              fill="#9aa0a6"
              fontFamily="'IBM Plex Mono', monospace"
            >
              {p.label}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

Object.assign(window, {
  Sparkline,
  DualLineChart,
  StanceBar,
  MiniBars,
  Gauge,
  NarrativeGraph,
  useMeasure,
  pct,
  fmtSignedPts
});
