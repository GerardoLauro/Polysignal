/* PolySignal — Market Detail page ------------------------------------------ */

function MarketDetail({ market, accent, density, system, chartStyle, onBack }) {
  const [showPressure, setShowPressure] = useState(chartStyle?.pressure !== false);
  const [area, setArea] = useState(chartStyle?.area || false);
  useEffect(() => { setShowPressure(chartStyle?.pressure !== false); setArea(chartStyle?.area || false); }, [chartStyle]);

  const gap = density === 'compact' ? 12 : 16;

  return (
    <div style={{ flex: 1, overflowY: 'auto' }}>
      <div style={{ maxWidth: 1320, margin: '0 auto', padding: density === 'compact' ? '18px 22px 40px' : '24px 28px 56px' }}>
        <DetailHeader market={market} accent={accent} onBack={onBack} />

        {/* summary KPIs */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap, margin: `${gap}px 0` }}>
          <SummaryCard label="Polymarket prob" big={pct(market.polyProb) + '%'} sub="order-book implied" color="rgba(255,255,255,0.78)" />
          <SummaryCard label="Social estimate" big={pct(market.signalProb) + '%'} sub="market-anchored · in-sample" color={accent} glow={accent} />
          <SummaryCard label="Delta" custom={<DeltaTag delta={market.delta} accent={accent} size="lg" />} sub="estimate − market" />
          <SummaryCard label="Confidence" custom={<Gauge value={market.conf} label="CONF" accent={accent} size={66} />} center />
          <SummaryCard label="Social pressure" custom={
            <div>
              <div style={{ font: "600 22px 'IBM Plex Mono', monospace", color: '#e8eaed' }}>{Math.round(market.pressureNow)}</div>
              <div style={{ font: "10px 'IBM Plex Mono', monospace", color: accent, marginTop: 2 }}>{market.pressureDir}</div>
            </div>
          } sub="strength · decision-time dir" />
        </div>

        {/* main chart */}
        <Panel
          title="Probability timeline"
          sub={`${system.coverageWindow} · daily`}
          pad={18}
          right={
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              <Legend accent={accent} />
              <div style={{ display: 'flex', gap: 6 }}>
                <ChartToggle on={showPressure} onClick={() => setShowPressure(v => !v)} accent={accent}>Pressure</ChartToggle>
                <ChartToggle on={area} onClick={() => setArea(v => !v)} accent={accent}>Area</ChartToggle>
              </div>
            </div>
          }
        >
          <DualLineChart market={market} accent={accent} showPressure={showPressure} area={area} height={330} />
          <EventLegend market={market} accent={accent} />
        </Panel>

        {/* panel grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.35fr) minmax(0, 1fr)', gap, marginTop: gap, alignItems: 'start' }}>
          <div style={{ display: 'grid', gap }}>
            <SocialPanel market={market} accent={accent} />
            <EvidencePanel market={market} accent={accent} />
          </div>
          <div style={{ display: 'grid', gap }}>
            <ScorecardPanel market={market} accent={accent} />
            <DivergencePanel market={market} accent={accent} />
            <GraphPanel market={market} accent={accent} />
            <SystemPanel system={system} accent={accent} />
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---- Header ------------------------------------------------------------- */
function DetailHeader({ market, accent, onBack }) {
  return (
    <div>
      <button onClick={onBack} style={{
        display: 'inline-flex', alignItems: 'center', gap: 7, background: 'none', border: 'none', cursor: 'pointer',
        font: "11px 'IBM Plex Mono', monospace", color: '#7d838a', padding: 0, marginBottom: 16,
      }}>← Market Universe</button>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 24 }}>
        <div style={{ maxWidth: 720 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
            <DomainTag domain={market.domain} />
            <StatusBadge status={market.status} />
            <DivergenceBadge delta={market.delta} accent={accent} />
            <span style={{ font: "10.5px 'IBM Plex Mono', monospace", color: '#5f656c' }}>
              {market.status === 'RESOLVED' ? market.resolution : 'Resolves ' + market.resolution}
            </span>
          </div>
          <h1 style={{ margin: 0, font: "500 25px/1.28 'IBM Plex Sans', sans-serif", color: '#f0f2f4', letterSpacing: '-0.01em' }}>{market.question}</h1>
        </div>
        <div style={{ flexShrink: 0, minWidth: 220 }}>
          <div style={{ font: "9.5px 'IBM Plex Mono', monospace", color: '#4a4f55', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 9 }}>Main drivers</div>
          <ul style={{ margin: 0, padding: 0, listStyle: 'none', display: 'grid', gap: 7 }}>
            {market.drivers.map((d, i) => (
              <li key={i} style={{ display: 'flex', gap: 8, font: "11.5px/1.4 'IBM Plex Sans', sans-serif", color: '#9aa0a6' }}>
                <span style={{ color: accent, flexShrink: 0 }}>·</span>{d}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function SummaryCard({ label, big, custom, sub, color, glow, center }) {
  return (
    <div style={{ background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 10, padding: 16, boxShadow: glow ? `inset 0 0 0 1px ${glow}22` : 'none' }}>
      <div style={{ font: "9.5px 'IBM Plex Mono', monospace", color: '#5f656c', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 10 }}>{label}</div>
      <div style={{ display: 'flex', justifyContent: center ? 'center' : 'flex-start' }}>
        {custom || <div style={{ font: "600 26px 'IBM Plex Mono', monospace", color: color || '#e8eaed', lineHeight: 1 }}>{big}</div>}
      </div>
      {sub && <div style={{ font: "10px 'IBM Plex Mono', monospace", color: '#5f656c', marginTop: 9 }}>{sub}</div>}
    </div>
  );
}

function Legend({ accent }) {
  return (
    <div style={{ display: 'flex', gap: 14, font: "10.5px 'IBM Plex Mono', monospace", color: '#9aa0a6' }}>
      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><span style={{ width: 14, height: 2, background: accent, display: 'inline-block' }} />Social est. (anchored)</span>
      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><span style={{ width: 14, height: 2, background: 'rgba(255,255,255,0.5)', display: 'inline-block' }} />Polymarket</span>
    </div>
  );
}

function ChartToggle({ on, onClick, children, accent }) {
  return (
    <button onClick={onClick} style={{
      font: "10px 'IBM Plex Mono', monospace", padding: '4px 9px', borderRadius: 5, cursor: 'pointer',
      border: '1px solid ' + (on ? accent + '55' : 'var(--border)'),
      background: on ? accent + '16' : 'transparent', color: on ? '#e8eaed' : '#7d838a',
    }}>{children}</button>
  );
}

function EventLegend({ market, accent }) {
  return (
    <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', marginTop: 12, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
      {market.eventMarks.map((ev, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 7, font: "10.5px 'IBM Plex Mono', monospace", color: '#9aa0a6' }}>
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: ev.kind === 'resolve' ? accent : '#f5a524', flexShrink: 0 }} />
          <span style={{ color: '#5f656c' }}>{market.dateLabels[ev.i]}</span>
          {ev.label}
        </div>
      ))}
    </div>
  );
}

/* ---- Social Signals panel ----------------------------------------------- */
function SocialPanel({ market, accent }) {
  const s = market;
  return (
    <Panel title="Social signals" sub={`${s.acceptedPosts} accepted of ${s.totalSeen} seen`}>
      <div style={{ font: "10px 'IBM Plex Mono', monospace", color: '#5f656c', letterSpacing: '0.05em', marginBottom: 10, textTransform: 'uppercase' }}>Stance distribution</div>
      <StanceBar stance={s.stance} accent={accent} />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 22, marginTop: 22 }}>
        <div>
          <div style={{ font: "10px 'IBM Plex Mono', monospace", color: '#5f656c', letterSpacing: '0.05em', marginBottom: 8, textTransform: 'uppercase' }}>Engagement volume</div>
          <MiniBars data={s.engagement} accent={accent} h={46} />
        </div>
        <div>
          <div style={{ font: "10px 'IBM Plex Mono', monospace", color: '#5f656c', letterSpacing: '0.05em', marginBottom: 8, textTransform: 'uppercase' }}>Social pressure</div>
          <MiniBars data={s.series.pressure.map((v, i) => [i, v])} accent={accent} h={46} />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginTop: 22, paddingTop: 18, borderTop: '1px solid var(--border)' }}>
        <Stat label="Source diversity" accent="#e8eaed">{Math.round(s.sourceDiversity * 100)}<span style={{ fontSize: 12, color: '#5f656c' }}>%</span></Stat>
        <Stat label="Pressure dir" sub="at decision time" accent={s.pressureDir === 'Toward NO' ? '#f76d6d' : s.pressureDir === 'Balanced' ? '#9aa0a6' : accent}>{s.pressureDir.replace('Toward ', '')}</Stat>
        <Stat label="Net pressure" sub="decision-time lean" accent={s.netPressure >= 0 ? accent : '#f76d6d'}>{s.netPressure >= 0 ? '+' : ''}{s.netPressure.toFixed(2)}</Stat>
        <Stat label="Stance balance" sub="full-window yes − no">{(s.stance.yes - s.stance.no) >= 0 ? '+' : ''}{s.stance.yes - s.stance.no} pts</Stat>
        <Stat label="Accepted links">{s.acceptedPosts}</Stat>
        <Stat label="Evidence quality" accent={accent}>{s.evidenceQuality}<span style={{ fontSize: 12, color: '#5f656c' }}>/100</span></Stat>
      </div>
    </Panel>
  );
}

/* ---- Forecast scorecard panel ------------------------------------------- */
function ScorecardPanel({ market, accent }) {
  const sc = market.scorecard;
  if (!sc) return null;
  const verdict = sc.edge === 'signal'
    ? 'The market-anchored estimate was closer than the order book here.'
    : sc.edge === 'market'
      ? 'The order book was closer than the social estimate here.'
      : 'Order book and social estimate were equally close.';
  const Row = ({ label, prob, ok, brier, won }) => (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '11px 0', borderTop: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 9 }}>
        <span style={{ width: 78, font: "11px 'IBM Plex Mono', monospace", color: won ? '#e8eaed' : '#9aa0a6' }}>{label}</span>
        <span style={{ font: "600 16px 'IBM Plex Mono', monospace", color: won ? accent : '#c4c8cc' }}>{Math.round(prob * 100)}%</span>
        <span style={{ font: "9px 'IBM Plex Mono', monospace", color: '#5f656c' }}>P(YES)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        <HitMark ok={ok} label={ok ? 'called it' : 'missed'} />
        <span style={{ font: "11px 'IBM Plex Mono', monospace", color: '#9aa0a6' }}>Brier {brier.toFixed(2)}</span>
        {won && <span style={{ font: "9px 'IBM Plex Mono', monospace", color: accent, border: `1px solid ${accent}44`, background: `${accent}14`, padding: '2px 6px', borderRadius: 4, letterSpacing: '0.04em' }}>CLOSER</span>}
      </div>
    </div>
  );
  return (
    <Panel title="Descriptive comparison" right={<OutcomeBadge outcome={sc.outcome} />}>
      <div style={{ font: "10px 'IBM Plex Mono', monospace", color: '#5f656c', letterSpacing: '0.05em', textTransform: 'uppercase', marginBottom: 4 }}>
        Resolved {sc.outcome} · in-sample · not the predictive benchmark
      </div>
      <Row label="Polymarket" prob={market.polyProb} ok={sc.marketCorrect} brier={sc.marketBrier} won={sc.edge === 'market'} />
      <Row label="Social est." prob={market.signalProb} ok={sc.signalCorrect} brier={sc.signalBrier} won={sc.edge === 'signal'} />
      <div style={{ marginTop: 12, paddingTop: 11, borderTop: '1px solid var(--border)', font: "11px/1.5 'IBM Plex Sans', sans-serif", color: '#9aa0a6' }}>
        {verdict} <span style={{ color: '#5f656c' }}>Lower Brier = closer to the outcome. This is a descriptive, in-sample comparison of the market-anchored estimate — see System status for the out-of-fold predictive benchmark.</span>
      </div>
    </Panel>
  );
}

/* ---- Divergence panel --------------------------------------------------- */
function DivergencePanel({ market, accent }) {
  const v = Math.round(market.delta * 100);
  const mag = Math.min(100, Math.abs(v) * 5);
  const strength = market.conf >= 80 ? 'Strong' : market.conf >= 65 ? 'Moderate' : 'Weak';
  return (
    <Panel title="Divergence" right={<DivergenceBadge delta={market.delta} accent={accent} />}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 18, marginBottom: 16 }}>
        <div style={{ flex: 1 }}>
          <div style={{ font: "10px 'IBM Plex Mono', monospace", color: '#5f656c', letterSpacing: '0.05em', marginBottom: 8, textTransform: 'uppercase' }}>Market ↔ social disagreement</div>
          <div style={{ position: 'relative', height: 8, background: '#16181a', borderRadius: 4, overflow: 'hidden' }}>
            <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: '#3a3d42' }} />
            <div style={{ position: 'absolute', top: 0, bottom: 0, left: v >= 0 ? '50%' : (50 + v * 5) + '%', width: Math.abs(v) * 5 + '%', background: v >= 0 ? accent : '#f76d6d', opacity: 0.85 }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', font: "9px 'IBM Plex Mono', monospace", color: '#5f656c', marginTop: 5 }}>
            <span>← market higher</span><span>social higher →</span>
          </div>
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14, marginBottom: 16 }}>
        <Stat label="Δ probability" accent={v >= 0 ? accent : '#f76d6d'}>{v >= 0 ? '+' : ''}{v} pts</Stat>
        <Stat label="Signal strength" accent={accent}>{strength}</Stat>
        <Stat label="Confidence">{market.conf}%</Stat>
      </div>
      <div style={{ background: '#0d0e0f', border: '1px solid var(--border)', borderRadius: 8, padding: 14 }}>
        <div style={{ font: "9.5px 'IBM Plex Mono', monospace", color: accent, letterSpacing: '0.06em', marginBottom: 8, textTransform: 'uppercase' }}>Why PolySignal moved probability</div>
        <p style={{ margin: 0, font: "12px/1.6 'IBM Plex Sans', sans-serif", color: '#bcc1c6' }}>{market.divergenceExplain}</p>
        <p style={{ margin: '10px 0 0', font: "10px/1.5 'IBM Plex Mono', monospace", color: '#5f656c' }}>Decision-support signal · controlled benchmark/demo view. Not a trading recommendation.</p>
      </div>
    </Panel>
  );
}

/* ---- Linked Evidence panel ---------------------------------------------- */
const PLATFORM_GLYPH = { X: '𝕏', Reddit: 'r/', Farcaster: 'fc', News: '◆' };
function EvidencePanel({ market, accent }) {
  if (!market.posts || market.posts.length === 0) {
    return (
      <Panel title="Linked evidence" sub="no social coverage">
        <div style={{ padding: '18px 4px', font: "12.5px/1.6 'IBM Plex Sans', sans-serif", color: '#9aa0a6' }}>
          No social discourse in the collected corpus names this market's subject, so there is
          no linked evidence. The Polymarket price history and resolution are shown for
          completeness — a <strong style={{ color: '#c4c8cc' }}>no-signal / coverage case</strong>.
        </div>
      </Panel>
    );
  }
  return (
    <Panel title="Linked evidence" sub={`${market.posts.length} shown · ranked by link confidence`}>
      <div style={{ display: 'grid', gap: 10 }}>
        {market.posts.map((p, i) => (
          <div key={i} style={{ border: '1px solid var(--border)', borderRadius: 8, padding: 13, background: '#0d0e0f' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                <span style={{ width: 22, height: 22, borderRadius: 5, background: '#16181a', border: '1px solid var(--border)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', font: "10px 'IBM Plex Mono', monospace", color: '#9aa0a6' }}>{PLATFORM_GLYPH[p.platform] || '•'}</span>
                <div>
                  <span style={{ font: "12px 'IBM Plex Mono', monospace", color: '#e8eaed' }}>{p.author}</span>
                  <span style={{ font: "10px 'IBM Plex Mono', monospace", color: '#5f656c', marginLeft: 8 }}>{p.handle} · {p.ago}</span>
                </div>
              </div>
              <StanceTag stance={p.stance} accent={accent} />
            </div>
            <p style={{ margin: '0 0 10px', font: "12.5px/1.5 'IBM Plex Sans', sans-serif", color: '#c4c8cc' }}>"{p.text}"</p>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap', font: "10px 'IBM Plex Mono', monospace", color: '#5f656c' }}>
              <Meter label="link" value={p.link} accent={accent} />
              <Meter label="conf" value={p.conf} accent="#9aa0a6" />
              <span>↻ {p.eng}</span>
              {p.url ? <a href={p.url} target="_blank" rel="noreferrer" style={{ color: accent, textDecoration: 'none' }}>source ↗</a> : null}
            </div>
            <PostFeatures post={p} accent={accent} />
            <div style={{ marginTop: 9, paddingTop: 9, borderTop: '1px solid var(--border)', font: "10.5px/1.5 'IBM Plex Sans', sans-serif", color: '#7d838a' }}>
              <span style={{ color: accent, fontFamily: "'IBM Plex Mono', monospace", fontSize: 9.5, letterSpacing: '0.04em' }}>WHY LINKED </span>{p.why}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

/* ---- Per-post inspector: extracted features (NER / sentiment / language) -- */
function PostFeatures({ post, accent }) {
  const ents = post.entities || [];
  const tags = post.hashtags || [];
  const hasFeatures = post.language || post.sentimentLabel || ents.length || tags.length || post.commentSample;
  if (!hasFeatures) return null;
  const sentColor = post.sentimentLabel === 'positive' ? accent
    : post.sentimentLabel === 'negative' ? '#f76d6d' : '#9aa0a6';
  const chip = (text, color, key) => (
    <span key={key} style={{ font: "9.5px 'IBM Plex Mono', monospace", color, border: `1px solid ${color}33`, background: `${color}10`, padding: '2px 6px', borderRadius: 4 }}>{text}</span>
  );
  return (
    <div style={{ marginTop: 9, paddingTop: 9, borderTop: '1px solid var(--border)', display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
      <span style={{ font: "9px 'IBM Plex Mono', monospace", color: '#5f656c', letterSpacing: '0.04em' }}>FEATURES</span>
      {post.language ? chip(post.language, '#9aa0a6', 'lang') : null}
      {post.sentimentLabel ? chip(`${post.sentimentLabel} ${(post.sentimentScore >= 0 ? '+' : '') + post.sentimentScore}`, sentColor, 'sent') : null}
      {ents.slice(0, 5).map((e, i) => chip(e, '#c4c8cc', 'e' + i))}
      {tags.slice(0, 4).map((t, i) => chip('#' + String(t).replace(/^#/, ''), accent, 't' + i))}
      {post.commentSample ? chip(`${post.commentSample} comments`, '#7d838a', 'cm') : null}
    </div>
  );
}

function StanceTag({ stance, accent }) {
  const map = { YES: accent, NO: '#f76d6d', NEUTRAL: '#9aa0a6', AMBIGUOUS: '#7d838a' };
  const c = map[stance] || '#9aa0a6';
  return <span style={{ font: "9.5px 'IBM Plex Mono', monospace", color: c, border: `1px solid ${c}44`, background: `${c}14`, padding: '2px 7px', borderRadius: 4, letterSpacing: '0.05em' }}>{stance}</span>;
}

function Meter({ label, value, accent }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
      {label}
      <span style={{ width: 32, height: 4, background: '#1c1e21', borderRadius: 2, overflow: 'hidden', display: 'inline-block' }}>
        <span style={{ display: 'block', height: '100%', width: Math.round(value * 100) + '%', background: accent }} />
      </span>
      <span style={{ color: '#9aa0a6' }}>{value.toFixed(2)}</span>
    </span>
  );
}

/* ---- Narrative graph panel ---------------------------------------------- */
function GraphPanel({ market, accent }) {
  return (
    <Panel title="Narrative graph" sub="market → entities → posts → platforms">
      <NarrativeGraph market={market} accent={accent} />
      <div style={{ display: 'flex', gap: 16, marginTop: 8, flexWrap: 'wrap', font: "9.5px 'IBM Plex Mono', monospace", color: '#7d838a' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}><span style={{ width: 8, height: 8, borderRadius: '50%', background: accent }} />YES-leaning</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}><span style={{ width: 8, height: 8, borderRadius: 2, background: '#f76d6d' }} />NO-leaning</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}><span style={{ width: 8, height: 8, borderRadius: 2, background: '#7d838a' }} />neutral post</span>
      </div>
    </Panel>
  );
}

/* ---- System Status panel ------------------------------------------------ */
function SystemPanel({ system, accent }) {
  return (
    <Panel title="System status" right={<span style={{ font: "9.5px 'IBM Plex Mono', monospace", color: accent, border: `1px solid ${accent}44`, background: `${accent}12`, padding: '2px 7px', borderRadius: 4 }}>{system.mode}</span>}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 16 }}>
        <Stat label="Last update" sub={system.lastUpdateAgo}><span style={{ fontSize: 13 }}>{system.lastUpdate}</span></Stat>
        <Stat label="Coverage">{system.coverageWindow}</Stat>
        <Stat label="Markets loaded">{system.marketsLoaded}</Stat>
        <Stat label="Posts loaded">{system.postsLoaded.toLocaleString()}</Stat>
      </div>
      <div style={{ font: "9.5px 'IBM Plex Mono', monospace", color: '#5f656c', letterSpacing: '0.05em', marginBottom: 6, textTransform: 'uppercase' }}>Active profile</div>
      <div style={{ font: "11.5px 'IBM Plex Mono', monospace", color: '#c4c8cc', marginBottom: 4 }}>{system.benchmark}</div>
      <div style={{ font: "10.5px 'IBM Plex Mono', monospace", color: '#7d838a', marginBottom: 16 }}>{system.profile}</div>
      {system.predictiveBenchmark && system.predictiveBenchmark.available && (() => {
        const b = system.predictiveBenchmark;
        const ab = b.ablations || {};
        const rows = [['market_only', 'Market only'], ['social_only', 'Social only'], ['market_social', 'Market+social']];
        return (
          <div style={{ marginBottom: 16 }}>
            <div style={{ font: "9.5px 'IBM Plex Mono', monospace", color: '#5f656c', letterSpacing: '0.05em', marginBottom: 8, textTransform: 'uppercase' }}>Predictive benchmark · out-of-fold</div>
            <div style={{ display: 'grid', gap: 5 }}>
              {rows.map(([k, label]) => {
                const m = ab[k];
                if (!m) return null;
                const best = b.best && b.best.ablation === k;
                return (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', font: "10.5px 'IBM Plex Mono', monospace" }}>
                    <span style={{ color: best ? accent : '#9aa0a6' }}>{label}{best ? ' ★' : ''}</span>
                    <span style={{ color: '#c4c8cc' }}>AUC {Number(m.auc).toFixed(2)} · Brier {Number(m.brier).toFixed(3)}</span>
                  </div>
                );
              })}
            </div>
            <p style={{ margin: '8px 0 0', font: "10px/1.5 'IBM Plex Sans', sans-serif", color: '#7d838a' }}>
              {b.scheme} · {b.nFolds} folds · {b.nContracts} contracts. {b.conclusion}
            </p>
          </div>
        );
      })()}
      <div style={{ font: "9.5px 'IBM Plex Mono', monospace", color: '#5f656c', letterSpacing: '0.05em', marginBottom: 8, textTransform: 'uppercase' }}>Artifacts</div>
      <div style={{ display: 'grid', gap: 6 }}>
        {system.artifacts.map((a, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', font: "10.5px 'IBM Plex Mono', monospace" }}>
            <span style={{ color: '#9aa0a6' }}>{a.name}</span>
            <span style={{ color: a.ok ? accent : '#f5a524', display: 'flex', alignItems: 'center', gap: 5 }}>
              <span style={{ width: 5, height: 5, borderRadius: '50%', background: a.ok ? accent : '#f5a524' }} />
              {a.ok ? 'ready' : 'offline'}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}

Object.assign(window, { MarketDetail });
