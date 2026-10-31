/* PolySignal — Market Universe page ---------------------------------------- */

function Segmented({ value, onChange, options, accent }) {
  return (
    <div style={{
      display: 'inline-flex',
      background: '#0f1011',
      border: '1px solid var(--border)',
      borderRadius: 7,
      padding: 2
    }}>
      {options.map((o) => (
        <button
          key={o.v}
          onClick={() => onChange(o.v)}
          style={{
            font: "11px 'IBM Plex Mono', monospace",
            letterSpacing: '0.03em',
            padding: '5px 11px',
            borderRadius: 5,
            border: 'none',
            cursor: 'pointer',
            background: value === o.v ? '#1c1e21' : 'transparent',
            color: value === o.v ? '#e8eaed' : '#7d838a',
          }}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function FilterChip({ active, onClick, children, accent }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'block',
        width: '100%',
        textAlign: 'left',
        font: "11.5px 'IBM Plex Mono', monospace",
        letterSpacing: '0.02em',
        padding: '7px 10px',
        borderRadius: 6,
        cursor: 'pointer',
        border: '1px solid ' + (active ? accent + '55' : 'transparent'),
        background: active ? accent + '14' : 'transparent',
        color: active ? '#e8eaed' : '#9aa0a6',
        marginBottom: 3,
        transition: 'background 0.12s',
      }}
    >
      {children}
    </button>
  );
}

function MarketUniverse({ markets, system, accent, density, onOpen }) {
  const [layout, setLayout] = useState('table');
  const [q, setQ] = useState('');
  const [domain, setDomain] = useState('ALL');
  const [evidence, setEvidence] = useState('ALL');
  const [quick, setQuick] = useState(null);
  const [sort, setSort] = useState('divergence');
  const [showAnalytics, setShowAnalytics] = useState(false);

  const rowPad = density === 'compact' ? '9px 14px' : '13px 14px';
  const domains = ['ALL', ...Object.keys(window.DOMAIN_META)];

  const filtered = useMemo(() => {
    let list = markets.slice();

    if (q.trim()) {
      const searchTerm = q.toLowerCase();
      list = list.filter(m =>
        m.question.toLowerCase().includes(searchTerm) ||
        m.domain.toLowerCase().includes(searchTerm)
      );
    }

    if (domain !== 'ALL') {
      list = list.filter(m => m.domain === domain);
    }

    if (evidence !== 'ALL') {
      list = list.filter(m => m.evidenceBucket === evidence);
    }

    if (quick === 'highDiv') {
      list = list.filter(m => Math.abs(m.delta) >= 0.1);
    }

    if (quick === 'strongSignal') {
      list = list.filter(m => m.pressureNow >= 65);
    }

    if (quick === 'bestDemo') {
      list = list.filter(m =>
        ['fed-jun-cut', 'agentic-model', 'boxoffice-1b'].includes(m.id)
      );
    }

    const comparators = {
      divergence: (a, b) => Math.abs(b.delta) - Math.abs(a.delta),
      confidence: (a, b) => b.conf - a.conf,
      pressure: (a, b) => b.pressureNow - a.pressureNow,
      latest: (a, b) => a.short.localeCompare(b.short),
      resolution: (a, b) => (b.closeTs || '').localeCompare(a.closeTs || ''),
      outcome: (a, b) => (a.outcome || 'ZZ').localeCompare(b.outcome || 'ZZ'),
    };

    return list.sort(comparators[sort]);
  }, [markets, q, domain, evidence, quick, sort]);

  // Aggregate KPIs are averaged ONLY over markets that carry a real social signal.
  // The no-signal coverage layer (coverageTier === 'none') has delta and conf set to a
  // structural 0 (its signal line is drawn equal to the price), so averaging it in pulls
  // both metrics toward zero and misrepresents the signal. This mirrors the backend
  // forecast scorecard, which scores only evidence-bearing markets for the same reason.
  const signalMarkets = markets.filter(m => m.coverageTier !== 'none');
  const confMarkets = markets.filter(m => m.conf > 0);
  const avgDiv = signalMarkets.length
    ? Math.round(
        signalMarkets.reduce((a, m) => a + Math.abs(m.delta), 0) / signalMarkets.length * 100
      )
    : 0;
  const highDivCount = markets.filter(m => Math.abs(m.delta) >= 0.1).length;
  const avgConf = confMarkets.length
    ? Math.round(confMarkets.reduce((a, m) => a + m.conf, 0) / confMarkets.length)
    : 0;

  const hasActiveFilters = q || domain !== 'ALL' || evidence !== 'ALL' || quick;

  function clearFilters() {
    setQ('');
    setDomain('ALL');
    setEvidence('ALL');
    setQuick(null);
  }

  return (
    <div style={{ display: 'flex', gap: 0, minHeight: 0, flex: 1 }}>
      {/* Filter rail */}
      <aside style={{
        width: 230,
        flexShrink: 0,
        borderRight: '1px solid var(--border)',
        padding: 18,
        overflowY: 'auto'
      }}>
        <div style={{ position: 'relative', marginBottom: 18 }}>
          <input
            value={q}
            onChange={e => setQ(e.target.value)}
            placeholder="Search markets…"
            style={{
              width: '100%',
              boxSizing: 'border-box',
              background: '#0f1011',
              border: '1px solid var(--border)',
              borderRadius: 7,
              padding: '8px 10px 8px 28px',
              color: '#e8eaed',
              font: "12px 'IBM Plex Mono', monospace",
              outline: 'none',
            }}
          />
          <span style={{
            position: 'absolute',
            left: 10,
            top: 9,
            color: '#5f656c',
            fontSize: 12
          }}>
            ⌕
          </span>
        </div>

        <RailLabel>Domain</RailLabel>
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 4,
          marginBottom: 18
        }}>
          {domains.map(d => (
            <button
              key={d}
              onClick={() => setDomain(d)}
              style={{
                font: "10.5px 'IBM Plex Mono', monospace",
                padding: '4px 8px',
                borderRadius: 5,
                cursor: 'pointer',
                border: '1px solid ' + (domain === d ? accent + '55' : 'var(--border)'),
                background: domain === d ? accent + '18' : 'transparent',
                color: domain === d ? '#e8eaed' : '#9aa0a6',
              }}
            >
              {d === 'ALL' ? 'All' : d}
            </button>
          ))}
        </div>

        <RailLabel>Evidence bucket</RailLabel>
        <div style={{ marginBottom: 18 }}>
          {['ALL', 'High', 'Moderate', 'Sparse'].map(b => (
            <FilterChip
              key={b}
              active={evidence === b}
              onClick={() => setEvidence(b)}
              accent={accent}
            >
              {b === 'ALL' ? 'All buckets' : b}
            </FilterChip>
          ))}
        </div>

        <RailLabel>Quick views</RailLabel>
        <div style={{ marginBottom: 8 }}>
          {[
            { v: 'highDiv', l: '◈ High divergence' },
            { v: 'strongSignal', l: '⇡ Strongest social signal' },
            { v: 'bestDemo', l: '★ Best demo cases' },
          ].map(o => (
            <FilterChip
              key={o.v}
              active={quick === o.v}
              onClick={() => setQuick(quick === o.v ? null : o.v)}
              accent={accent}
            >
              {o.l}
            </FilterChip>
          ))}
        </div>

        {hasActiveFilters && (
          <button
            onClick={clearFilters}
            style={{
              font: "10.5px 'IBM Plex Mono', monospace",
              color: '#5f656c',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '6px 0',
            }}
          >
            ✕ clear filters
          </button>
        )}
      </aside>

      {/* Main content */}
      <div style={{
        flex: 1,
        minWidth: 0,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden'
      }}>
        <PredictiveBenchmarkBanner benchmark={system && system.predictiveBenchmark} accent={accent} />
        <ScorecardBanner scorecard={system && system.scorecard} accent={accent} />

        {/* KPI strip + controls */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '14px 22px',
          borderBottom: '1px solid var(--border)',
          gap: 16,
          flexWrap: 'wrap'
        }}>
          <div style={{ display: 'flex', gap: 26 }}>
            <Kpi
              label="Markets"
              value={filtered.length}
              sub={`of ${markets.length} saved`}
            />
            <Kpi
              label="Avg divergence"
              value={avgDiv}
              suffix="pts"
              sub={`${signalMarkets.length} w/ signal`}
              accent={accent}
            />
            <Kpi
              label="High divergence"
              value={highDivCount}
              sub="≥10 pts"
            />
            <Kpi
              label="Avg confidence"
              value={avgConf}
              suffix="%"
              sub={`${confMarkets.length} w/ signal`}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <span style={{
                font: "10px 'IBM Plex Mono', monospace",
                color: '#5f656c',
                letterSpacing: '0.05em'
              }}>
                SORT
              </span>
              <select
                value={sort}
                onChange={e => setSort(e.target.value)}
                style={{
                  background: '#0f1011',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  color: '#e8eaed',
                  font: "11px 'IBM Plex Mono', monospace",
                  padding: '5px 8px',
                  outline: 'none',
                  cursor: 'pointer',
                }}
              >
                <option value="divergence">Divergence</option>
                <option value="confidence">Confidence</option>
                <option value="pressure">Social pressure</option>
                <option value="resolution">Resolution date</option>
                <option value="outcome">Outcome</option>
                <option value="latest">Latest update</option>
              </select>
            </div>

            {system.analytics && system.analytics.available && (
              <button
                onClick={() => setShowAnalytics(v => !v)}
                style={{
                  background: showAnalytics ? accent + '18' : 'transparent',
                  border: '1px solid ' + (showAnalytics ? accent + '55' : 'var(--border)'),
                  borderRadius: 6, color: showAnalytics ? '#e8eaed' : '#9aa0a6',
                  font: "11px 'IBM Plex Mono', monospace", padding: '5px 10px', cursor: 'pointer',
                }}
              >
                {showAnalytics ? '▾' : '▸'} Analytics
              </button>
            )}

            <Segmented
              value={layout}
              onChange={setLayout}
              accent={accent}
              options={[
                { v: 'table', label: '☰ Table' },
                { v: 'cards', label: '▦ Cards' }
              ]}
            />
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
          {showAnalytics && <AnalyticsPanel system={system} accent={accent} />}
          {layout === 'table' ? (
            <UniverseTable
              markets={filtered}
              accent={accent}
              rowPad={rowPad}
              onOpen={onOpen}
            />
          ) : (
            <UniverseCards
              markets={filtered}
              accent={accent}
              onOpen={onOpen}
              density={density}
            />
          )}

          {filtered.length === 0 && (
            <div style={{
              padding: 60,
              textAlign: 'center',
              color: '#5f656c',
              font: "12px 'IBM Plex Mono', monospace"
            }}>
              No markets match these filters.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---- Task-2 analytics / distributions panel ----------------------------- */
function AnalyticsPanel({ system, accent }) {
  const a = system.analytics || {};
  const card = { background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 9, padding: 14 };
  const h = { font: "10px 'IBM Plex Mono', monospace", color: '#5f656c', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 10 };
  const cell = { font: "11px 'IBM Plex Mono', monospace", padding: '3px 8px' };
  const corr = a.correlation || {};
  const xp = a.crossPlatformState || {};
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 12, padding: '14px 16px' }}>
      {/* Price informativeness curve */}
      <div style={card}>
        <div style={h}>Price informativeness vs horizon</div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr style={{ color: '#7d838a' }}>
            {['T-days', 'n', 'Brier', 'Acc', 'AUC'].map(c => <th key={c} style={{ ...cell, textAlign: 'right', color: '#5f656c' }}>{c}</th>)}
          </tr></thead>
          <tbody>
            {(a.priceInformativeness || []).map((r, i) => (
              <tr key={i} style={{ color: '#c4c8cc' }}>
                <td style={{ ...cell, textAlign: 'right', color: accent }}>{r.horizon_days}</td>
                <td style={{ ...cell, textAlign: 'right' }}>{r.n_contracts}</td>
                <td style={{ ...cell, textAlign: 'right' }}>{r.brier}</td>
                <td style={{ ...cell, textAlign: 'right' }}>{r.accuracy}</td>
                <td style={{ ...cell, textAlign: 'right' }}>{r.auc}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ font: "10px/1.5 'IBM Plex Sans', sans-serif", color: '#7d838a', marginTop: 8 }}>
          Market price becomes more informative of the resolved outcome as resolution approaches.
        </div>
      </div>

      {/* Sentiment distribution per platform/domain */}
      <div style={card}>
        <div style={h}>Sentiment per platform · domain</div>
        <div style={{ display: 'grid', gap: 5 }}>
          {(a.sentimentDistribution || []).map((r, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ ...cell, color: '#9aa0a6', minWidth: 128 }}>{r.platform}·{r.domain}</span>
              <span style={{ flex: 1, display: 'flex', height: 8, borderRadius: 3, overflow: 'hidden', background: '#1c1e21' }}>
                <span style={{ width: (r.pos_share * 100) + '%', background: accent }} />
                <span style={{ width: (r.neu_share * 100) + '%', background: '#5f656c' }} />
                <span style={{ width: (r.neg_share * 100) + '%', background: '#f76d6d' }} />
              </span>
              <span style={{ ...cell, color: '#7d838a' }}>{r.n_enriched}</span>
            </div>
          ))}
        </div>
        <div style={{ font: "10px 'IBM Plex Mono', monospace", color: '#7d838a', marginTop: 8 }}>
          <span style={{ color: accent }}>■</span> pos <span style={{ color: '#5f656c' }}>■</span> neu <span style={{ color: '#f76d6d' }}>■</span> neg
        </div>
      </div>

      {/* Signal-market correlation + cross-platform */}
      <div style={card}>
        <div style={h}>Signal ↔ market correlation</div>
        <div style={{ display: 'grid', gap: 6, font: "11px 'IBM Plex Mono', monospace", color: '#c4c8cc' }}>
          <Row2 k="markets analyzed" v={corr.n_markets} />
          <Row2 k="social leads" v={(corr.relation_counts || {}).social_leads} />
          <Row2 k="coincident" v={(corr.relation_counts || {}).coincident} />
          <Row2 k="social follows" v={(corr.relation_counts || {}).social_follows} />
          <Row2 k="sentiment↔price aligned" v={corr.alignment_rate != null ? Math.round(corr.alignment_rate * 100) + '%' : '—'} />
          <Row2 k="volume spikes studied" v={corr.total_spikes} />
        </div>
        <div style={{ ...h, marginTop: 14 }}>Cross-platform</div>
        <div style={{ display: 'grid', gap: 6, font: "11px 'IBM Plex Mono', monospace", color: '#c4c8cc' }}>
          <Row2 k="state" v={xp.state} />
          <Row2 k="multi-platform contracts" v={xp.multi_platform_contracts} />
          <Row2 k="platforms present" v={(xp.platforms_present || []).join(', ') || '—'} />
        </div>
      </div>
    </div>
  );
}

function Row2({ k, v }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
      <span style={{ color: '#7d838a' }}>{k}</span>
      <span>{v == null ? '—' : String(v)}</span>
    </div>
  );
}

function RailLabel({ children }) {
  return (
    <div style={{
      font: "9.5px 'IBM Plex Mono', monospace",
      color: '#4a4f55',
      letterSpacing: '0.1em',
      textTransform: 'uppercase',
      marginBottom: 8
    }}>
      {children}
    </div>
  );
}

function Kpi({ label, value, sub, suffix, accent }) {
  return (
    <div>
      <div style={{
        font: "9.5px 'IBM Plex Mono', monospace",
        color: '#5f656c',
        letterSpacing: '0.06em',
        textTransform: 'uppercase',
        marginBottom: 4
      }}>
        {label}
      </div>
      <div style={{
        font: "600 18px 'IBM Plex Mono', monospace",
        color: accent || '#e8eaed'
      }}>
        {value}
        {suffix && (
          <span style={{
            fontSize: 11,
            color: '#5f656c',
            fontWeight: 400,
            marginLeft: 2
          }}>
            {suffix}
          </span>
        )}
      </div>
      {sub && (
        <div style={{
          font: "9.5px 'IBM Plex Mono', monospace",
          color: '#4a4f55',
          marginTop: 2
        }}>
          {sub}
        </div>
      )}
    </div>
  );
}

/* ---- Table view --------------------------------------------------------- */
function UniverseTable({ markets, accent, rowPad, onOpen }) {
  const headers = [
    'Market',
    'Status',
    'Result',
    'Poly',
    'Signal',
    'Δ',
    'Conf',
    'Pressure',
    'Evidence',
    'Trend'
  ];

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          {headers.map((h, i) => (
            <th
              key={h}
              style={{
                textAlign: i === 0 ? 'left' : i >= 3 && i <= 6 ? 'right' : 'left',
                font: "9.5px 'IBM Plex Mono', monospace",
                color: '#5f656c',
                letterSpacing: '0.07em',
                textTransform: 'uppercase',
                fontWeight: 500,
                padding: '10px 14px',
                borderBottom: '1px solid var(--border)',
                position: 'sticky',
                top: 0,
                background: 'var(--bg)',
                zIndex: 1,
                whiteSpace: 'nowrap',
              }}
            >
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {markets.map((m) => (
          <tr
            key={m.id}
            onClick={() => onOpen(m)}
            className="ps-row"
            style={{
              cursor: 'pointer',
              borderBottom: '1px solid var(--border)'
            }}
          >
            <td style={{ padding: rowPad, maxWidth: 360 }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                marginBottom: 4
              }}>
                <DomainTag domain={m.domain} />
                <DivergenceBadge delta={m.delta} accent={accent} />
              </div>
              <div style={{
                font: "13.5px 'IBM Plex Sans', sans-serif",
                color: '#e8eaed',
                lineHeight: 1.3,
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}>
                {m.question}
              </div>
            </td>

            <td style={{ padding: rowPad }}>
              <StatusBadge status={m.status} />
            </td>

            <td style={{ padding: rowPad }}>
              {m.scorecard ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <OutcomeBadge outcome={m.scorecard.outcome} />
                  <HitMark ok={m.scorecard.signalCorrect} label="sig" />
                </div>
              ) : (
                <span style={{
                  color: '#5f656c',
                  font: "10px 'IBM Plex Mono', monospace"
                }}>
                  —
                </span>
              )}
            </td>

            <td style={{
              padding: rowPad,
              textAlign: 'right',
              font: "13px 'IBM Plex Mono', monospace",
              color: 'rgba(255,255,255,0.62)'
            }}>
              {pct(m.polyProb)}%
            </td>

            <td style={{
              padding: rowPad,
              textAlign: 'right',
              font: "600 14px 'IBM Plex Mono', monospace",
              color: accent
            }}>
              {pct(m.signalProb)}%
            </td>

            <td style={{ padding: rowPad, textAlign: 'right' }}>
              <DeltaTag delta={m.delta} accent={accent} />
            </td>

            <td style={{
              padding: rowPad,
              textAlign: 'right',
              font: "13px 'IBM Plex Mono', monospace",
              color: '#c4c8cc'
            }}>
              {m.conf}
            </td>

            <td style={{ padding: rowPad }}>
              <PressureChip
                value={m.pressureNow}
                dir={m.pressureDir}
                accent={accent}
              />
            </td>

            <td style={{ padding: rowPad }}>
              <EvidencePip bucket={m.evidenceBucket} />
            </td>

            <td style={{ padding: rowPad }}>
              <Sparkline
                poly={m.series.poly}
                signal={m.series.signal}
                accent={accent}
                w={92}
                h={26}
              />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ---- Card view ---------------------------------------------------------- */
function UniverseCards({ markets, accent, onOpen, density }) {
  const minCardWidth = density === 'compact' ? 300 : 340;

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: `repeat(auto-fill, minmax(${minCardWidth}px, 1fr))`,
      gap: 14,
      padding: 20
    }}>
      {markets.map((m) => (
        <div
          key={m.id}
          onClick={() => onOpen(m)}
          className="ps-card"
          style={{
            background: 'var(--panel)',
            border: '1px solid var(--border)',
            borderRadius: 10,
            padding: 16,
            cursor: 'pointer',
          }}
        >
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: 10
          }}>
            <DomainTag domain={m.domain} />
            <StatusBadge status={m.status} />
          </div>

          <div style={{
            font: "14.5px/1.35 'IBM Plex Sans', sans-serif",
            color: '#e8eaed',
            minHeight: 40,
            marginBottom: 14
          }}>
            {m.question}
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'space-between',
            marginBottom: 12
          }}>
            <div style={{ display: 'flex', gap: 18 }}>
              <div>
                <div style={{
                  font: "9px 'IBM Plex Mono', monospace",
                  color: '#5f656c',
                  letterSpacing: '0.05em',
                  marginBottom: 2
                }}>
                  POLY
                </div>
                <div style={{
                  font: "15px 'IBM Plex Mono', monospace",
                  color: 'rgba(255,255,255,0.6)'
                }}>
                  {pct(m.polyProb)}%
                </div>
              </div>

              <div>
                <div style={{
                  font: "9px 'IBM Plex Mono', monospace",
                  color: '#5f656c',
                  letterSpacing: '0.05em',
                  marginBottom: 2
                }}>
                  SIGNAL
                </div>
                <div style={{
                  font: "600 18px 'IBM Plex Mono', monospace",
                  color: accent
                }}>
                  {pct(m.signalProb)}%
                </div>
              </div>

              <div>
                <div style={{
                  font: "9px 'IBM Plex Mono', monospace",
                  color: '#5f656c',
                  letterSpacing: '0.05em',
                  marginBottom: 4
                }}>
                  Δ
                </div>
                <DeltaTag delta={m.delta} accent={accent} />
              </div>
            </div>

            <Sparkline
              poly={m.series.poly}
              signal={m.series.signal}
              accent={accent}
              w={92}
              h={34}
            />
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            paddingTop: 12,
            borderTop: '1px solid var(--border)'
          }}>
            <EvidencePip bucket={m.evidenceBucket} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{
                font: "10px 'IBM Plex Mono', monospace",
                color: '#5f656c'
              }}>
                conf {m.conf}
              </span>
              <PressureChip
                value={m.pressureNow}
                dir={m.pressureDir}
                accent={accent}
              />
            </div>
          </div>

          {Math.abs(m.delta) >= 0.05 && (
            <div style={{ marginTop: 10 }}>
              <DivergenceBadge delta={m.delta} accent={accent} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

Object.assign(window, { MarketUniverse });
