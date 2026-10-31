/* PolySignal — Constants ---------------------------------------------------- */

// Spacing
export const SPACING = {
  COMPACT_GAP: 12,
  REGULAR_GAP: 16,
  COMPACT_PADDING: '18px 22px 40px',
  REGULAR_PADDING: '24px 28px 56px',
  COMPACT_ROW: '9px 14px',
  REGULAR_ROW: '13px 14px',
};

// Chart dimensions
export const CHART = {
  SPARKLINE_WIDTH: 120,
  SPARKLINE_HEIGHT: 30,
  SPARKLINE_TABLE_WIDTH: 92,
  SPARKLINE_TABLE_HEIGHT: 26,
  SPARKLINE_CARD_WIDTH: 92,
  SPARKLINE_CARD_HEIGHT: 34,
  DETAIL_HEIGHT: 330,
  PADDING: {
    LEFT: 44,
    RIGHT: 16,
    TOP: 16,
    BOTTOM: 28,
  },
  PRESSURE_HEIGHT: 54,
  PRESSURE_GAP: 10,
};

// Colors
export const COLORS = {
  ACCENT_GREEN: '#34d399',
  ACCENT_BLUE: '#5b9dff',
  ACCENT_AMBER: '#f5a524',

  // Stance colors
  STANCE_YES: '#34d399', // Uses accent
  STANCE_NO: '#f76d6d',
  STANCE_NEUTRAL: '#9aa0a6',
  STANCE_AMBIGUOUS: '#7d838a',

  // Domain colors
  DOMAIN_MACRO: '#5b9dff',
  DOMAIN_POLITICS: '#c084fc',
  DOMAIN_CRYPTO: '#f5a524',
  DOMAIN_AI: '#34d399',
  DOMAIN_SPORTS: '#f76d6d',
  DOMAIN_CULTURE: '#f2a0c4',

  // UI colors (references to CSS variables)
  BACKGROUND: 'var(--bg)',
  PANEL: 'var(--panel)',
  BORDER: 'var(--border)',
  ACCENT: 'var(--accent)',
};

// Typography
export const FONT = {
  MONO: "'IBM Plex Mono', monospace",
  SANS: "'IBM Plex Sans', sans-serif",
};

// Thresholds
export const THRESHOLDS = {
  HIGH_DIVERGENCE: 0.1,
  MEDIUM_DIVERGENCE: 0.05,
  HIGH_CONFIDENCE: 80,
  MEDIUM_CONFIDENCE: 65,
  STRONG_PRESSURE: 65,
};

// Platform glyphs
export const PLATFORM_GLYPH = {
  X: '𝕏',
  Reddit: 'r/',
  Farcaster: 'fc',
  News: '◆',
};

// Domain metadata
export const DOMAIN_META = {
  MACRO: { label: 'Macro', dot: COLORS.DOMAIN_MACRO },
  POLITICS: { label: 'Politics', dot: COLORS.DOMAIN_POLITICS },
  CRYPTO: { label: 'Crypto', dot: COLORS.DOMAIN_CRYPTO },
  AI: { label: 'AI', dot: COLORS.DOMAIN_AI },
  SPORTS: { label: 'Sports', dot: COLORS.DOMAIN_SPORTS },
  CULTURE: { label: 'Culture', dot: COLORS.DOMAIN_CULTURE },
};

// Accent options
export const ACCENTS = {
  [COLORS.ACCENT_GREEN]: 'Terminal green',
  [COLORS.ACCENT_BLUE]: 'Electric blue',
  [COLORS.ACCENT_AMBER]: 'Amber',
};

// Animation durations (ms)
export const ANIMATION = {
  FAST: 120,
  NORMAL: 150,
  TRANSITION: '0.12s',
};

// Resize observer debounce
export const RESIZE_DEBOUNCE = 10;

// Default tweak settings
export const TWEAK_DEFAULTS = {
  accent: COLORS.ACCENT_GREEN,
  density: 'regular',
  bgDepth: 'deep',
  chartPressure: true,
  chartArea: false,
};

// Evidence bucket config
export const EVIDENCE_BUCKETS = {
  High: { n: 3, c: COLORS.ACCENT_GREEN },
  Moderate: { n: 2, c: COLORS.ACCENT_AMBER },
  Sparse: { n: 1, c: COLORS.STANCE_NO },
  None: { n: 0, c: '#5f656c' },
};

// Status config
export const STATUS_MAP = {
  ACTIVE: { c: COLORS.ACCENT_GREEN, bg: 'rgba(52,211,153,0.1)', label: 'ACTIVE' },
  CLOSING: { c: COLORS.ACCENT_AMBER, bg: 'rgba(245,165,36,0.1)', label: 'CLOSING' },
  RESOLVED: { c: COLORS.STANCE_AMBIGUOUS, bg: 'rgba(125,131,138,0.12)', label: 'RESOLVED' },
};
