/* PolySignal — Utility functions -------------------------------------------- */

/**
 * Converts a decimal probability to a percentage (0..1 -> 0..100)
 * @param {number} n - Probability value (0 to 1)
 * @returns {number} - Percentage value (0 to 100)
 */
export function pct(n) {
  return Math.round(n * 100);
}

/**
 * Formats a delta as signed percentage points
 * @param {number} n - Delta value
 * @returns {string} - Formatted string like "+5 pts" or "-3 pts"
 */
export function fmtSignedPts(n) {
  const v = Math.round(n * 100);
  return (v > 0 ? '+' : '') + v + ' pts';
}

/**
 * Formats a number as percentage with sign
 * @param {number} n - Number to format
 * @returns {string} - Formatted percentage like "+12%" or "-5%"
 */
export function formatPercent(n) {
  const v = Math.round(n * 100);
  return (v > 0 ? '+' : '') + v + '%';
}

/**
 * Formats a signed number
 * @param {number} n - Number to format
 * @returns {string} - Number with sign prefix
 */
export function formatSigned(n) {
  return (n >= 0 ? '+' : '') + n;
}

/**
 * Formats large numbers with locale separators
 * @param {number} n - Number to format
 * @returns {string} - Formatted number like "1,234,567"
 */
export function formatLargeNumber(n) {
  return n.toLocaleString();
}

/**
 * Clamps a value between min and max
 * @param {number} value - Value to clamp
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {number} - Clamped value
 */
export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

/**
 * Interpolates between keyframe data points
 * @param {Array<[number, number]>} keyframes - Array of [x, y] keyframe pairs
 * @param {number} numSamples - Number of samples to generate
 * @returns {Array<number>} - Interpolated y values
 */
export function interpolateKeyframes(keyframes, numSamples) {
  const out = [];
  const maxX = keyframes[keyframes.length - 1][0];

  for (let i = 0; i < numSamples; i++) {
    const x = (i / (numSamples - 1)) * maxX;
    let y = keyframes[0][1];

    for (let j = 0; j < keyframes.length - 1; j++) {
      const [x0, y0] = keyframes[j];
      const [x1, y1] = keyframes[j + 1];

      if (x >= x0 && x <= x1) {
        y = y0 + (y1 - y0) * ((x - x0) / (x1 - x0));
        break;
      }
    }
    out.push(y);
  }

  return out;
}

/**
 * Gets stance color based on stance value
 * @param {string} stance - Stance value (YES, NO, NEUTRAL, AMBIGUOUS)
 * @param {string} accent - Accent color for YES stance
 * @returns {string} - Color code
 */
export function getStanceColor(stance, accent = '#34d399') {
  const colorMap = {
    YES: accent,
    NO: '#f76d6d',
    NEUTRAL: '#9aa0a6',
    AMBIGUOUS: '#7d838a',
  };
  return colorMap[stance] || '#9aa0a6';
}

/**
 * Determines signal strength based on confidence
 * @param {number} confidence - Confidence value (0-100)
 * @returns {string} - Strength label
 */
export function getSignalStrength(confidence) {
  if (confidence >= 80) return 'Strong';
  if (confidence >= 65) return 'Moderate';
  return 'Weak';
}
