// Port of eco_impact_filter.py `_fmt_num`, `_fmt_range` and `_comparison`.
// CO2-equivalence tiers: (threshold_g, grams_per_unit, singular, plural, emoji).
const CO2_COMPARISONS: ReadonlyArray<readonly [number, number, string, string, string]> = [
  [0.0, 0.2, "breath", "breaths", "🌬️"],
  [1.0, 0.05, "m of walking", "m of walking", "🚶"],
  [10.0, 120.0, "km of driving", "km of driving", "🚗"],
];
export function formatNumber(v: number): string { return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2); }
/** The value `formatNumber` would print, so unit choice follows the rounded display rather than the raw value. */
const shown = (v: number): number => Number(formatNumber(v));
/**
 * Renders "lo–hi unit", collapsing to "~x unit" when tight and to "0 unit" when both ends are zero. Scales g/Wh down
 * to mg/mWh and, for cumulative totals, up to kg/kWh, choosing the unit after rounding so 999.6 g prints "~1.00 kg"
 * and 0.09996 g prints "~0.10 g" (never "1000 g" or "100.0 mg").
 */
export function formatRange(lo: number, hi: number, unit: "g" | "Wh" | "mg" | "mWh" | "kg" | "kWh"): string {
  if (lo <= 0 && hi <= 0) return `0 ${unit}`;
  let [l, h] = [lo, hi];
  let u: string = unit;
  if (u === "g" && shown(h * 1000) < 100) [l, h, u] = [l * 1000, h * 1000, "mg"];
  if (u === "Wh" && shown(h * 1000) < 100) [l, h, u] = [l * 1000, h * 1000, "mWh"];
  if (u === "g" && shown(h) >= 1000) [l, h, u] = [l / 1000, h / 1000, "kg"];
  if (u === "Wh" && shown(h) >= 1000) [l, h, u] = [l / 1000, h / 1000, "kWh"];
  if (l <= 0 || h <= l * 1.02) return `~${formatNumber(h)} ${u}`;
  return `${formatNumber(l)}–${formatNumber(h)} ${u}`;
}
export function comparison(co2GramsMid: number): string {
  let selected = CO2_COMPARISONS[0];
  if (selected === undefined) throw new Error("CO2_COMPARISONS must not be empty");
  for (const tier of CO2_COMPARISONS) if (co2GramsMid >= tier[0]) selected = tier;
  const [, perUnit, singular, plural, emoji] = selected;
  const value = co2GramsMid / perUnit;
  const label = Math.round(value * 10) / 10 === 1 ? singular : plural;
  return `${emoji} ~${formatNumber(value)} ${label}`;
}
