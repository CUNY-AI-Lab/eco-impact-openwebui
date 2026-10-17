// Port of eco_impact_filter.py `_fmt_num`, `_fmt_range` and `_comparison`.
// CO2-equivalence tiers: (threshold_g, grams_per_unit, singular, plural, emoji).
const CO2_COMPARISONS: ReadonlyArray<readonly [number, number, string, string, string]> = [
  [0.0, 0.2, "breath", "breaths", "🌬️"],
  [1.0, 0.05, "m of walking", "m of walking", "🚶"],
  [10.0, 120.0, "km of driving", "km of driving", "🚗"],
];
export function formatNumber(v: number): string { return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2); }
/** Renders "lo–hi unit", collapsing to "~x unit" when tight. Scales g/Wh down to mg/mWh and, for cumulative totals, up to kg/kWh. */
export function formatRange(lo: number, hi: number, unit: "g" | "Wh" | "mg" | "mWh" | "kg" | "kWh"): string {
  let [l, h] = [lo, hi];
  let u: string = unit;
  if (u === "g" && h < 0.1) [l, h, u] = [l * 1000, h * 1000, "mg"];
  if (u === "Wh" && h < 0.1) [l, h, u] = [l * 1000, h * 1000, "mWh"];
  if (u === "g" && h >= 1000) [l, h, u] = [l / 1000, h / 1000, "kg"];
  if (u === "Wh" && h >= 1000) [l, h, u] = [l / 1000, h / 1000, "kWh"];
  if (l <= 0 || h <= l * 1.02) return `~${formatNumber(h)} ${u}`;
  return `${formatNumber(l)}–${formatNumber(h)} ${u}`;
}
export function comparison(co2GramsMid: number): string {
  // SAFETY: CO2_COMPARISONS is a non-empty literal above.
  let selected = CO2_COMPARISONS[0]!;
  for (const tier of CO2_COMPARISONS) if (co2GramsMid >= tier[0]) selected = tier;
  const [, perUnit, singular, plural, emoji] = selected;
  const value = co2GramsMid / perUnit;
  const label = Math.round(value * 10) / 10 === 1 ? singular : plural;
  return `${emoji} ~${formatNumber(value)} ${label}`;
}
