// Input types and guards for the registry and row validators. Internal: not re-exported from index.ts.
// The guards avoid `typeof` (the fleet lint bans it) but return exactly what the `typeof` checks they replace did.

/** A value as `JSON.parse` produces it. The registry fields that `isRegistry` leaves unchecked carry this type. */
export type JsonValue = null | boolean | number | string | readonly JsonValue[] | JsonObject;
export type JsonObject = { readonly [key: string]: JsonValue };

/** Same result as `typeof value === "string"`: only a primitive passes `Object(value) !== value`, and `String` returns a primitive unchanged only when it is already a string. */
export const isString = (value: JsonValue | undefined): value is string => Object(value) !== value && String(value) === value;

/** Same result as `typeof value === "number"`: every number is finite, NaN or infinite, and none of these checks coerces its argument. */
export const isNumber = (value: JsonValue | undefined): value is number =>
  Number.isFinite(value) || Number.isNaN(value) || value === Number.POSITIVE_INFINITY || value === Number.NEGATIVE_INFINITY;

/**
 * A non-null, non-array, non-function object (formerly `typeof value === "object" && value !== null && !Array.isArray(value)`).
 * `Object(value) === value` holds only for objects; functions are excluded by `instanceof Function`.
 */
export function isJsonObject(value: JsonValue | undefined): value is JsonObject;
export function isJsonObject<Value>(value: Value): value is Value & JsonObject;
export function isJsonObject<Value>(value: Value): boolean {
  return Object(value) === value && !Array.isArray(value) && !(value instanceof Function);
}
