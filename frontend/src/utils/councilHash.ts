/**
 * Deterministic 32-bit string hash, used as the key for the council-text
 * translation table.
 *
 * Why a hash and not the string itself: the seven Chinese memos in the Seed-21
 * ledger are 0.8–3.7 kB each. Keying the English rendering on the raw text would
 * mean the translation module carries ~15 kB of Chinese source that exists for no
 * reason other than to be looked up. Keying on a hash lets the translations live
 * as readable English, while `council-zh-source.json` keeps the originals in one
 * place for the "Raw ledger" toggle.
 *
 * FNV-1a is chosen because it is *exactly* specified in a few lines: the extractor
 * script and the browser must agree on every key, and a hash that needs a library
 * to reproduce is a hash that will silently drift between the two.
 */

/** FNV-1a (32-bit) over UTF-16 code units, as 8 lowercase hex characters. */
export function councilHash(text: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    // Multiply by the FNV prime (16777619) with 32-bit wraparound.
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}
