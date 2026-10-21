/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Override the API origin for a cross-origin static deploy (default: same origin). */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
