import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Dev proxy: the console talks to the FastAPI backend on its own origin path
 * (`/api/...`), so no CORS preflight is needed in development and the same
 * relative URLs keep working behind any reverse proxy in production.
 *
 * The backend already sends `Access-Control-Allow-Origin: *`, so a static build
 * served from a different origin (e.g. `vite preview`, S3, nginx) can also call
 * it directly by setting `VITE_API_BASE=http://127.0.0.1:8000` at build time.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: false,
        // SSE (`/api/stream-round`) must not be buffered by the proxy.
        ws: false,
      },
    },
  },
  preview: { port: 4173 },
  build: {
    outDir: "dist",
    sourcemap: true,
    target: "es2022",
    chunkSizeWarningLimit: 900,
  },
});
