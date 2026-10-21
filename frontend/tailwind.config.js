/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class", '[data-theme="dark"]'],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Semantic method palette — identical in every chart and legend.
        obsidian: "#090D16",
        panel: "#111827",
        method: {
          m5: "#10B981",
          m5b: "#06B6D4",
          m3: "#6366F1",
          m2: "#F59E0B",
          m4: "#A855F7",
          m1: "#64748B",
          zero: "#F43F5E",
        },
        slot: {
          exploit: "#10B981",
          voi: "#0EA5E9",
          h0: "#F59E0B",
          harvest: "#EAB308",
        },
      },
      fontFamily: {
        sans: ["Inter", "Plus Jakarta Sans", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "0.95rem", letterSpacing: "0.02em" }],
      },
      boxShadow: {
        glass: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 18px 40px -24px rgba(0,0,0,0.9)",
        hero: "0 0 0 1px rgba(16,185,129,0.35), 0 24px 70px -30px rgba(16,185,129,0.55)",
      },
      backgroundImage: {
        grid: "linear-gradient(to right, rgba(148,163,184,0.07) 1px, transparent 1px), linear-gradient(to bottom, rgba(148,163,184,0.07) 1px, transparent 1px)",
      },
      keyframes: {
        pulseSoft: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.55" },
        },
      },
      animation: { "pulse-soft": "pulseSoft 2.4s ease-in-out infinite" },
    },
  },
  plugins: [],
};
