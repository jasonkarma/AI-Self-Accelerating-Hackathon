# ── build ────────────────────────────────────────────────────────────────────
# Built on Railway's Linux builder, never from the committed node_modules: the
# lockfile carries every platform's optional binary, so `npm ci` here resolves
# esbuild/rollup for linux-arm64 rather than inheriting a macOS tree.
FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY . .
# `npm run build` is `tsc --noEmit && vite build`: a type error fails the image,
# which is the same gate `npm run check` applies locally.
RUN npm run build

# ── serve ────────────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine

# `envsubst` ships in the official image, but pinning it makes the template step
# independent of the base image's package list.
RUN apk add --no-cache gettext \
 && rm -f /etc/nginx/conf.d/default.conf

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
# Sourced before the image's own envsubst step: validates and normalises
# API_ORIGIN so a missing value fails with an actionable message rather than an
# nginx syntax error, and a trailing slash is stripped.
COPY docker-entrypoint.d/10-check-env.envsh /docker-entrypoint.d/10-check-env.envsh
RUN chmod +x /docker-entrypoint.d/10-check-env.envsh

# Railway injects PORT at runtime; API_ORIGIN is the backend's public origin.
# Both are substituted by the image's own envsubst bootstrap, so one image
# serves any backend URL without a rebuild.
ENV PORT=8080
EXPOSE 8080
