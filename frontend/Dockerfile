# Образ клиентской части (этап 9): сборка Vite в Node, раздача статики и прокси /api в nginx.
# Контекст сборки - папка frontend:  docker build -t vehicle-reid-frontend ./frontend
# nginx проксирует /api и /health на сервис backend (имя сервиса в docker-compose), поэтому браузер видит
# фронтенд и API с одного адреса, как в разработке (Vite-прокси), и CORS не нужен.

# --- этап 1: сборка. Debian-вариант Node (не alpine): у Vite 8 нативные модули под glibc ставятся без сюрпризов.
FROM node:22-slim AS build
WORKDIR /app
# package-lock.json копируется первым: пока зависимости не менялись, слой npm ci берётся из кэша.
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# --- этап 2: nginx только со статикой. Node и исходники в итоговый образ не попадают.
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ > /dev/null || exit 1