import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // Проксирует запросы к backend через тот же origin, что и
    // фронтенд - избавляет от необходимости настраивать CORS для
    // локальной разработки (см. комментарий в src/api/client.js).
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})