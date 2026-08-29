import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During dev the Vite dev server proxies /api and /auth to the FastAPI backend
// on port 8000, so the frontend can call relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': 'http://localhost:8000',
      '/auth': 'http://localhost:8000',
      '/datasets': 'http://localhost:8000',
      '/tokenizers': 'http://localhost:8000',
      '/training': 'http://localhost:8000',
      '/models': 'http://localhost:8000',
      '/evaluations': 'http://localhost:8000',
      '/memory': 'http://localhost:8000',
      '/inference': 'http://localhost:8000',
      '/chat': 'http://localhost:8000',
      '/dashboard': 'http://localhost:8000',
      '/schedules': 'http://localhost:8000',
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 1500,
  },
})
