import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const proxyTarget = 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '127.0.0.1',
    proxy: {
      '/metadata': { target: proxyTarget, changeOrigin: true },
      '/metrics': { target: proxyTarget, changeOrigin: true },
      '/visuals': { target: proxyTarget, changeOrigin: true },
      '/runs': { target: proxyTarget, changeOrigin: true },
      '/select': { target: proxyTarget, changeOrigin: true },
      '/chat': { target: proxyTarget, changeOrigin: true },
      '/data': { target: proxyTarget, changeOrigin: true },
      '/train': { target: proxyTarget, changeOrigin: true },
      '/tokenizer': { target: proxyTarget, changeOrigin: true },
    },
  },
});
