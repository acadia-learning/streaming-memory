import { defineConfig } from 'vite';

import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy for local development only (not used in production)
      "/health": {
        target:
          "https://bryanhoulton--streaming-memory-familyassistant-serve.modal.run",
        changeOrigin: true,
      },
      "/chat": {
        target:
          "https://bryanhoulton--streaming-memory-familyassistant-serve.modal.run",
        changeOrigin: true,
      },
    },
  },
});
