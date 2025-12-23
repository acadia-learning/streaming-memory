import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat': 'https://bryanhoulton--aryan-tutor-streaming-tutorservice-serve.modal.run'
    }
  }
})

