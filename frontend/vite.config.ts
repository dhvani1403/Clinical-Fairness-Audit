import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs separately (uvicorn on :8000); in production FastAPI
// serves the built files from frontend/dist, so /api is same-origin.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
});
