import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fakeApiPlugin } from "./src/dev/fakeApiPlugin";

// https://vite.dev/config/
export default defineConfig({
  base: process.env.VITE_BASE ?? "/",
  plugins: [react(), fakeApiPlugin()],
  server: {
    port: 5173,
  },
});
