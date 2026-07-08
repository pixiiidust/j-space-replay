import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fakeApiPlugin } from "./src/dev/fakeApiPlugin";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), fakeApiPlugin()],
  server: {
    port: 5173,
  },
});
