/**
 * inject-env.js
 * ─────────────
 * Reads VITE_API_URL from the OS environment (set by Replit Secrets)
 * and writes it into frontend/.env.local so Vite picks it up at build.
 * Run before `vite build` in Replit environments.
 */
import { writeFileSync } from "fs";

const apiUrl = process.env.VITE_API_URL || "";
const content = `VITE_API_URL=${apiUrl}\n`;

writeFileSync(".env.local", content);
console.log(`[inject-env] VITE_API_URL="${apiUrl}" written to .env.local`);
