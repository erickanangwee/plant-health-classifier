/**
 * ConfidenceBar
 * ─────────────
 * Animated horizontal bar showing probability for each class.
 */
import { useEffect, useRef } from "react";

export default function ConfidenceBar({ label, value, color, delay = 0 }) {
  const barRef = useRef(null);

  useEffect(() => {
    const bar = barRef.current;
    if (!bar) return;
    // Small timeout ensures the CSS transition triggers after mount
    const id = setTimeout(() => {
      bar.style.width = `${(value * 100).toFixed(1)}%`;
    }, delay);
    return () => clearTimeout(id);
  }, [value, delay]);

  const pct = (value * 100).toFixed(1);

  return (
    <div style={{ marginBottom: "12px" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: "6px",
        }}
      >
        <span
          style={{
            fontFamily: "'DM Sans', sans-serif",
            fontSize: "0.82rem",
            fontWeight: 500,
            color: "var(--moss)",
            letterSpacing: "0.02em",
          }}
        >
          {label}
        </span>
        <span
          style={{
            fontFamily: "'Playfair Display', serif",
            fontSize: "1rem",
            fontWeight: 600,
            color: color,
          }}
        >
          {pct}%
        </span>
      </div>
      <div
        style={{
          height: "8px",
          background: "rgba(26,46,26,0.08)",
          borderRadius: "100px",
          overflow: "hidden",
        }}
      >
        <div
          ref={barRef}
          style={{
            height: "100%",
            width: "0%",
            background: color,
            borderRadius: "100px",
            transition: `width 0.9s cubic-bezier(0.4, 0, 0.2, 1) ${delay}ms`,
          }}
        />
      </div>
    </div>
  );
}
