/**
 * StatusBadge
 * ───────────
 * Displays HEALTHY, DISEASED, or REJECTED with appropriate colour coding.
 */
export default function StatusBadge({ status }) {
  const variants = {
    HEALTHY: {
      bg: "#e8f5e9",
      color: "#1b5e20",
      border: "#4caf50",
      icon: "✦",
      label: "HEALTHY",
    },
    DISEASED: {
      bg: "#fce4ec",
      color: "#880e4f",
      border: "#e91e63",
      icon: "⚠",
      label: "DISEASED",
    },
    REJECTED: {
      bg: "#fff8e1",
      color: "#e65100",
      border: "#ff9800",
      icon: "◈",
      label: "NOT A LEAF",
    },
  };

  const v = variants[status] || variants.REJECTED;

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        padding: "6px 16px",
        borderRadius: "100px",
        backgroundColor: v.bg,
        color: v.color,
        border: `1.5px solid ${v.border}`,
        fontFamily: "'DM Sans', sans-serif",
        fontWeight: 600,
        fontSize: "0.78rem",
        letterSpacing: "0.08em",
        textTransform: "uppercase",
      }}
    >
      <span style={{ fontSize: "0.9em" }}>{v.icon}</span>
      {v.label}
    </span>
  );
}
