/**
 * ResultCard
 * ──────────
 * Displays the full prediction result: status badge, confidence bars,
 * leaf similarity score, model name, and action message.
 */
import StatusBadge from "./StatusBadge.jsx";
import ConfidenceBar from "./ConfidenceBar.jsx";

export default function ResultCard({ result, status, preview, onReset }) {
  const isSuccess = status === "success";
  const isRejected = status === "rejected";

  return (
    <div
      style={{
        animation: "fadeUp 0.5s cubic-bezier(0.4,0,0.2,1) both",
        background: "var(--white)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-lg)",
        overflow: "hidden",
        maxWidth: "640px",
        width: "100%",
      }}
    >
      {/* ── Image preview strip ──────────────────────────── */}
      {preview && (
        <div
          style={{
            width: "100%",
            height: "220px",
            overflow: "hidden",
            position: "relative",
          }}
        >
          <img
            src={preview}
            alt="Uploaded leaf"
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              display: "block",
            }}
          />
          {/* Gradient overlay for text legibility */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              background:
                "linear-gradient(to top, rgba(26,46,26,0.6) 0%, transparent 60%)",
            }}
          />
          {/* Status badge overlaid on image */}
          <div
            style={{
              position: "absolute",
              bottom: "16px",
              left: "20px",
            }}
          >
            <StatusBadge
              status={isRejected ? "REJECTED" : result?.prediction}
            />
          </div>
        </div>
      )}

      {/* ── Content area ────────────────────────────────── */}
      <div style={{ padding: "28px 32px 32px" }}>
        {/* ── REJECTED state ────────────────────────────── */}
        {isRejected && (
          <>
            <h2
              style={{
                fontFamily: "'Playfair Display', serif",
                fontSize: "1.5rem",
                color: "var(--amber)",
                marginBottom: "10px",
              }}
            >
              Image Rejected
            </h2>
            <p
              style={{
                color: "var(--moss)",
                lineHeight: 1.7,
                fontSize: "0.9rem",
                marginBottom: "18px",
              }}
            >
              {result?.reason ||
                "This image does not appear to be a plant leaf."}
            </p>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "10px",
                padding: "12px 16px",
                background: "#fff8e1",
                borderRadius: "var(--radius)",
                border: "1px solid #ffe082",
                marginBottom: "24px",
              }}
            >
              <span style={{ fontSize: "1.2rem" }}>◈</span>
              <div>
                <div
                  style={{
                    fontSize: "0.78rem",
                    color: "#e65100",
                    fontWeight: 600,
                    letterSpacing: "0.05em",
                    textTransform: "uppercase",
                  }}
                >
                  Similarity Score
                </div>
                <div
                  style={{
                    fontFamily: "'Playfair Display', serif",
                    fontSize: "1.3rem",
                    color: "var(--forest)",
                  }}
                >
                  {result?.leaf_similarity !== undefined
                    ? `${(result.leaf_similarity * 100).toFixed(1)}%`
                    : "—"}
                  <span
                    style={{
                      fontSize: "0.8rem",
                      color: "var(--sage)",
                      marginLeft: "6px",
                    }}
                  >
                    (threshold:{" "}
                    {result?.threshold_used !== undefined
                      ? `${(result.threshold_used * 100).toFixed(0)}%`
                      : "68%"}
                    )
                  </span>
                </div>
              </div>
            </div>
          </>
        )}

        {/* ── SUCCESS state ─────────────────────────────── */}
        {isSuccess && (
          <>
            {/* Heading */}
            <h2
              style={{
                fontFamily: "'Playfair Display', serif",
                fontSize: "1.6rem",
                color: result.prediction === "HEALTHY" ? "#1b5e20" : "#880e4f",
                marginBottom: "6px",
              }}
            >
              {result.prediction === "HEALTHY"
                ? "Leaf is Healthy"
                : "Disease Detected"}
            </h2>
            <p
              style={{
                color: "var(--sage)",
                fontSize: "0.88rem",
                marginBottom: "24px",
              }}
            >
              {result.message}
            </p>

            {/* Confidence bars */}
            <div style={{ marginBottom: "24px" }}>
              <ConfidenceBar
                label="Healthy"
                value={result.healthy_prob}
                color="#4caf50"
                delay={0}
              />
              <ConfidenceBar
                label="Diseased"
                value={result.diseased_prob}
                color="#e91e63"
                delay={120}
              />
            </div>

            {/* Meta row */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "12px",
                marginBottom: "24px",
              }}
            >
              {[
                {
                  icon: "◉",
                  label: "Confidence",
                  value: `${(result.confidence * 100).toFixed(1)}%`,
                },
                {
                  icon: "◈",
                  label: "Leaf Similarity",
                  value: `${(result.leaf_similarity * 100).toFixed(1)}%`,
                },
                { icon: "◆", label: "Model", value: result.model_used },
                {
                  icon: "◇",
                  label: "File",
                  value:
                    result.filename?.length > 18
                      ? result.filename.slice(0, 16) + "…"
                      : result.filename || "—",
                },
              ].map(({ icon, label, value }) => (
                <div
                  key={label}
                  style={{
                    background: "var(--cream)",
                    borderRadius: "var(--radius)",
                    padding: "12px 14px",
                    border: "1px solid var(--parchment)",
                  }}
                >
                  <div
                    style={{
                      fontSize: "0.72rem",
                      color: "var(--sage)",
                      fontWeight: 500,
                      letterSpacing: "0.06em",
                      textTransform: "uppercase",
                      marginBottom: "4px",
                    }}
                  >
                    {icon} {label}
                  </div>
                  <div
                    style={{
                      fontFamily: "'DM Sans', sans-serif",
                      fontWeight: 600,
                      fontSize: "0.94rem",
                      color: "var(--forest)",
                    }}
                  >
                    {value}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* ── Analyse another button ──────────────────── */}
        <button
          onClick={onReset}
          style={{
            width: "100%",
            padding: "14px",
            background: "var(--forest)",
            color: "var(--cream)",
            border: "none",
            borderRadius: "var(--radius)",
            fontFamily: "'DM Sans', sans-serif",
            fontSize: "0.9rem",
            fontWeight: 600,
            letterSpacing: "0.04em",
            cursor: "pointer",
            transition: "background var(--transition)",
          }}
          onMouseEnter={(e) => (e.target.style.background = "var(--moss)")}
          onMouseLeave={(e) => (e.target.style.background = "var(--forest)")}
        >
          Analyse Another Leaf
        </button>
      </div>
    </div>
  );
}
