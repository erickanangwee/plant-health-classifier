/**
 * App.jsx
 * ───────
 * Root application component. Manages the page-level layout and
 * orchestrates the upload → loading → result flow.
 */
import { useEffect } from "react";
import UploadZone from "./components/UploadZone.jsx";
import ResultCard from "./components/ResultCard.jsx";
import { usePrediction, STATES } from "./hooks/usePrediction.js";

const API_BASE = import.meta.env.VITE_API_URL || "";

export default function App() {
  const { state, result, error, preview, predict, reset } = usePrediction();

  const isIdle = state === STATES.IDLE;
  const isLoading = state === STATES.LOADING;
  const isSuccess = state === STATES.SUCCESS;
  const isRejected = state === STATES.REJECTED;
  const isError = state === STATES.ERROR;
  const showResult = isSuccess || isRejected;

  // Scroll result card into view on mobile
  useEffect(() => {
    if (showResult || isError) {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }, [showResult, isError]);

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* ── Header ──────────────────────────────────────────── */}
      <header
        style={{
          padding: "24px 32px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: "1px solid rgba(168,197,160,0.3)",
          backdropFilter: "blur(12px)",
          background: "rgba(245,240,232,0.85)",
          position: "sticky",
          top: 0,
          zIndex: 100,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <span style={{ fontSize: "1.5rem" }}>🌿</span>
          <div>
            <h1
              style={{
                fontFamily: "'Playfair Display', serif",
                fontSize: "1.2rem",
                color: "var(--forest)",
                lineHeight: 1,
              }}
            >
              PlantGuard
            </h1>
            <p
              style={{
                fontSize: "0.7rem",
                color: "var(--sage)",
                marginTop: "2px",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
              }}
            >
              Leaf Health Classifier
            </p>
          </div>
        </div>

        <a
          href={`${API_BASE}/docs`}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            padding: "8px 16px",
            background: "transparent",
            color: "var(--fern)",
            border: "1.5px solid var(--mint)",
            borderRadius: "100px",
            fontSize: "0.78rem",
            fontWeight: 600,
            letterSpacing: "0.04em",
            textDecoration: "none",
            transition: "all var(--transition)",
          }}
          onMouseEnter={(e) => {
            e.target.style.background = "var(--fern)";
            e.target.style.color = "var(--cream)";
          }}
          onMouseLeave={(e) => {
            e.target.style.background = "transparent";
            e.target.style.color = "var(--fern)";
          }}
        >
          API Docs
        </a>
      </header>

      {/* ── Main content ────────────────────────────────────── */}
      <main
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          padding: "60px 24px 80px",
        }}
      >
        {/* Hero text — only shown on idle */}
        {isIdle && (
          <div
            style={{
              textAlign: "center",
              marginBottom: "48px",
              animation: "fadeUp 0.6s cubic-bezier(0.4,0,0.2,1) both",
            }}
          >
            <h2
              style={{
                fontFamily: "'Playfair Display', serif",
                fontSize: "clamp(2rem, 5vw, 3rem)",
                color: "var(--forest)",
                lineHeight: 1.15,
                marginBottom: "16px",
              }}
            >
              Is your plant leaf
              <br />
              <em style={{ color: "var(--fern)", fontStyle: "italic" }}>
                healthy?
              </em>
            </h2>
            <p
              style={{
                color: "var(--sage)",
                fontSize: "1rem",
                maxWidth: "480px",
                margin: "0 auto",
                lineHeight: 1.7,
              }}
            >
              Upload a photo of any plant leaf. Our AI analyses it in seconds
              and tells you whether it shows signs of disease or pest
              infestation.
            </p>
          </div>
        )}

        {/* Upload zone — shown when idle or loading */}
        {(isIdle || isLoading) && (
          <div
            style={{
              width: "100%",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "24px",
            }}
          >
            <UploadZone onFile={predict} disabled={isLoading} />

            {/* Loading state overlay */}
            {isLoading && (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: "12px",
                  animation: "fadeIn 0.4s ease both",
                }}
              >
                {/* Preview thumbnail while loading */}
                {preview && (
                  <div
                    style={{
                      width: "120px",
                      height: "120px",
                      borderRadius: "var(--radius)",
                      overflow: "hidden",
                      boxShadow: "var(--shadow-md)",
                      border: "3px solid var(--mint)",
                    }}
                  >
                    <img
                      src={preview}
                      alt="Uploading"
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "cover",
                      }}
                    />
                  </div>
                )}
                <div
                  style={{ display: "flex", alignItems: "center", gap: "10px" }}
                >
                  <div
                    style={{
                      width: "18px",
                      height: "18px",
                      border: "2.5px solid var(--mint)",
                      borderTopColor: "var(--fern)",
                      borderRadius: "50%",
                      animation: "spin 0.8s linear infinite",
                    }}
                  />
                  <span
                    style={{
                      color: "var(--moss)",
                      fontSize: "0.9rem",
                      fontWeight: 500,
                    }}
                  >
                    Analysing leaf…
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Result card */}
        {showResult && (
          <ResultCard
            result={result}
            status={state}
            preview={preview}
            onReset={reset}
          />
        )}

        {/* Error state */}
        {isError && (
          <div
            style={{
              maxWidth: "520px",
              width: "100%",
              padding: "28px 32px",
              background: "var(--white)",
              borderRadius: "var(--radius-lg)",
              boxShadow: "var(--shadow-md)",
              textAlign: "center",
              animation: "fadeUp 0.4s ease both",
            }}
          >
            <div style={{ fontSize: "2.5rem", marginBottom: "12px" }}>⚠</div>
            <h3
              style={{
                fontFamily: "'Playfair Display', serif",
                fontSize: "1.3rem",
                color: "var(--rust)",
                marginBottom: "8px",
              }}
            >
              Something went wrong
            </h3>
            <p
              style={{
                color: "var(--sage)",
                fontSize: "0.9rem",
                lineHeight: 1.6,
                marginBottom: "24px",
              }}
            >
              {error}
            </p>
            <button
              onClick={reset}
              style={{
                padding: "12px 28px",
                background: "var(--forest)",
                color: "var(--cream)",
                border: "none",
                borderRadius: "var(--radius)",
                fontFamily: "'DM Sans', sans-serif",
                fontSize: "0.9rem",
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              Try Again
            </button>
          </div>
        )}

        {/* How it works — shown on idle only */}
        {isIdle && (
          <div
            style={{
              marginTop: "80px",
              width: "100%",
              maxWidth: "640px",
              animation: "fadeUp 0.7s 0.2s cubic-bezier(0.4,0,0.2,1) both",
            }}
          >
            <h3
              style={{
                fontFamily: "'Playfair Display', serif",
                fontSize: "1.1rem",
                color: "var(--forest)",
                marginBottom: "20px",
                textAlign: "center",
              }}
            >
              How it works
            </h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: "16px",
              }}
            >
              {[
                {
                  step: "01",
                  icon: "📷",
                  title: "Upload",
                  desc: "Drag & drop a JPEG, PNG, or WebP photo of any plant leaf.",
                },
                {
                  step: "02",
                  icon: "🔬",
                  title: "Analyse",
                  desc: "EfficientNet-B0 extracts features; the champion classifier predicts.",
                },
                {
                  step: "03",
                  icon: "📋",
                  title: "Result",
                  desc: "Get HEALTHY or DISEASED with confidence score and probability breakdown.",
                },
              ].map(({ step, icon, title, desc }) => (
                <div
                  key={step}
                  style={{
                    background: "rgba(255,255,255,0.7)",
                    borderRadius: "var(--radius)",
                    padding: "20px 18px",
                    border: "1px solid var(--parchment)",
                    backdropFilter: "blur(8px)",
                  }}
                >
                  <div
                    style={{
                      fontSize: "1.6rem",
                      marginBottom: "10px",
                    }}
                  >
                    {icon}
                  </div>
                  <div
                    style={{
                      fontSize: "0.65rem",
                      color: "var(--sage)",
                      letterSpacing: "0.1em",
                      textTransform: "uppercase",
                      marginBottom: "4px",
                      fontWeight: 600,
                    }}
                  >
                    Step {step}
                  </div>
                  <h4
                    style={{
                      fontFamily: "'Playfair Display', serif",
                      fontSize: "0.95rem",
                      color: "var(--forest)",
                      marginBottom: "6px",
                    }}
                  >
                    {title}
                  </h4>
                  <p
                    style={{
                      color: "var(--sage)",
                      fontSize: "0.78rem",
                      lineHeight: 1.55,
                    }}
                  >
                    {desc}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* ── Footer ──────────────────────────────────────────── */}
      <footer
        style={{
          padding: "20px 32px",
          textAlign: "center",
          borderTop: "1px solid rgba(168,197,160,0.2)",
          color: "var(--sage)",
          fontSize: "0.78rem",
        }}
      >
        PlantGuard · Trained on PlantDoc (2,568 images · 38 classes) ·{" "}
        <a
          href={`${API_BASE}/health`}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "var(--fern)", textDecoration: "none" }}
        >
          API status
        </a>
      </footer>
    </div>
  );
}
