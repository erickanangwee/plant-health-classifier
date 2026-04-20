/**
 * UploadZone
 * ──────────
 * Drag-and-drop or click-to-upload area with visual feedback.
 * Validates file type and size before passing to the predict hook.
 */
import { useState, useCallback, useRef } from "react";

const MAX_SIZE_MB = 10;
const ACCEPTED = ["image/jpeg", "image/jpg", "image/png", "image/webp"];

export default function UploadZone({ onFile, disabled }) {
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState(null);
  const inputRef = useRef(null);

  const validate = (file) => {
    if (!ACCEPTED.includes(file.type)) {
      setFileError("Please upload a JPEG, PNG, or WebP image.");
      return false;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setFileError(`File too large. Maximum size is ${MAX_SIZE_MB} MB.`);
      return false;
    }
    setFileError(null);
    return true;
  };

  const handleFile = useCallback(
    (file) => {
      if (file && validate(file)) onFile(file);
    },
    [onFile],
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const onDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };
  const onDragLeave = () => setDragging(false);
  const onInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    // Reset so same file can be re-uploaded
    e.target.value = "";
  };

  return (
    <div style={{ width: "100%", maxWidth: "640px" }}>
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload plant leaf image"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) =>
          e.key === "Enter" && !disabled && inputRef.current?.click()
        }
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        style={{
          position: "relative",
          border: `2px dashed ${dragging ? "var(--fern)" : "var(--mint)"}`,
          borderRadius: "var(--radius-lg)",
          padding: "56px 40px",
          textAlign: "center",
          cursor: disabled ? "not-allowed" : "pointer",
          background: dragging
            ? "rgba(61,107,61,0.06)"
            : "rgba(255,255,255,0.7)",
          backdropFilter: "blur(8px)",
          transition: "all var(--transition)",
          boxShadow: dragging ? "var(--shadow-md)" : "var(--shadow-sm)",
          transform: dragging ? "scale(1.01)" : "scale(1)",
          opacity: disabled ? 0.5 : 1,
        }}
      >
        {/* Icon */}
        <div
          style={{
            width: "72px",
            height: "72px",
            margin: "0 auto 20px",
            background: "linear-gradient(135deg, var(--fern), var(--sage))",
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "2rem",
            animation: dragging
              ? "pulse-ring 1.5s ease-in-out infinite"
              : "none",
            boxShadow: "var(--shadow-sm)",
          }}
        >
          🌿
        </div>

        <h3
          style={{
            fontFamily: "'Playfair Display', serif",
            fontSize: "1.35rem",
            color: "var(--forest)",
            marginBottom: "8px",
          }}
        >
          {dragging ? "Release to analyse" : "Upload a plant leaf"}
        </h3>

        <p
          style={{
            color: "var(--sage)",
            fontSize: "0.88rem",
            lineHeight: 1.6,
          }}
        >
          Drag & drop your image here, or{" "}
          <span
            style={{
              color: "var(--fern)",
              fontWeight: 600,
              textDecoration: "underline",
            }}
          >
            click to browse
          </span>
        </p>

        <p
          style={{
            color: "var(--mint)",
            fontSize: "0.78rem",
            marginTop: "10px",
          }}
        >
          JPEG · PNG · WebP · up to {MAX_SIZE_MB} MB
        </p>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          onChange={onInputChange}
          style={{ display: "none" }}
          disabled={disabled}
        />
      </div>

      {/* File validation error */}
      {fileError && (
        <p
          style={{
            marginTop: "10px",
            color: "var(--rust)",
            fontSize: "0.84rem",
            textAlign: "center",
            animation: "fadeIn 0.3s ease both",
          }}
        >
          ⚠ {fileError}
        </p>
      )}
    </div>
  );
}
