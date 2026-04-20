/**
 * usePrediction
 * ─────────────
 * Custom React hook that manages the full lifecycle of an image
 * prediction request:  idle → loading → success / error / rejected
 */
import { useState, useCallback } from "react";
import axios from "axios";

// API base URL: set VITE_API_URL in .env.local for production,
// leave empty to use the Vite dev proxy.
const API_BASE = import.meta.env.VITE_API_URL || "";

export const STATES = {
  IDLE: "idle",
  LOADING: "loading",
  SUCCESS: "success",
  REJECTED: "rejected", // 422 — leaf guard rejected the image
  ERROR: "error",
};

export function usePrediction() {
  const [state, setState] = useState(STATES.IDLE);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);

  const predict = useCallback(async (file) => {
    // Generate local preview URL
    const previewUrl = URL.createObjectURL(file);
    setPreview(previewUrl);
    setState(STATES.LOADING);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(`${API_BASE}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 30000, // 30s — EfficientNet inference can be slow on cold start
      });
      setResult(response.data);
      setState(STATES.SUCCESS);
    } catch (err) {
      if (err.response?.status === 422) {
        // Leaf guard rejection — this is an expected flow, not an error
        setResult(err.response.data.detail);
        setState(STATES.REJECTED);
      } else if (err.response?.status === 415) {
        setError(
          "Unsupported file type. Please upload a JPEG, PNG, or WebP image.",
        );
        setState(STATES.ERROR);
      } else if (err.code === "ECONNABORTED") {
        setError(
          "Request timed out. The server may be starting up — please try again.",
        );
        setState(STATES.ERROR);
      } else if (!err.response) {
        setError(
          "Cannot reach the API server. Please check that it is running.",
        );
        setState(STATES.ERROR);
      } else {
        setError(err.response?.data?.detail || "An unexpected error occurred.");
        setState(STATES.ERROR);
      }
    }
  }, []);

  const reset = useCallback(() => {
    setState(STATES.IDLE);
    setResult(null);
    setError(null);
    if (preview) {
      URL.revokeObjectURL(preview);
      setPreview(null);
    }
  }, [preview]);

  return { state, result, error, preview, predict, reset };
}
