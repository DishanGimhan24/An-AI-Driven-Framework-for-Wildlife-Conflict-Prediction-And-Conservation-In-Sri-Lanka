// src/hooks/usePrediction.js
import { useState, useCallback } from "react";
import { api } from "../services/api";

export function usePrediction() {
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const predict = useCallback(async (params) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.predict(params);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, predict, reset };
}

export function useBatchPredict() {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const fetch = useCallback(async (month) => {
    setLoading(true);
    setError(null);
    try {
      const d = await api.batchPredict(month);
      setData(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return { data, loading, error, fetch };
}
