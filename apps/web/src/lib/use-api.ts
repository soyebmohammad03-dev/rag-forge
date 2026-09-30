"use client";

import { useCallback, useEffect, useState } from "react";
import { API_URL, api } from "./api";

export interface ApiState<T> {
  data?: T;
  error?: string;
  loading: boolean;
  reload: () => void;
}

type Fetcher<T> = () => Promise<{ data?: T; error?: unknown; response: Response }>;

/**
 * Fetch from the typed API, optionally polling. `fetcher` must be stable (module-level or
 * useCallback); changing it refetches.
 * ponytail: no cache or dedupe; add a query library when several screens share data.
 */
export function useApi<T>(fetcher: Fetcher<T>, pollMs?: number): ApiState<T> {
  const [state, setState] = useState<Omit<ApiState<T>, "reload">>({ loading: true });
  const [tick, setTick] = useState(0);
  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const { data, error, response } = await fetcher();
        if (!alive) return;
        setState(
          error === undefined
            ? { data, loading: false }
            : { loading: false, error: `API responded ${response.status}` },
        );
      } catch {
        if (alive) setState({ loading: false, error: `API unreachable at ${API_URL}` });
      }
    };
    load();
    const id = pollMs ? setInterval(load, pollMs) : undefined;
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [fetcher, pollMs, tick]);

  return { ...state, reload };
}

export const fetchHealth = () => api.GET("/api/v1/health");
export const fetchEnvironment = () => api.GET("/api/v1/provenance/environment");
export const fetchExperiments = () => api.GET("/api/v1/experiments");
export const fetchCorpora = () => api.GET("/api/v1/corpora");
export const fetchFormats = () => api.GET("/api/v1/ingestion/formats");
