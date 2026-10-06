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

/** A fixed interval, or a function of the latest data (return undefined to stop polling). */
export type Poll<T> = number | ((data: T | undefined) => number | undefined);

/**
 * Fetch from the typed API, optionally polling. `fetcher` and a function `poll` must be stable
 * (module-level or useCallback); changing either refetches. Polls chain with setTimeout, so a
 * slow request never overlaps the next one.
 * ponytail: no cache or dedupe; add a query library when several screens share data.
 */
export function useApi<T>(fetcher: Fetcher<T>, poll?: Poll<T>): ApiState<T> {
  const [state, setState] = useState<Omit<ApiState<T>, "reload">>({ loading: true });
  const [tick, setTick] = useState(0);
  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = async () => {
      let data: T | undefined;
      try {
        const res = await fetcher();
        if (!alive) return;
        data = res.data;
        setState(
          res.error === undefined
            ? { data, loading: false }
            : { loading: false, error: `API responded ${res.response.status}` },
        );
      } catch {
        if (!alive) return;
        setState({ loading: false, error: `API unreachable at ${API_URL}` });
      }
      const delay = typeof poll === "function" ? poll(data) : poll;
      if (alive && delay) timer = setTimeout(load, delay);
    };
    load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [fetcher, poll, tick]);

  return { ...state, reload };
}

export const fetchHealth = () => api.GET("/api/v1/health");
export const fetchEnvironment = () => api.GET("/api/v1/provenance/environment");
export const fetchExperiments = () => api.GET("/api/v1/experiments");
export const fetchCorpora = () => api.GET("/api/v1/corpora");
export const fetchFormats = () => api.GET("/api/v1/ingestion/formats");
export const fetchRagComponents = () => api.GET("/api/v1/rag/components");
