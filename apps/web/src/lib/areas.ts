import {
  Activity,
  Database,
  FlaskConical,
  GitBranch,
  History,
  LayoutDashboard,
  type LucideIcon,
  Microscope,
  ScanSearch,
  Swords,
  TrendingUp,
} from "lucide-react";

export interface Area {
  slug: string; // "" = overview
  label: string;
  group: string;
  icon: LucideIcon;
  summary: string;
}

/** Every area of the app. Each is built and talks to the real API. */
export const AREAS: Area[] = [
  {
    slug: "",
    label: "Overview",
    group: "Workspace",
    icon: LayoutDashboard,
    summary: "The implemented pipeline, models, corpora, experiments and reproducibility state.",
  },
  {
    slug: "corpus",
    label: "Corpus",
    group: "Data",
    icon: Database,
    summary: "Versioned corpora: ingestion, documents, version history and chunk inspection.",
  },
  {
    slug: "retrieval",
    label: "Retrieval Lab",
    group: "Retrieval",
    icon: ScanSearch,
    summary: "Query a corpus version with BM25, dense, hybrid, reranking or the router; inspect evidence and provenance.",
  },
  {
    slug: "router",
    label: "Router",
    group: "Retrieval",
    icon: GitBranch,
    summary: "Analyse a query, inspect the router's rule trace and compare its choice with fixed strategies.",
  },
  {
    slug: "evidence",
    label: "Evidence Lab",
    group: "Retrieval",
    icon: Microscope,
    summary: "Generate an answer from selected evidence only and trace every claim to the passages that support it.",
  },
  {
    slug: "arena",
    label: "Arena",
    group: "Evaluate",
    icon: Swords,
    summary: "Benchmark datasets, controlled runs of several configurations and paired statistics, with provenance.",
  },
  {
    slug: "experiments",
    label: "Experiments",
    group: "Evaluate",
    icon: FlaskConical,
    summary: "Build an experiment: dataset, configuration matrix, ablations, metrics and limits; then run it.",
  },
  {
    slug: "results",
    label: "Results",
    group: "Evaluate",
    icon: TrendingUp,
    summary: "Leaderboards, statistical comparisons, per-case results, failures and latency of recorded runs.",
  },
  {
    slug: "replay",
    label: "Replay",
    group: "Evaluate",
    icon: History,
    summary: "Inspect a recorded run as an audit trail, replay it stage by stage and export its report and manifest.",
  },
  {
    slug: "system",
    label: "System",
    group: "Operate",
    icon: Activity,
    summary: "Component health and the environment recorded into every run's provenance.",
  },
];

export const areaHref = (a: Area) => `/${a.slug}`;
export const findArea = (slug: string) => AREAS.find((a) => a.slug === slug);
export const AREA_GROUPS = [...new Set(AREAS.map((a) => a.group))];
