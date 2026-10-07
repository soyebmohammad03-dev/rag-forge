import {
  Activity,
  Anvil,
  Database,
  FlaskConical,
  GitBranch,
  History,
  LayoutDashboard,
  type LucideIcon,
  Microscope,
  Network,
  ScanSearch,
  Swords,
  TrendingUp,
} from "lucide-react";

export type AreaStatus = "live" | "planned";

export interface Area {
  slug: string; // "" = overview
  label: string;
  group: string;
  icon: LucideIcon;
  status: AreaStatus;
  summary: string;
  /** What this area will let a researcher do. Shown on planned areas. */
  capabilities?: string[];
  /** Existing API contracts the area builds on. */
  contracts?: string[];
}

export const AREAS: Area[] = [
  {
    slug: "",
    label: "Overview",
    group: "Workspace",
    icon: LayoutDashboard,
    status: "live",
    summary: "System state, recent experiments and retrieval activity at a glance.",
  },
  {
    slug: "forge",
    label: "Forge",
    group: "Workspace",
    icon: Anvil,
    status: "planned",
    summary: "Compose a RAG configuration stage by stage and run it against a corpus.",
    capabilities: [
      "Pick chunker, embedding model, retrievers, reranker and generator",
      "See the configuration hash that identifies every run it produces",
      "Run single queries with full retrieval traces",
    ],
    contracts: ["ConfigurationSnapshot", "POST /api/v1/retrieval/search"],
  },
  {
    slug: "router",
    label: "Router",
    group: "Retrieval",
    icon: GitBranch,
    status: "live",
    summary: "Analyse a query, inspect the router's rule trace and compare its choice with fixed strategies.",
  },
  {
    slug: "retrieval",
    label: "Retrieval Lab",
    group: "Retrieval",
    icon: ScanSearch,
    status: "live",
    summary: "Query a corpus version with BM25, dense, hybrid, reranking or the router; inspect evidence and provenance.",
  },
  {
    slug: "evidence",
    label: "Evidence Lab",
    group: "Retrieval",
    icon: Microscope,
    status: "live",
    summary: "Generate an answer from selected evidence only and trace every claim to the passages that support it.",
  },
  {
    slug: "arena",
    label: "Arena",
    group: "Evaluate",
    icon: Swords,
    status: "live",
    summary: "Benchmark datasets, controlled runs of several configurations and paired statistics, with provenance.",
  },
  {
    slug: "experiments",
    label: "Experiments",
    group: "Evaluate",
    icon: FlaskConical,
    status: "live",
    summary: "Build an experiment: dataset, configuration matrix, ablations, metrics and limits; then run it.",
  },
  {
    slug: "results",
    label: "Results",
    group: "Evaluate",
    icon: TrendingUp,
    status: "live",
    summary: "Leaderboards, statistical comparisons, per-case results, failures and latency of recorded runs.",
  },
  {
    slug: "replay",
    label: "Replay",
    group: "Evaluate",
    icon: History,
    status: "planned",
    summary: "Reconstruct a past run from its recorded configuration and provenance.",
    capabilities: ["Re-execute with pinned versions", "Diff a replay against the original"],
    contracts: ["ProvenanceRecord", "GET /api/v1/provenance/environment"],
  },
  {
    slug: "corpus",
    label: "Corpus",
    group: "Data",
    icon: Database,
    status: "live",
    summary: "Versioned corpora: ingestion, documents, version history and chunk inspection.",
  },
  {
    slug: "knowledge",
    label: "Knowledge",
    group: "Data",
    icon: Network,
    status: "planned",
    summary: "Explore documents, chunks, entities, claims and their relationships.",
    capabilities: ["Entity and relation graph", "Chunk neighbourhoods", "Graph-retrieval paths"],
    contracts: ["Chunk", "Claim"],
  },
  {
    slug: "system",
    label: "System",
    group: "Operate",
    icon: Activity,
    status: "live",
    summary: "Component health and the environment recorded into every run's provenance.",
  },
];

export const areaHref = (a: Area) => `/${a.slug}`;
export const findArea = (slug: string) => AREAS.find((a) => a.slug === slug);
export const AREA_GROUPS = [...new Set(AREAS.map((a) => a.group))];
