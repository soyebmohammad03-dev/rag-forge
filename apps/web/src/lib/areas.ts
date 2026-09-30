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
    contracts: ["RAGConfiguration", "POST /api/v1/retrieval/search"],
  },
  {
    slug: "router",
    label: "Router",
    group: "Retrieval",
    icon: GitBranch,
    status: "planned",
    summary: "Watch the adaptive router analyse a query and choose retrieval strategies.",
    capabilities: [
      "Inspect query features and classification",
      "Compare the chosen policy against fixed pipelines on the same query",
      "Audit rationale for every RouterDecision",
    ],
    contracts: ["RouterDecision", "POST /api/v1/router/decide"],
  },
  {
    slug: "retrieval",
    label: "Retrieval Lab",
    group: "Retrieval",
    icon: ScanSearch,
    status: "planned",
    summary: "Inspect query transformations, candidates, scores, fusion and reranking.",
    capabilities: [
      "Side-by-side candidate lists per retriever",
      "Rank movement through fusion and reranking",
      "Score distributions and overlap between strategies",
    ],
    contracts: ["RetrievalResult", "RetrievalStrategy"],
  },
  {
    slug: "evidence",
    label: "Evidence",
    group: "Retrieval",
    icon: Microscope,
    status: "planned",
    summary: "Trace every generated claim to the evidence that supports or contradicts it.",
    capabilities: [
      "Claim-level support: supported, contradicted, unverified",
      "Span-level highlighting in source chunks",
      "Clear separation of retrieved, inferred and generated content",
    ],
    contracts: ["Claim", "Evidence", "ContentOrigin"],
  },
  {
    slug: "arena",
    label: "Arena",
    group: "Evaluate",
    icon: Swords,
    status: "planned",
    summary: "Run configurations head-to-head on the same dataset and query set.",
    capabilities: [
      "Naive, sparse, dense, hybrid, reranked, graph and routed RAG",
      "Paired comparisons with significance testing",
      "Per-query win/loss breakdowns",
    ],
    contracts: ["Experiment", "POST /api/v1/evaluation/retrieval"],
  },
  {
    slug: "experiments",
    label: "Experiments",
    group: "Evaluate",
    icon: FlaskConical,
    status: "planned",
    summary: "Define hypotheses, configurations and datasets; execute and reproduce runs.",
    capabilities: ["Experiment definitions with hypotheses", "Ablation grids", "Run queueing"],
    contracts: ["Experiment", "ExperimentRun", "GET /api/v1/experiments"],
  },
  {
    slug: "results",
    label: "Results",
    group: "Evaluate",
    icon: TrendingUp,
    status: "planned",
    summary: "Metrics, statistical comparisons, ablations and visualisations.",
    capabilities: [
      "Recall@K, Precision@K, MRR, nDCG, faithfulness, latency, cost",
      "Confidence intervals, never bare point estimates",
    ],
    contracts: ["Metric", "EvaluationResult"],
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
    status: "planned",
    summary: "Datasets, documents, versions and indexing jobs.",
    capabilities: ["Versioned ingestion", "Parsing and chunking previews", "Index job status"],
    contracts: ["Corpus", "Document", "DocumentVersion", "GET /api/v1/corpora"],
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
