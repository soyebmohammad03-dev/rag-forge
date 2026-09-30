import type { components, paths } from "./api";

export type { components, paths };

type Schemas = components["schemas"];

export type HealthStatus = Schemas["HealthStatus"];
export type ComponentHealth = Schemas["ComponentHealth"];
export type ComponentState = Schemas["ComponentState"];
export type Corpus = Schemas["Corpus"];
export type Experiment = Schemas["Experiment"];
export type ExperimentRun = Schemas["ExperimentRun"];
export type ExperimentStatus = Schemas["ExperimentStatus"];
export type Metric = Schemas["Metric"];
export type ContentOrigin = Schemas["ContentOrigin"];
export type RetrievalStrategy = Schemas["RetrievalStrategy"];
export type EnvironmentSnapshot = Schemas["EnvironmentSnapshot"];
export type NotImplementedDetail = Schemas["NotImplementedDetail"];
export type ChunkingConfig = Schemas["ChunkingConfig-Output"];
export type ChunkingConfigInput = Schemas["ChunkingConfig-Input"];
export type ChunkingStrategy = Schemas["ChunkingStrategy"];
export type CorpusSummary = Schemas["CorpusSummary"];
export type CorpusStats = Schemas["CorpusStats"];
export type CorpusVersion = Schemas["CorpusVersion"];
export type Document = Schemas["Document"];
export type DocumentSummary = Schemas["DocumentSummary"];
export type DocumentDetail = Schemas["DocumentDetail"];
export type DocumentVersion = Schemas["DocumentVersion"];
export type Chunk = Schemas["Chunk"];
export type ChunkPage = Schemas["ChunkPage"];
export type FileOutcome = Schemas["FileOutcome"];
export type IngestionFileResult = Schemas["IngestionFileResult"];
export type IngestionRecord = Schemas["IngestionRecord"];
export type IngestionStatus = Schemas["IngestionStatus"];
export type SupportedFormat = Schemas["SupportedFormat"];
export type VersionChange = Schemas["VersionChange"];
