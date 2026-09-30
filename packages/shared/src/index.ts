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
