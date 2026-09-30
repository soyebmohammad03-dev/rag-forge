import type { IngestionRecord } from "@rag-forge/shared";
import { API_URL } from "./api";

export type UploadPhase = { phase: "uploading"; fraction: number } | { phase: "processing" };

/**
 * POST /api/v1/corpora/{id}/documents with upload progress. fetch() cannot report request
 * progress, so this one call uses XHR; the response type still comes from the contracts.
 */
export function uploadDocuments(
  corpusId: string,
  files: File[],
  onPhase: (p: UploadPhase) => void,
): Promise<IngestionRecord> {
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/api/v1/corpora/${encodeURIComponent(corpusId)}/documents`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => e.lengthComputable && onPhase({ phase: "uploading", fraction: e.loaded / e.total });
    xhr.upload.onload = () => onPhase({ phase: "processing" });
    xhr.onload = () =>
      xhr.status === 200
        ? resolve(xhr.response as IngestionRecord)
        : reject(new Error(xhr.response?.detail ?? `API responded ${xhr.status}`));
    xhr.onerror = () => reject(new Error(`API unreachable at ${API_URL}`));
    xhr.send(form);
  });
}
