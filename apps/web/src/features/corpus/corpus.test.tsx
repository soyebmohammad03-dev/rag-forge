import type { Chunk, IngestionRecord } from "@rag-forge/shared";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { formatBytes, timeAgo } from "@/lib/format";
import { ChunkList, overlaps } from "./chunk-preview";
import { CreateCorpusDialog } from "./create-corpus-dialog";
import { Uploader } from "./uploader";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));
const upload = vi.hoisted(() => vi.fn());
vi.mock("@/lib/upload", () => ({ uploadDocuments: upload }));

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

const chunk = (ordinal: number, start: number, end: number, text: string): Chunk => ({
  id: `chk_${ordinal}`,
  document_version_id: "dv",
  chunking_hash: "h",
  ordinal,
  text,
  char_start: start,
  char_end: end,
  metadata: { words: text.split(" ").length },
});

describe("chunk preview", () => {
  it("computes overlap with the previous chunk and marks it", () => {
    const chunks = [chunk(0, 0, 20, "alpha beta gamma del"), chunk(1, 14, 30, "gamma delta epsi")];
    expect(overlaps(chunks)).toEqual([0, 6]);
    render(<ChunkList chunks={chunks} textChars={30} />);
    expect(screen.getByTitle("Repeated from the previous chunk")).toHaveTextContent("gamma");
    expect(screen.getByText("↩ 6 overlap")).toBeInTheDocument();
  });
});

describe("Uploader", () => {
  it("uploads dropped files and shows per-file outcomes from the API", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json([{ media_type: "text/plain", extensions: [".txt"], parser: "plaintext@1" }])));
    const record = {
      id: "ing_1",
      status: "completed",
      version_before: 0,
      version_after: 1,
      files: [
        { filename: "a.txt", outcome: "added", chunk_count: 3, content_sha256: "ab".repeat(32), warnings: [] },
        { filename: "b.txt", outcome: "duplicate", duplicate_of: "doc_1", duplicate_of_filename: "a.txt", warnings: [] },
        { filename: "c.exe", outcome: "rejected", error: "unsupported file type '.exe'", warnings: [] },
      ],
    } as unknown as IngestionRecord;
    upload.mockResolvedValue(record);
    const onIngested = vi.fn();
    const { container } = render(<Uploader corpusId="cor_1" onIngested={onIngested} />);
    const files = [new File(["x"], "a.txt"), new File(["x"], "b.txt"), new File(["MZ"], "c.exe")];
    fireEvent.drop(container.querySelector("label")!, { dataTransfer: { files } });

    expect(await screen.findByText("unsupported file type '.exe'")).toBeInTheDocument();
    expect(upload).toHaveBeenCalledWith("cor_1", files, expect.any(Function));
    expect(screen.getByText("same content as a.txt")).toBeInTheDocument();
    expect(screen.getByText("3 chunks")).toBeInTheDocument();
    expect(screen.getByText("v1")).toBeInTheDocument();
    expect(onIngested).toHaveBeenCalledOnce();
  });

  it("surfaces upload failures instead of pretending success", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json([])));
    upload.mockRejectedValue(new Error("API unreachable at http://localhost:8000"));
    const onIngested = vi.fn();
    const { container } = render(<Uploader corpusId="c" onIngested={onIngested} />);
    fireEvent.drop(container.querySelector("label")!, { dataTransfer: { files: [new File(["x"], "a.txt")] } });
    expect(await screen.findByRole("alert")).toHaveTextContent("Upload failed: API unreachable");
    expect(onIngested).not.toHaveBeenCalled();
  });
});

describe("CreateCorpusDialog", () => {
  it("blocks invalid chunking and reports a name conflict from the API", async () => {
    const fetchMock = vi.fn(async () => Response.json({ detail: "exists" }, { status: 409 }));
    vi.stubGlobal("fetch", fetchMock);
    render(<CreateCorpusDialog open onClose={() => {}} />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Name"), "papers");
    const overlap = screen.getByLabelText("Overlap");
    await user.clear(overlap);
    await user.type(overlap, "1000");
    expect(screen.getByText("Must be ≥ 0 and smaller than chunk size")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create corpus" })).toBeDisabled();

    await user.clear(overlap);
    await user.type(overlap, "100");
    await user.click(screen.getByRole("button", { name: "Create corpus" }));
    await waitFor(() => expect(screen.getByText("A corpus with this name already exists.")).toBeInTheDocument());
    const body = await (fetchMock.mock.calls[0] as unknown as [Request])[0].json();
    expect(body).toEqual({ name: "papers", description: "", chunking: { strategy: "recursive", chunk_size: 1000, chunk_overlap: 100 } });
    expect(push).not.toHaveBeenCalled();
  });
});

describe("format", () => {
  it("formats bytes and relative time", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(timeAgo("2026-01-01T00:00:00Z", Date.parse("2026-01-01T00:05:00Z"))).toBe("5 minutes ago");
  });
});
