/**
 * Cognitive Memory System - Remote Adapter
 *
 * Talks to a `cm-daemon` over a Unix domain socket using length-delimited
 * JSON. Mirror of the Rust protocol crate types — see
 * `../../cognitive-memory-daemon/PROTOCOL.md` for the wire spec.
 *
 * Full feature parity with the SDK's `MemoryAdapter` interface plus the
 * paper-faithful `createBatch` for co-creation associations.
 */

import * as net from "node:net";
import * as os from "node:os";
import * as path from "node:path";
import type { Memory, ScoredMemory } from "../core/types";
import { MemoryAdapter, type MemoryFilters } from "./base";

const IPC_PROTOCOL_VERSION = 1;

// =============================================================================
// Wire types (mirror crates/protocol/src/lib.rs)
// =============================================================================

type IpcMessage = {
  id: number;
  payload:
    | { kind: "Request"; body: RemoteRequest }
    | { kind: "Response"; body: RemoteResponse }
    | { kind: "Event"; body: unknown };
};

type RemoteRequest =
  | { bucket: "Diagnostics"; op: "Status" }
  | { bucket: "Diagnostics"; op: "Counts"; user_id: string }
  | { bucket: "Diagnostics"; op: "MintBridgeToken"; user_id: string; scope: "read" | "write" | "admin"; ttl_seconds: number }
  | { bucket: "Memory"; op: "Store"; user_id: string; content: string; category: string; memory_type: string; metadata: string }
  | { bucket: "Memory"; op: "StoreBatch"; user_id: string; memories: BatchEntry[]; initial_link_weight: number }
  | { bucket: "Memory"; op: "Search"; user_id: string; query: string; limit: number; deep_recall: boolean; hybrid: boolean }
  | { bucket: "Memory"; op: "SearchLexical"; user_id: string; query: string; limit: number }
  | { bucket: "Memory"; op: "VectorSearch"; user_id: string; embedding: number[]; embedding_provider: string; embedding_model: string; limit: number; deep_recall: boolean }
  | { bucket: "Memory"; op: "Get"; user_id: string; id: string }
  | { bucket: "Memory"; op: "GetMany"; user_id: string; ids: string[] }
  | { bucket: "Memory"; op: "List"; user_id: string; categories?: string[]; memory_types?: string[]; min_retention_floor?: number; min_importance?: number; created_after?: number; created_before?: number; limit?: number; offset?: number; include_superseded: boolean; include_cold: boolean; include_stubs: boolean }
  | { bucket: "Memory"; op: "Update"; user_id: string; id: string; content?: string; category?: string; memory_type?: string; metadata?: string; retention_floor?: number; importance?: number; stability?: number; valid_until?: number }
  | { bucket: "Memory"; op: "Delete"; user_id: string; id: string }
  | { bucket: "Memory"; op: "DeleteMany"; user_id: string; ids: string[] }
  | { bucket: "Memory"; op: "Link"; user_id: string; source_id: string; target_id: string; strength: number; bidirectional: boolean; kind: string }
  | { bucket: "Memory"; op: "Unlink"; user_id: string; source_id: string; target_id: string; bidirectional: boolean }
  | { bucket: "Memory"; op: "GetLinked"; user_id: string; source_id: string; min_strength: number }
  | { bucket: "Memory"; op: "GetLinkedMany"; user_id: string; source_ids: string[]; min_strength: number }
  | { bucket: "Memory"; op: "BatchUpdate"; user_id: string; updates: { id: string; retention_floor: number }[] }
  | { bucket: "Lifecycle"; op: "Tick"; synchronous: boolean }
  | { bucket: "Lifecycle"; op: "FindFading"; user_id: string; max_retention: number; limit: number }
  | { bucket: "Lifecycle"; op: "FindStable"; user_id: string; min_stability: number; min_access_count: number; limit: number }
  | { bucket: "Lifecycle"; op: "MarkSuperseded"; user_id: string; ids: string[]; summary_id: string }
  | { bucket: "Lifecycle"; op: "MigrateToCold"; user_id: string; id: string; cold_since: number }
  | { bucket: "Lifecycle"; op: "MigrateToHot"; user_id: string; id: string }
  | { bucket: "Lifecycle"; op: "ConvertToStub"; user_id: string; id: string; stub_content: string }
  | { bucket: "Lifecycle"; op: "UpdateRetention"; user_id: string; id: string; retention_floor: number }
  | { bucket: "Lifecycle"; op: "Clear"; user_id: string; confirm: boolean };

type BatchEntry = {
  content: string;
  category: string;
  memory_type: string;
  metadata: string;
};

type RemoteMemory = {
  id: string;
  user_id: string;
  content: string;
  category: string;
  memory_type: string;
  created_at: number;
  last_accessed_at: number;
  valid_from: number | null;
  valid_until: number | null;
  retention_floor: number;
  retrieval_count: number;
  importance: number;
  stability: number;
  is_cold: boolean;
  cold_since: number | null;
  is_superseded: boolean;
  superseded_by: string | null;
  is_stub: boolean;
  stub_content: string | null;
  metadata: string;
};

type RemoteResponse = {
  ok: boolean;
  data?:
    | { kind: "Status"; daemon_version: string; uptime_seconds: number; memory_count: number }
    | { kind: "Counts"; hot: number; cold: number; stub: number; total: number }
    | { kind: "MemoryStored"; id: string }
    | { kind: "MemoryStoredBatch"; ids: string[]; associations_created: number }
    | { kind: "MemorySearchResults"; results: SearchHit[] }
    | { kind: "Memory"; [k: string]: unknown }
    | { kind: "Memories"; memories: RemoteMemory[] }
    | { kind: "Affected"; affected: number }
    | { kind: "LinkedMemories"; memories: { memory: RemoteMemory; link_strength: number }[] }
    | { kind: "LinkStrength"; strength: number }
    | { kind: "LexicalIds"; ids: string[] }
    | { kind: "Tick"; completed: boolean; memories_decayed: number }
    | { kind: "BridgeToken"; token: string; expires_at_unix: number };
  error?: { kind: string; message: string; retriable: boolean };
};

type SearchHit = {
  memory_id: string;
  content: string;
  category: string;
  memory_type: string;
  score: number;
};

// =============================================================================
// Errors
// =============================================================================

export class RemoteAdapterError extends Error {
  // Parameter properties aren't supported by Node's --experimental-strip-types
  // — spell the field out so the file is runnable directly without tsc.
  readonly cause?: unknown;

  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = "RemoteAdapterError";
    this.cause = cause;
  }
}

// =============================================================================
// Length-prefixed framing
// =============================================================================

function frame(json: object): Buffer {
  const body = Buffer.from(JSON.stringify(json), "utf8");
  const header = Buffer.alloc(4);
  header.writeUInt32BE(body.length, 0);
  return Buffer.concat([header, body]);
}

class FrameReader {
  private buffer = Buffer.alloc(0);

  feed(chunk: Buffer): unknown[] {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    const out: unknown[] = [];
    while (this.buffer.length >= 4) {
      const len = this.buffer.readUInt32BE(0);
      if (this.buffer.length < 4 + len) break;
      const body = this.buffer.subarray(4, 4 + len);
      this.buffer = this.buffer.subarray(4 + len);
      out.push(JSON.parse(body.toString("utf8")));
    }
    return out;
  }
}

// =============================================================================
// RemoteAdapter — full SDK MemoryAdapter parity
// =============================================================================

export interface RemoteAdapterOptions {
  socketPath?: string;
  userId: string;
  clientLabel?: string;
}

export class RemoteAdapter extends MemoryAdapter {
  private socket: net.Socket | null = null;
  private reader = new FrameReader();
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (r: RemoteResponse) => void; reject: (e: unknown) => void }
  >();
  private connected = false;
  private connectPromise: Promise<void> | null = null;
  private readonly socketPath: string;
  private readonly userId: string;
  private readonly clientLabel: string;

  constructor(options: RemoteAdapterOptions) {
    super();
    this.socketPath = options.socketPath ?? defaultSocketPath();
    this.userId = options.userId;
    this.clientLabel = options.clientLabel ?? "cognitive-memory-sdk-ts";
  }

  async connect(): Promise<void> {
    if (this.connected) return;
    if (this.connectPromise) return this.connectPromise;
    this.connectPromise = (async () => {
      const socket = net.createConnection({ path: this.socketPath });
      await new Promise<void>((resolve, reject) => {
        socket.once("connect", () => resolve());
        socket.once("error", reject);
      });
      socket.write(frame({
        kind: "Hello",
        client: this.clientLabel,
        protocol_version: IPC_PROTOCOL_VERSION,
        user_id: this.userId,
      }));
      const welcome = await readOne(socket, this.reader);
      const w = welcome as { protocol_version?: number };
      if (w.protocol_version !== IPC_PROTOCOL_VERSION) {
        socket.destroy();
        throw new RemoteAdapterError(`daemon protocol mismatch (expected v${IPC_PROTOCOL_VERSION}, got ${w.protocol_version})`);
      }
      socket.on("data", (chunk) => this.dispatch(chunk));
      socket.on("error", (e) => this.failPending(e));
      socket.on("close", () => this.failPending(new RemoteAdapterError("socket closed")));
      this.socket = socket;
      this.connected = true;
    })();
    try { await this.connectPromise; } finally { this.connectPromise = null; }
  }

  async close(): Promise<void> {
    this.socket?.destroy();
    this.socket = null;
    this.connected = false;
    this.failPending(new RemoteAdapterError("adapter closed"));
  }

  private dispatch(chunk: Buffer): void {
    for (const msg of this.reader.feed(chunk)) {
      const m = msg as IpcMessage;
      const pending = this.pending.get(m.id);
      if (!pending) continue;
      this.pending.delete(m.id);
      if (m.payload.kind === "Response") pending.resolve(m.payload.body);
      else pending.reject(new RemoteAdapterError(`unexpected payload: ${m.payload.kind}`));
    }
  }

  private failPending(err: unknown): void {
    for (const { reject } of this.pending.values()) reject(err);
    this.pending.clear();
  }

  private async sendRequest(req: RemoteRequest): Promise<RemoteResponse> {
    await this.connect();
    if (!this.socket) throw new RemoteAdapterError("not connected");
    const id = this.nextId++;
    return new Promise<RemoteResponse>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket!.write(frame({ id, payload: { kind: "Request", body: req } } as IpcMessage), (err) => {
        if (err) { this.pending.delete(id); reject(err); }
      });
    });
  }

  private unwrap(resp: RemoteResponse, expectedKind?: string): RemoteResponse["data"] {
    if (!resp.ok) {
      throw new RemoteAdapterError(`daemon error (${resp.error?.kind ?? "?"}): ${resp.error?.message ?? "?"}`);
    }
    const data = resp.data;
    if (expectedKind && data?.kind !== expectedKind) {
      throw new RemoteAdapterError(`expected ${expectedKind}, got ${data?.kind ?? "?"}`);
    }
    return data;
  }

  // -- CRUD --

  async createMemory(memory: Omit<Memory, "id" | "createdAt" | "updatedAt">): Promise<string> {
    const m = memory as { memoryType?: string; metadata?: unknown };
    const data = this.unwrap(
      await this.sendRequest({
        bucket: "Memory", op: "Store", user_id: this.userId,
        content: memory.content, category: memory.category,
        memory_type: m.memoryType ?? "fact",
        metadata: JSON.stringify(m.metadata ?? {}),
      }),
      "MemoryStored",
    );
    return (data as { id: string }).id;
  }

  async getMemory(id: string): Promise<Memory | null> {
    try {
      const data = this.unwrap(
        await this.sendRequest({ bucket: "Memory", op: "Get", user_id: this.userId, id }),
        "Memory",
      );
      return remoteMemoryToMemory(data as unknown as RemoteMemory);
    } catch (e) {
      if (e instanceof RemoteAdapterError && e.message.includes("NotFound")) return null;
      throw e;
    }
  }

  async getMemories(ids: string[]): Promise<Memory[]> {
    if (ids.length === 0) return [];
    const data = this.unwrap(
      await this.sendRequest({ bucket: "Memory", op: "GetMany", user_id: this.userId, ids }),
      "Memories",
    );
    return (data as { memories: RemoteMemory[] }).memories.map(remoteMemoryToMemory);
  }

  async queryMemories(filters: MemoryFilters): Promise<Memory[]> {
    const data = this.unwrap(
      await this.sendRequest({
        bucket: "Memory", op: "List", user_id: this.userId,
        categories: filters.categories,
        min_retention_floor: filters.minRetention,
        min_importance: filters.minImportance,
        created_after: filters.createdAfter,
        created_before: filters.createdBefore,
        limit: filters.limit, offset: filters.offset,
        include_superseded: filters.includeSuperseded ?? false,
        include_cold: filters.includeCold ?? false,
        include_stubs: filters.includeStubs ?? false,
      }),
      "Memories",
    );
    return (data as { memories: RemoteMemory[] }).memories.map(remoteMemoryToMemory);
  }

  async updateMemory(id: string, updates: Partial<Memory>): Promise<void> {
    const u = updates as Partial<Memory> & { memoryType?: string; metadata?: unknown };
    this.unwrap(
      await this.sendRequest({
        bucket: "Memory", op: "Update", user_id: this.userId, id,
        content: u.content, category: u.category, memory_type: u.memoryType,
        metadata: u.metadata !== undefined ? JSON.stringify(u.metadata) : undefined,
        importance: u.importance, stability: u.stability,
      }),
      "Affected",
    );
  }

  async deleteMemory(id: string): Promise<void> {
    this.unwrap(
      await this.sendRequest({ bucket: "Memory", op: "Delete", user_id: this.userId, id }),
      "Affected",
    );
  }

  async deleteMemories(ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    this.unwrap(
      await this.sendRequest({ bucket: "Memory", op: "DeleteMany", user_id: this.userId, ids }),
      "Affected",
    );
  }

  // -- Search --

  async vectorSearch(embedding: number[], _filters?: MemoryFilters): Promise<ScoredMemory[]> {
    const data = this.unwrap(
      await this.sendRequest({
        bucket: "Memory", op: "VectorSearch", user_id: this.userId,
        embedding, embedding_provider: "local", embedding_model: "bge-small-en-v1.5",
        limit: 10, deep_recall: false,
      }),
      "MemorySearchResults",
    );
    return (data as { results: SearchHit[] }).results.map((hit) =>
      hitToScoredMemory(hit),
    );
  }

  override async searchLexical(query: string, _filters?: MemoryFilters): Promise<ScoredMemory[]> {
    const data = this.unwrap(
      await this.sendRequest({ bucket: "Memory", op: "SearchLexical", user_id: this.userId, query, limit: 10 }),
      "LexicalIds",
    );
    const ids = (data as { ids: string[] }).ids;
    const memories = await this.getMemories(ids);
    return memories.map((m, i) => {
      const score = ids.length - i;
      return { ...m, relevanceScore: score, finalScore: score };
    });
  }

  // -- Retention --

  async updateRetentionScores(updates: Map<string, number>): Promise<void> {
    if (updates.size === 0) return;
    this.unwrap(
      await this.sendRequest({
        bucket: "Memory", op: "BatchUpdate", user_id: this.userId,
        updates: Array.from(updates.entries()).map(([id, retention_floor]) => ({ id, retention_floor })),
      }),
      "Affected",
    );
  }

  // -- Links --

  async createOrStrengthenLink(sourceId: string, targetId: string, strength: number): Promise<void> {
    this.unwrap(
      await this.sendRequest({
        bucket: "Memory", op: "Link", user_id: this.userId,
        source_id: sourceId, target_id: targetId, strength,
        bidirectional: true, kind: "explicit",
      }),
      "LinkStrength",
    );
  }

  async getLinkedMemories(memoryId: string, minStrength?: number): Promise<Array<Memory & { linkStrength: number }>> {
    const data = this.unwrap(
      await this.sendRequest({
        bucket: "Memory", op: "GetLinked", user_id: this.userId,
        source_id: memoryId, min_strength: minStrength ?? 0.0,
      }),
      "LinkedMemories",
    );
    return (data as { memories: { memory: RemoteMemory; link_strength: number }[] }).memories.map((lm) => ({
      ...remoteMemoryToMemory(lm.memory),
      linkStrength: lm.link_strength,
    }));
  }

  async getLinkedMemoriesMultiple(memoryIds: string[], minStrength?: number): Promise<Array<Memory & { linkStrength: number }>> {
    const data = this.unwrap(
      await this.sendRequest({
        bucket: "Memory", op: "GetLinkedMany", user_id: this.userId,
        source_ids: memoryIds, min_strength: minStrength ?? 0.0,
      }),
      "LinkedMemories",
    );
    return (data as { memories: { memory: RemoteMemory; link_strength: number }[] }).memories.map((lm) => ({
      ...remoteMemoryToMemory(lm.memory),
      linkStrength: lm.link_strength,
    }));
  }

  async deleteLink(sourceId: string, targetId: string): Promise<void> {
    this.unwrap(
      await this.sendRequest({
        bucket: "Memory", op: "Unlink", user_id: this.userId,
        source_id: sourceId, target_id: targetId, bidirectional: true,
      }),
      "Affected",
    );
  }

  // -- Consolidation helpers --

  async findFadingMemories(userId: string, maxRetention: number): Promise<Memory[]> {
    if (userId !== this.userId) throw new RemoteAdapterError(`adapter scoped to ${this.userId}`);
    const data = this.unwrap(
      await this.sendRequest({ bucket: "Lifecycle", op: "FindFading", user_id: userId, max_retention: maxRetention, limit: 100 }),
      "Memories",
    );
    return (data as { memories: RemoteMemory[] }).memories.map(remoteMemoryToMemory);
  }

  async findStableMemories(userId: string, minStability: number, minAccessCount: number): Promise<Memory[]> {
    if (userId !== this.userId) throw new RemoteAdapterError(`adapter scoped to ${this.userId}`);
    const data = this.unwrap(
      await this.sendRequest({
        bucket: "Lifecycle", op: "FindStable", user_id: userId,
        min_stability: minStability, min_access_count: minAccessCount, limit: 100,
      }),
      "Memories",
    );
    return (data as { memories: RemoteMemory[] }).memories.map(remoteMemoryToMemory);
  }

  async markSuperseded(memoryIds: string[], summaryId: string): Promise<void> {
    if (memoryIds.length === 0) return;
    this.unwrap(
      await this.sendRequest({
        bucket: "Lifecycle", op: "MarkSuperseded", user_id: this.userId,
        ids: memoryIds, summary_id: summaryId,
      }),
      "Affected",
    );
  }

  // -- Tiered storage --

  async migrateToCold(memoryId: string, coldSince: number): Promise<void> {
    this.unwrap(
      await this.sendRequest({
        bucket: "Lifecycle", op: "MigrateToCold", user_id: this.userId,
        id: memoryId, cold_since: coldSince,
      }),
      "Affected",
    );
  }

  async migrateToHot(memoryId: string): Promise<void> {
    this.unwrap(
      await this.sendRequest({ bucket: "Lifecycle", op: "MigrateToHot", user_id: this.userId, id: memoryId }),
      "Affected",
    );
  }

  async convertToStub(memoryId: string, stubContent: string): Promise<void> {
    this.unwrap(
      await this.sendRequest({
        bucket: "Lifecycle", op: "ConvertToStub", user_id: this.userId,
        id: memoryId, stub_content: stubContent,
      }),
      "Affected",
    );
  }

  // -- Traversal --

  async allActive(): Promise<Memory[]> {
    return this.queryMemories({ includeCold: false, includeStubs: false, includeSuperseded: false });
  }

  async allHot(): Promise<Memory[]> {
    return this.queryMemories({ includeCold: false, includeStubs: false });
  }

  async allCold(): Promise<Memory[]> {
    const all = await this.queryMemories({ includeCold: true, includeStubs: false });
    return all.filter((m) => (m as { isCold?: boolean }).isCold === true);
  }

  // -- Counts --

  async hotCount(): Promise<number> { return (await this.fetchCounts()).hot; }
  async coldCount(): Promise<number> { return (await this.fetchCounts()).cold; }
  async stubCount(): Promise<number> { return (await this.fetchCounts()).stub; }
  async totalCount(): Promise<number> { return (await this.fetchCounts()).total; }

  private async fetchCounts(): Promise<{ hot: number; cold: number; stub: number; total: number }> {
    const data = this.unwrap(
      await this.sendRequest({ bucket: "Diagnostics", op: "Counts", user_id: this.userId }),
      "Counts",
    );
    return data as { hot: number; cold: number; stub: number; total: number; kind: "Counts" };
  }

  // -- Reset --

  async clear(): Promise<void> {
    this.unwrap(
      await this.sendRequest({ bucket: "Lifecycle", op: "Clear", user_id: this.userId, confirm: true }),
      "Affected",
    );
  }

  // -- Transaction (no daemon-side primitive in v1) --

  async transaction<T>(callback: (adapter: MemoryAdapter) => Promise<T>): Promise<T> {
    return callback(this);
  }

  // -- Daemon-only extras (not on MemoryAdapter base) --

  /**
   * Paper-faithful batch storage with co-creation associations
   * (paper §3.6). Memories created together get bidirectional links.
   */
  async createBatch(
    memories: Array<Omit<Memory, "id" | "createdAt" | "updatedAt">>,
    initialLinkWeight = 0.5,
  ): Promise<{ ids: string[]; associationsCreated: number }> {
    const data = this.unwrap(
      await this.sendRequest({
        bucket: "Memory", op: "StoreBatch", user_id: this.userId,
        memories: memories.map((m) => {
          const mm = m as { memoryType?: string; metadata?: unknown };
          return {
            content: m.content,
            category: m.category,
            memory_type: mm.memoryType ?? "fact",
            metadata: JSON.stringify(mm.metadata ?? {}),
          };
        }),
        initial_link_weight: initialLinkWeight,
      }),
      "MemoryStoredBatch",
    );
    const d = data as { ids: string[]; associations_created: number };
    return { ids: d.ids, associationsCreated: d.associations_created };
  }

  /** Mint a bearer token for cm-http. */
  async mintBridgeToken(scope: "read" | "write" | "admin", ttlSeconds: number): Promise<{ token: string; expiresAtUnix: number }> {
    const data = this.unwrap(
      await this.sendRequest({
        bucket: "Diagnostics", op: "MintBridgeToken",
        user_id: this.userId, scope, ttl_seconds: ttlSeconds,
      }),
      "BridgeToken",
    );
    const d = data as { token: string; expires_at_unix: number };
    return { token: d.token, expiresAtUnix: d.expires_at_unix };
  }
}

function hitToScoredMemory(hit: SearchHit): ScoredMemory {
  // The wire SearchHit only carries id/content/category/type/score —
  // we synthesise a minimal Memory shape here. Callers that need more
  // fields should follow up with `getMemory(hit.memory_id)`.
  return {
    id: hit.memory_id,
    userId: "",
    content: hit.content,
    category: hit.category as Memory["category"],
    embedding: [],
    importance: 0,
    stability: 0,
    accessCount: 0,
    lastAccessed: 0,
    retention: 0,
    createdAt: 0,
    updatedAt: 0,
    associations: {},
    sessionIds: [],
    isCold: false,
    coldSince: null,
    daysAtFloor: 0,
    isSuperseded: false,
    supersededBy: null,
    isStub: false,
    contradictedBy: null,
    semanticType: hit.memory_type as Memory["semanticType"],
    relevanceScore: hit.score,
    finalScore: hit.score,
  };
}

function remoteMemoryToMemory(rm: RemoteMemory): Memory {
  return {
    id: rm.id,
    userId: rm.user_id,
    content: rm.content,
    category: rm.category as Memory["category"],
    embedding: [],
    importance: rm.importance,
    stability: rm.stability,
    accessCount: rm.retrieval_count,
    lastAccessed: rm.last_accessed_at,
    retention: rm.retention_floor,
    createdAt: rm.created_at,
    updatedAt: rm.last_accessed_at,
    metadata: rm.metadata ? JSON.parse(rm.metadata) : undefined,
    associations: {},
    sessionIds: [],
    isCold: rm.is_cold,
    coldSince: rm.cold_since,
    daysAtFloor: 0,
    isSuperseded: rm.is_superseded,
    supersededBy: rm.superseded_by,
    isStub: rm.is_stub,
    contradictedBy: null,
    semanticType: rm.memory_type as Memory["semanticType"],
    validFrom: rm.valid_from,
    validUntil: rm.valid_until,
    ttlSeconds: null,
  };
}

function defaultSocketPath(): string {
  if (process.platform === "darwin") {
    return path.join(os.homedir(), "Library", "Application Support", "cognitive-memory", "cm.sock");
  }
  const xdg = process.env.XDG_RUNTIME_DIR;
  if (xdg) return path.join(xdg, "cognitive-memory", "cm.sock");
  return path.join(os.tmpdir(), "cognitive-memory", "cm.sock");
}

async function readOne(socket: net.Socket, reader: FrameReader): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const onData = (chunk: Buffer) => {
      const messages = reader.feed(chunk);
      if (messages.length > 0) {
        socket.off("data", onData);
        socket.off("error", onError);
        resolve(messages[0]);
      }
    };
    const onError = (err: Error) => {
      socket.off("data", onData);
      socket.off("error", onError);
      reject(err);
    };
    socket.on("data", onData);
    socket.on("error", onError);
  });
}
