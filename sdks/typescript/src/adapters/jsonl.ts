/**
 * JSONL File Adapter for Cognitive Memory
 *
 * Append-only event log + in-memory three-tier index.
 * Node-only (uses fs).
 */

import { randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  appendFile,
  mkdir,
  open,
  readdir,
  rename,
  stat,
  writeFile,
} from "node:fs/promises";
import { basename, dirname } from "node:path";
import * as readline from "node:readline";
import type { Memory, ScoredMemory } from "../core/types";
import { createDefaultMemory } from "../core/types";
import { cosineSimilarity } from "../utils/embeddings";
import { MemoryAdapter, type MemoryFilters } from "./base";

type MemoryEvent =
  | { type: "meta"; version: 1; createdAt: number }
  | { type: "memory"; memory: Memory }
  | { type: "memory_delete"; id: string; at: number }
  | {
      type: "link";
      a: string;
      b: string;
      strength: number;
      createdAt: number;
      updatedAt: number;
    }
  | { type: "link_delete"; a: string; b: string; at: number };

type LinkRow = { strength: number; createdAt: number; updatedAt: number };

function canonicalPair(a: string, b: string): [string, string] {
  return a < b ? [a, b] : [b, a];
}

function linkKey(a: string, b: string): string {
  const [x, y] = canonicalPair(a, b);
  return `${x}|${y}`;
}

export type JsonlFileAdapterOptions = {
  path: string;
  fsync?: boolean;
  rollover?: { maxLines?: number; enabled?: boolean };
  compact?: { maxLines?: number; onStart?: boolean };
  now?: () => number;
  idFactory?: () => string;
};

export class JsonlFileAdapter extends MemoryAdapter {
  private path: string;
  private fsync: boolean;
  private maxLines: number;
  private rolloverEnabled: boolean;
  private compactOnStart: boolean;
  private now: () => number;
  private idFactory: () => string;

  private loaded = false;
  private loadPromise: Promise<void> | null = null;
  private writeChain: Promise<void> = Promise.resolve();

  private lineCount = 0;
  private hot = new Map<string, Memory>();
  private cold = new Map<string, Memory>();
  private stubs = new Map<string, Memory>();
  private links = new Map<string, LinkRow>();

  constructor(options: JsonlFileAdapterOptions) {
    super();
    this.path = options.path;
    this.fsync = options.fsync ?? false;
    this.maxLines =
      options.rollover?.maxLines ?? options.compact?.maxLines ?? 200_000;
    this.rolloverEnabled = options.rollover?.enabled ?? true;
    this.compactOnStart = options.compact?.onStart ?? false;
    this.now = options.now ?? Date.now;
    this.idFactory = options.idFactory ?? randomUUID;
  }

  async ready(): Promise<void> {
    if (this.loaded) return;
    if (!this.loadPromise) this.loadPromise = this.load();
    await this.loadPromise;
  }

  async transaction<T>(
    callback: (adapter: MemoryAdapter) => Promise<T>,
  ): Promise<T> {
    await this.ready();
    return callback(this);
  }

  async createMemory(
    memory: Omit<Memory, "id" | "createdAt" | "updatedAt">,
  ): Promise<string> {
    await this.ready();
    const id = this.idFactory();
    const now = this.now();
    const m = createDefaultMemory({
      ...memory,
      id,
      createdAt: now,
      updatedAt: now,
    });
    this.hot.set(id, m);
    await this.append({ type: "memory", memory: m });
    return id;
  }

  async getMemory(id: string): Promise<Memory | null> {
    await this.ready();
    return this.hot.get(id) ?? this.cold.get(id) ?? this.stubs.get(id) ?? null;
  }

  async getMemories(ids: string[]): Promise<Memory[]> {
    await this.ready();
    return ids
      .map((id) => this.hot.get(id) ?? this.cold.get(id) ?? this.stubs.get(id))
      .filter(Boolean) as Memory[];
  }

  async queryMemories(filters: MemoryFilters): Promise<Memory[]> {
    await this.ready();
    let items = Array.from(this.hot.values());

    if (filters.includeCold) {
      items = [...items, ...Array.from(this.cold.values())];
    }
    if (filters.includeStubs) {
      items = [...items, ...Array.from(this.stubs.values())];
    }

    if (!filters.includeSuperseded)
      items = items.filter((m) => !m.isSuperseded);
    if (!filters.includeStubs) items = items.filter((m) => !m.isStub);

    if (filters.userId)
      items = items.filter((m) => m.userId === filters.userId);
    if (filters.categories)
      items = items.filter((m) => filters.categories!.includes(m.category));
    if (filters.minRetention !== undefined)
      items = items.filter((m) => m.retention >= filters.minRetention!);
    if (filters.minImportance !== undefined)
      items = items.filter((m) => m.importance >= filters.minImportance!);
    if (filters.createdAfter !== undefined)
      items = items.filter((m) => m.createdAt >= filters.createdAfter!);
    if (filters.createdBefore !== undefined)
      items = items.filter((m) => m.createdAt <= filters.createdBefore!);
    if (filters.offset) items = items.slice(filters.offset);
    if (filters.limit) items = items.slice(0, filters.limit);
    return items;
  }

  async updateMemory(id: string, updates: Partial<Memory>): Promise<void> {
    await this.ready();
    const tier = this.getTier(id);
    if (!tier) return;
    const existing = tier.get(id)!;
    const next = { ...existing, ...updates, id, createdAt: existing.createdAt };
    tier.set(id, next);
    await this.append({ type: "memory", memory: next });
  }

  async deleteMemory(id: string): Promise<void> {
    await this.ready();
    this.hot.delete(id);
    this.cold.delete(id);
    this.stubs.delete(id);
    await this.append({ type: "memory_delete", id, at: this.now() });
  }

  async deleteMemories(ids: string[]): Promise<void> {
    await this.ready();
    await this.withWriteLock(async () => {
      for (const id of ids) {
        this.hot.delete(id);
        this.cold.delete(id);
        this.stubs.delete(id);
        await this.appendUnlocked({
          type: "memory_delete",
          id,
          at: this.now(),
        });
      }
    });
  }

  async vectorSearch(
    embedding: number[],
    filters?: MemoryFilters,
  ): Promise<ScoredMemory[]> {
    await this.ready();
    let items = Array.from(this.hot.values());

    if (filters?.includeCold) {
      items = [...items, ...Array.from(this.cold.values())];
    }
    if (filters?.includeStubs) {
      items = [...items, ...Array.from(this.stubs.values())];
    }

    if (filters?.userId)
      items = items.filter((m) => m.userId === filters.userId);
    if (filters?.categories)
      items = items.filter((m) => filters.categories!.includes(m.category));
    if (filters?.minRetention !== undefined)
      items = items.filter((m) => m.retention >= filters.minRetention!);
    if (!filters?.includeSuperseded)
      items = items.filter((m) => !m.isSuperseded);
    if (!filters?.includeStubs) items = items.filter((m) => !m.isStub);

    return items
      .map((m) => {
        const relevanceScore = cosineSimilarity(embedding, m.embedding);
        return {
          ...m,
          relevanceScore,
          finalScore: relevanceScore * m.retention,
        };
      })
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, filters?.limit ?? 5);
  }

  async updateRetentionScores(updates: Map<string, number>): Promise<void> {
    await this.ready();
    await this.withWriteLock(async () => {
      for (const [id, retention] of updates.entries()) {
        const tier = this.getTier(id);
        if (!tier) continue;
        const m = tier.get(id)!;
        const next = { ...m, retention };
        tier.set(id, next);
        await this.appendUnlocked({ type: "memory", memory: next });
      }
    });
  }

  async createOrStrengthenLink(
    sourceId: string,
    targetId: string,
    strength: number,
  ): Promise<void> {
    await this.ready();
    const [a, b] = canonicalPair(sourceId, targetId);
    const key = `${a}|${b}`;
    const existing = this.links.get(key);
    const now = this.now();
    const nextStrength = Math.min(1, (existing?.strength ?? 0) + strength);
    const row: LinkRow = {
      strength: nextStrength,
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    };
    this.links.set(key, row);
    await this.append({
      type: "link",
      a,
      b,
      strength: row.strength,
      createdAt: row.createdAt,
      updatedAt: row.updatedAt,
    });
  }

  async getLinkedMemories(
    memoryId: string,
    minStrength: number = 0.3,
  ): Promise<Array<Memory & { linkStrength: number }>> {
    return this.getLinkedMemoriesMultiple([memoryId], minStrength);
  }

  async getLinkedMemoriesMultiple(
    memoryIds: string[],
    minStrength: number = 0.3,
  ): Promise<Array<Memory & { linkStrength: number }>> {
    await this.ready();
    const out = new Map<string, Memory & { linkStrength: number }>();
    for (const id of memoryIds) {
      for (const [key, row] of this.links.entries()) {
        if (row.strength < minStrength) continue;
        const [a, b] = key.split("|");
        const other = a === id ? b : b === id ? a : null;
        if (!other) continue;
        const m = await this.getMemory(other);
        if (!m) continue;
        const prev = out.get(other);
        out.set(
          other,
          prev
            ? { ...m, linkStrength: Math.max(prev.linkStrength, row.strength) }
            : { ...m, linkStrength: row.strength },
        );
      }
    }
    return Array.from(out.values());
  }

  async deleteLink(sourceId: string, targetId: string): Promise<void> {
    await this.ready();
    const [a, b] = canonicalPair(sourceId, targetId);
    this.links.delete(`${a}|${b}`);
    await this.append({ type: "link_delete", a, b, at: this.now() });
  }

  // ------------------------------------------------------------------
  // Lexical search (BM25)
  // ------------------------------------------------------------------

  override async searchLexical(
    query: string,
    filters?: MemoryFilters,
  ): Promise<ScoredMemory[]> {
    await this.ready();
    const queryTokens = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (queryTokens.length === 0) return [];

    let items = Array.from(this.hot.values());
    if (filters?.includeCold)
      items = [...items, ...Array.from(this.cold.values())];
    if (filters?.includeStubs)
      items = [...items, ...Array.from(this.stubs.values())];
    if (!filters?.includeSuperseded)
      items = items.filter((m) => !m.isSuperseded);
    if (!filters?.includeStubs) items = items.filter((m) => !m.isStub);
    if (filters?.userId)
      items = items.filter((m) => m.userId === filters.userId);

    const N = items.length;
    if (N === 0) return [];

    const docs = items.map((m) => ({
      mem: m,
      tokens: m.content.toLowerCase().split(/\s+/).filter(Boolean),
    }));

    const avgdl = docs.reduce((sum, d) => sum + d.tokens.length, 0) / N;
    const k1 = 1.2;
    const b = 0.75;

    const df = new Map<string, number>();
    for (const t of queryTokens) {
      let count = 0;
      for (const d of docs) {
        if (d.tokens.includes(t)) count++;
      }
      df.set(t, count);
    }

    const idf = new Map<string, number>();
    for (const t of queryTokens) {
      const n = df.get(t) ?? 0;
      idf.set(t, Math.log((N - n + 0.5) / (n + 0.5) + 1.0));
    }

    const scored: ScoredMemory[] = [];
    for (const d of docs) {
      let score = 0;
      for (const t of queryTokens) {
        const tf = d.tokens.filter((tok) => tok === t).length;
        if (tf === 0) continue;
        const tfidf = idf.get(t) ?? 0;
        score += tfidf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (d.tokens.length / avgdl))));
      }
      if (score > 0) {
        scored.push({
          ...d.mem,
          relevanceScore: score,
          finalScore: score * d.mem.retention,
        });
      }
    }

    return scored
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, filters?.limit ?? 10);
  }

  async findFadingMemories(
    userId: string,
    maxRetention: number,
  ): Promise<Memory[]> {
    await this.ready();
    return [
      ...Array.from(this.hot.values()),
      ...Array.from(this.cold.values()),
    ].filter(
      (m) =>
        m.userId === userId &&
        m.retention < maxRetention &&
        !m.isSuperseded &&
        !m.isStub,
    );
  }

  async findStableMemories(
    userId: string,
    minStability: number,
    minAccessCount: number,
  ): Promise<Memory[]> {
    await this.ready();
    return Array.from(this.hot.values()).filter(
      (m) =>
        m.userId === userId &&
        m.stability >= minStability &&
        m.accessCount >= minAccessCount,
    );
  }

  async markSuperseded(memoryIds: string[], summaryId: string): Promise<void> {
    await this.ready();
    await this.withWriteLock(async () => {
      for (const id of memoryIds) {
        const tier = this.getTier(id);
        if (!tier) continue;
        const m = tier.get(id)!;
        const next: Memory = {
          ...m,
          isSuperseded: true,
          supersededBy: summaryId,
          metadata: { ...(m.metadata ?? {}), supersededBy: summaryId },
        };
        tier.set(id, next);
        await this.appendUnlocked({ type: "memory", memory: next });
      }
    });
  }

  // ------------------------------------------------------------------
  // Tiered storage
  // ------------------------------------------------------------------

  async migrateToCold(memoryId: string, coldSince: number): Promise<void> {
    await this.ready();
    const m = this.hot.get(memoryId);
    if (!m) return;
    this.hot.delete(memoryId);
    const coldMem = { ...m, isCold: true, coldSince };
    this.cold.set(memoryId, coldMem);
    await this.append({ type: "memory", memory: coldMem });
  }

  async migrateToHot(memoryId: string): Promise<void> {
    await this.ready();
    const m = this.cold.get(memoryId);
    if (!m) return;
    this.cold.delete(memoryId);
    const hotMem = { ...m, isCold: false, coldSince: null, daysAtFloor: 0 };
    this.hot.set(memoryId, hotMem);
    await this.append({ type: "memory", memory: hotMem });
  }

  async convertToStub(memoryId: string, stubContent: string): Promise<void> {
    await this.ready();
    const m = this.cold.get(memoryId) ?? this.hot.get(memoryId);
    if (!m) return;
    this.hot.delete(memoryId);
    this.cold.delete(memoryId);
    const stubMem: Memory = {
      ...m,
      content: stubContent,
      isStub: true,
      isCold: false,
      coldSince: null,
      embedding: [],
    };
    this.stubs.set(memoryId, stubMem);
    await this.append({ type: "memory", memory: stubMem });
  }

  // ------------------------------------------------------------------
  // Traversal
  // ------------------------------------------------------------------

  async allActive(): Promise<Memory[]> {
    await this.ready();
    return [
      ...Array.from(this.hot.values()),
      ...Array.from(this.cold.values()),
    ];
  }

  async allHot(): Promise<Memory[]> {
    await this.ready();
    return Array.from(this.hot.values());
  }

  async allCold(): Promise<Memory[]> {
    await this.ready();
    return Array.from(this.cold.values());
  }

  // ------------------------------------------------------------------
  // Counts
  // ------------------------------------------------------------------

  async hotCount(): Promise<number> {
    await this.ready();
    return this.hot.size;
  }

  async coldCount(): Promise<number> {
    await this.ready();
    return this.cold.size;
  }

  async stubCount(): Promise<number> {
    await this.ready();
    return this.stubs.size;
  }

  async totalCount(): Promise<number> {
    await this.ready();
    return this.hot.size + this.cold.size + this.stubs.size;
  }

  // ------------------------------------------------------------------
  // Reset
  // ------------------------------------------------------------------

  async clear(): Promise<void> {
    await this.ready();
    this.hot.clear();
    this.cold.clear();
    this.stubs.clear();
    this.links.clear();
    const meta: MemoryEvent = {
      type: "meta",
      version: 1,
      createdAt: this.now(),
    };
    await writeFile(this.path, `${JSON.stringify(meta)}\n`, "utf8");
    this.lineCount = 1;
  }

  // ------------------------------------------------------------------
  // Private
  // ------------------------------------------------------------------

  private getTier(id: string): Map<string, Memory> | null {
    if (this.hot.has(id)) return this.hot;
    if (this.cold.has(id)) return this.cold;
    if (this.stubs.has(id)) return this.stubs;
    return null;
  }

  private async load(): Promise<void> {
    await mkdir(dirname(this.path), { recursive: true });

    try {
      await stat(this.path);
    } catch {
      const meta: MemoryEvent = {
        type: "meta",
        version: 1,
        createdAt: this.now(),
      };
      await writeFile(this.path, `${JSON.stringify(meta)}\n`, "utf8");
    }

    const files = await this.listLogFiles();
    let baseLines = 0;
    for (const file of files) {
      const rl = readline.createInterface({
        input: createReadStream(file, { encoding: "utf8" }),
        crlfDelay: Number.POSITIVE_INFINITY,
      });

      let lines = 0;
      for await (const line of rl) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        lines += 1;
        const evt = JSON.parse(trimmed) as MemoryEvent;
        this.apply(evt);
      }
      if (file === this.path) baseLines = lines;
    }
    this.lineCount = baseLines;
    this.loaded = true;

    if (this.compactOnStart) {
      await this.compact();
    }
  }

  private apply(evt: MemoryEvent) {
    if (evt.type === "meta") return;
    if (evt.type === "memory") {
      const m = createDefaultMemory({
        ...evt.memory,
        id: evt.memory.id,
        userId: evt.memory.userId,
        content: evt.memory.content,
        embedding: evt.memory.embedding,
      });
      if (m.isStub) {
        this.hot.delete(m.id);
        this.cold.delete(m.id);
        this.stubs.set(m.id, m);
      } else if (m.isCold) {
        this.hot.delete(m.id);
        this.stubs.delete(m.id);
        this.cold.set(m.id, m);
      } else {
        this.cold.delete(m.id);
        this.stubs.delete(m.id);
        this.hot.set(m.id, m);
      }
      return;
    }
    if (evt.type === "memory_delete") {
      this.hot.delete(evt.id);
      this.cold.delete(evt.id);
      this.stubs.delete(evt.id);
      return;
    }
    if (evt.type === "link") {
      const key = linkKey(evt.a, evt.b);
      this.links.set(key, {
        strength: evt.strength,
        createdAt: evt.createdAt,
        updatedAt: evt.updatedAt,
      });
      return;
    }
    if (evt.type === "link_delete") {
      const key = linkKey(evt.a, evt.b);
      this.links.delete(key);
    }
  }

  private async withWriteLock<T>(fn: () => Promise<T>): Promise<T> {
    const prev = this.writeChain;
    let release!: () => void;
    this.writeChain = new Promise<void>((r) => {
      release = r;
    });
    await prev;
    try {
      return await fn();
    } finally {
      release();
    }
  }

  private async append(evt: MemoryEvent): Promise<void> {
    await this.withWriteLock(async () => this.appendUnlocked(evt));
  }

  private async appendUnlocked(evt: MemoryEvent): Promise<void> {
    const line = `${JSON.stringify(evt)}\n`;
    if (this.fsync) {
      const fh = await open(this.path, "a");
      try {
        await fh.write(line, undefined, "utf8");
        await fh.sync();
      } finally {
        await fh.close();
      }
    } else {
      await appendFile(this.path, line, "utf8");
    }
    this.lineCount += 1;
    if (this.rolloverEnabled && this.lineCount >= this.maxLines) {
      await this.rollover();
    }
  }

  private async compact(): Promise<void> {
    const archive = await this.nextArchivePath();
    await rename(this.path, archive);

    const tmp = `${this.path}.tmp`;
    const meta: MemoryEvent = {
      type: "meta",
      version: 1,
      createdAt: this.now(),
    };
    const lines: string[] = [JSON.stringify(meta)];
    for (const store of [this.hot, this.cold, this.stubs]) {
      for (const m of store.values()) {
        lines.push(
          JSON.stringify({ type: "memory", memory: m } satisfies MemoryEvent),
        );
      }
    }
    for (const [key, row] of this.links.entries()) {
      const [a, b] = key.split("|");
      lines.push(
        JSON.stringify({
          type: "link",
          a,
          b,
          strength: row.strength,
          createdAt: row.createdAt,
          updatedAt: row.updatedAt,
        } satisfies MemoryEvent),
      );
    }
    await writeFile(tmp, `${lines.join("\n")}\n`, "utf8");
    await rename(tmp, this.path);
    this.lineCount = lines.length;
  }

  private async rollover(): Promise<void> {
    const archive = await this.nextArchivePath();
    await rename(this.path, archive);
    const meta: MemoryEvent = {
      type: "meta",
      version: 1,
      createdAt: this.now(),
    };
    await writeFile(this.path, `${JSON.stringify(meta)}\n`, "utf8");
    this.lineCount = 1;
  }

  private async listLogFiles(): Promise<string[]> {
    const dir = dirname(this.path);
    const base = basename(this.path);
    const escaped = base.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`^${escaped}(?:\\.(\\d+)(?:\\.(\\d+))?)?$`);

    const entries = await readdir(dir, { withFileTypes: true });
    const matches = entries
      .filter((e) => e.isFile())
      .map((e) => e.name)
      .map((name) => ({ name, m: re.exec(name) }))
      .filter((x) => x.m)
      .map((x) => ({
        name: x.name,
        ts: x.m?.[1] ? Number.parseInt(x.m[1], 10) : Number.NaN,
        seq: x.m?.[2] ? Number.parseInt(x.m[2], 10) : 0,
      }));

    const rotated = matches
      .filter((x) => Number.isFinite(x.ts))
      .sort((a, b) => a.ts! - b.ts! || a.seq - b.seq)
      .map((x) => `${dir}/${x.name}`);

    return [...rotated, this.path];
  }

  private async nextArchivePath(): Promise<string> {
    const baseTs = this.now();
    for (let seq = 0; seq < 1000; seq += 1) {
      const candidate = `${this.path}.${baseTs}.${seq}`;
      try {
        await stat(candidate);
      } catch {
        return candidate;
      }
    }
    return `${this.path}.${baseTs}.${randomUUID()}`;
  }
}
