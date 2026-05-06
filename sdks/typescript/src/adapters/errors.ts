/**
 * Typed adapter errors.
 *
 * Per spec/adapter-interface.md Implementation Note 1: adapters should
 * throw typed errors rather than generic Error, so callers can catch
 * backend issues precisely.
 */

export class AdapterError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AdapterError";
  }
}

export class MemoryNotFoundError extends AdapterError {
  readonly memoryId: string;

  constructor(memoryId: string) {
    super(`Memory not found: ${memoryId}`);
    this.name = "MemoryNotFoundError";
    this.memoryId = memoryId;
  }
}

export class DuplicateMemoryError extends AdapterError {
  readonly memoryId: string;

  constructor(memoryId: string) {
    super(`Memory already exists: ${memoryId}`);
    this.name = "DuplicateMemoryError";
    this.memoryId = memoryId;
  }
}
