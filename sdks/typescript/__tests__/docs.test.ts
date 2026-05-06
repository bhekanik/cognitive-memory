import { readFile, readdir } from "node:fs/promises";
import { join, relative } from "node:path";

const repoRoot = new URL("../../../", import.meta.url).pathname;

async function markdownFiles(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "node_modules" || entry.name === "dist") continue;
      files.push(...await markdownFiles(path));
    } else if (/\.(md|mdx|astro)$/.test(entry.name)) {
      files.push(path);
    }
  }
  return files;
}

describe("documentation freshness", () => {
  test("public docs do not contain stale benchmark or API claims", async () => {
    const files = [
      join(repoRoot, "README.md"),
      join(repoRoot, "sdks/typescript/README.md"),
      join(repoRoot, "sdks/python/README.md"),
      ...await markdownFiles(join(repoRoot, "docs/src")),
    ];

    const forbidden = [
      /42\.4/,
      /47\.1/,
      /66%/,
      /relevance \* retention(?!\^alpha)/,
      /PostgresAdapter\(\{\s*connectionString/,
      /nothing is truly lost/,
    ];

    const failures: string[] = [];
    for (const file of files) {
      const text = await readFile(file, "utf8");
      for (const pattern of forbidden) {
        if (pattern.test(text)) {
          failures.push(`${relative(repoRoot, file)} contains ${pattern}`);
        }
      }
    }

    expect(failures).toEqual([]);
  });
});
