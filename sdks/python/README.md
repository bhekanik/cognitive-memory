# cognitive-memory

Biologically-inspired agent memory with decay, consolidation, and tiered storage.

[![PyPI version](https://img.shields.io/pypi/v/cognitive-memory.svg)](https://pypi.org/project/cognitive-memory/)

## Install

```bash
pip install cognitive-memory
```

Requires Python 3.10+.

## Quick Start

```python
from cognitive_memory import SyncCognitiveMemory

mem = SyncCognitiveMemory(embedder="hash")

mem.add("User is allergic to shellfish", category="core", importance=0.95)

response = mem.search("what allergies does the user have?")
for r in response.results:
    print(r.memory.content, f"(score: {r.combined_score:.2f})")
```

For production, install the OpenAI extra and set `OPENAI_API_KEY`:

```bash
pip install "cognitive-memory[openai]"
```

```python
from cognitive_memory import CognitiveMemory

mem = CognitiveMemory()  # defaults to OpenAI embeddings

await mem.add("User prefers dark mode", category="semantic", importance=0.7)
response = await mem.search("UI preferences")
```

For sync usage in scripts and notebooks, use `SyncCognitiveMemory` (same API, no `await`).

## Docs

Full documentation, guides, and API reference at **[bhekanik.github.io/cognitive-memory](https://bhekanik.github.io/cognitive-memory)**.

## License

[MIT](../../LICENSE)
