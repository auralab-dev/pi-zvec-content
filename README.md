# pi-zvec-content

One bounded Pi content tool:

```text
file_content_search(path, query)
```

Use it for a project-relative file/directory discovered by `find` or returned by `download_file`. It returns short bounded excerpts rather than whole files.

Behavior:
- resolves symlinks and rejects paths outside the current project;
- serializes operations per target so parallel agent calls cannot race the same zvec index;
- normal queries use zvec hybrid indexed retrieval;
- intentional regex-like queries automatically use managed `zg query --rg`;
- zvec stdout returned to the model is capped by `maxResultChars`;
- zvec failures are thrown as real Pi tool errors with bounded, path-redacted diagnostics;
- private indexes/models/state live under the Pi config directory.

`.pi/zvec-content.json`:

```json
{
  "embedding": "local/potion-multilingual-128m",
  "limit": 8,
  "maxResultChars": 4000
}
```

Requires Node.js 22.19+ and `pnpm install` at the harness root.

## Runtime mode

Indexed operations use zvec `auto` mode. The first search creates the per-target index; later searches query it with `--refresh wait` instead of invoking `zg index` again. This avoids daemon write-lease conflicts while still picking up file changes.

## Retrieval pipeline (0.5)

- Polish/multilingual default: `local/potion-multilingual-128m`.
- Indexed search uses fused `--hybrid` + `--fts` candidate retrieval.
- `limit` is the internal candidate pool; final model-visible output remains bounded by `maxResultChars`.
- zvec previews are disabled. The wrapper parses ranked source locations and privately reopens only the matched Markdown sections to construct evidence.
- Markdown containing `<!-- Page N -->` markers is projected into private per-page search files under `.pi/zvec-project-indexes/`; HTML tables are normalized to explicit `TABLE N ROW N: CELL N: ...` records there. The original `content.md` is not modified.
- Queries containing literal dates, numbered sections, `Art.`/`§` identifiers, or quoted phrases automatically run a private exact-verification lane over that projection. Exact hits outrank semantic candidates, and semantic candidates that omit all hard anchors are excluded from evidence.
- Evidence expansion recognizes Markdown headings and numbered procedural clauses such as `3.4.16`, then reads only that bounded section.
- Footnote markers in selected evidence trigger same-page footnote-context expansion when the note can be located.
- A wrapper metadata file records the embedding model; changing `embedding` automatically rebuilds an existing private index once.
