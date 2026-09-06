import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import {
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import type { Stats } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  basename,
  dirname,
  isAbsolute,
  join,
  relative,
  resolve,
  sep,
} from "node:path";

const DEFAULT_EMBEDDING = "local/potion-multilingual-128m";
const DEFAULT_RESULT_LIMIT = 8;
const DEFAULT_MAX_RESULT_CHARS = 4000;
const PROCESS_OUTPUT_LIMIT = 64_000;
const DIAGNOSTIC_TAIL_CHARS = 4000;
const WRAPPER_INDEX_VERSION = 3;
const MAX_EVIDENCE_HITS = 4;
const MAX_EXACT_MATCHES_PER_ANCHOR = 8;
const MAX_EXACT_FILES = 5000;
const PAGE_MARKER_RE = /^<!--\s*Page\s+(\d+)\s*-->\s*$/gm;

export interface ZvecContentConfig {
  embedding: string;
  limit: number;
  maxResultChars: number;
}

export class ZgProcessError extends Error {
  readonly stage: "index" | "query" | "rg";
  readonly exitCode: number | null;
  readonly signal: NodeJS.Signals | null;
  readonly stdoutTail: string;
  readonly stderrTail: string;

  constructor(args: {
    stage: "index" | "query" | "rg";
    exitCode: number | null;
    signal: NodeJS.Signals | null;
    stdoutTail: string;
    stderrTail: string;
  }) {
    const diagnostic = args.stderrTail || args.stdoutTail;
    super(`zvec-grep ${args.stage} failed with exit ${args.exitCode ?? "unknown"}${diagnostic ? `\n${diagnostic}` : ""}`);
    this.name = "ZgProcessError";
    this.stage = args.stage;
    this.exitCode = args.exitCode;
    this.signal = args.signal;
    this.stdoutTail = args.stdoutTail;
    this.stderrTail = args.stderrTail;
  }
}

function configDir(): string {
  const explicit = process.env.PI_CODING_AGENT_DIR?.trim();
  if (explicit) return resolve(explicit);
  return resolve(process.cwd(), ".pi");
}

function configPath(): string {
  return join(configDir(), "zvec-content.json");
}

function positiveInt(value: unknown, fallback: number, max: number): number {
  return typeof value === "number" && Number.isInteger(value) && value > 0
    ? Math.min(value, max)
    : fallback;
}

export function loadZvecContentConfig(): ZvecContentConfig {
  let raw: Record<string, unknown> = {};
  const path = configPath();
  if (existsSync(path)) {
    try {
      const parsed = JSON.parse(readFileSync(path, "utf8"));
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) raw = parsed as Record<string, unknown>;
    } catch (err) {
      throw new Error(`Failed to parse ${path}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  return {
    embedding: typeof raw.embedding === "string" && raw.embedding.trim() ? raw.embedding.trim() : DEFAULT_EMBEDDING,
    limit: positiveInt(raw.limit, DEFAULT_RESULT_LIMIT, 20),
    maxResultChars: positiveInt(raw.maxResultChars, DEFAULT_MAX_RESULT_CHARS, 20_000),
  };
}

function ensurePrivateDirectory(path: string): void {
  mkdirSync(path, { recursive: true, mode: 0o700 });
  const stat = lstatSync(path);
  if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error("Unsafe private zvec directory");
}

function zvecHome(): string {
  return join(configDir(), "zvec-grep");
}

function projectIndexesDir(): string {
  return join(configDir(), "zvec-project-indexes");
}

function resolveZgCli(): string {
  let entry: string;
  try {
    entry = fileURLToPath(import.meta.resolve("@zvec/zvec-grep"));
  } catch {
    throw new Error("zvec-grep is not installed; run pnpm install at the harness root");
  }
  const cli = join(dirname(entry), "cli", "index.js");
  if (!existsSync(cli)) throw new Error("zvec-grep CLI not found");
  return cli;
}

function localEnv(config: ZvecContentConfig): NodeJS.ProcessEnv {
  const home = zvecHome();
  const models = join(home, "models");
  ensurePrivateDirectory(home);
  ensurePrivateDirectory(models);
  return {
    ...process.env,
    ZVEC_GREP_HOME: home,
    ZVEC_GREP_MODEL_CACHE: models,
    ZVEC_GREP_MODE: "auto",
    ZVEC_GREP_EMBEDDING: config.embedding,
  };
}

function tail(text: string, max = DIAGNOSTIC_TAIL_CHARS): string {
  const trimmed = text.trim();
  return trimmed.length <= max ? trimmed : `[...truncated...]\n${trimmed.slice(-max)}`;
}

function redactDiagnostic(text: string, cwd: string): string {
  if (!text) return "";
  const config = configDir();
  const home = zvecHome();
  const indexes = projectIndexesDir();
  const replacements: Array<[string, string]> = [
    [cwd, "<zvec-workspace>"],
    [cwd.replaceAll("\\", "/"), "<zvec-workspace>"],
    [indexes, "<zvec-indexes>"],
    [indexes.replaceAll("\\", "/"), "<zvec-indexes>"],
    [home, "<zvec-home>"],
    [home.replaceAll("\\", "/"), "<zvec-home>"],
    [config, "<pi-config>"],
    [config.replaceAll("\\", "/"), "<pi-config>"],
  ];
  let result = text;
  for (const [from, to] of replacements) {
    if (from) result = result.replaceAll(from, to);
  }
  return tail(result);
}

async function runZg(
  stage: "index" | "query" | "rg",
  args: string[],
  cwd: string,
  config: ZvecContentConfig,
  signal?: AbortSignal,
): Promise<string> {
  const cli = resolveZgCli();
  return await new Promise<string>((resolvePromise, reject) => {
    const child = spawn(process.execPath, [cli, ...args], {
      cwd,
      env: localEnv(config),
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";

    const capture = (current: string, chunk: Buffer | string): string => {
      if (current.length >= PROCESS_OUTPUT_LIMIT) return current;
      const text = String(chunk);
      const room = PROCESS_OUTPUT_LIMIT - current.length;
      return current + text.slice(0, room);
    };

    child.stdout?.on("data", (chunk) => { stdout = capture(stdout, chunk); });
    child.stderr?.on("data", (chunk) => { stderr = capture(stderr, chunk); });

    const abort = () => child.kill("SIGTERM");
    if (signal) {
      if (signal.aborted) abort();
      else signal.addEventListener("abort", abort, { once: true });
    }

    child.on("error", reject);
    child.on("close", (code, killedSignal) => {
      signal?.removeEventListener("abort", abort);
      if (signal?.aborted) return reject(new Error("zvec-grep operation aborted"));
      if (code !== 0) {
        return reject(new ZgProcessError({
          stage,
          exitCode: code,
          signal: killedSignal,
          stdoutTail: redactDiagnostic(stdout, cwd),
          stderrTail: redactDiagnostic(stderr, cwd),
        }));
      }
      resolvePromise(stdout);
    });
  });
}

const targetLocks = new Map<string, Promise<void>>();

async function withTargetLock<T>(key: string, signal: AbortSignal | undefined, fn: () => Promise<T>): Promise<T> {
  const previous = targetLocks.get(key) ?? Promise.resolve();
  let release!: () => void;
  const gate = new Promise<void>((resolveGate) => { release = resolveGate; });
  const queued = previous.catch(() => {}).then(() => gate);
  targetLocks.set(key, queued);
  await previous.catch(() => {});
  signal?.throwIfAborted();
  try {
    return await fn();
  } finally {
    release();
    if (targetLocks.get(key) === queued) targetLocks.delete(key);
  }
}

function looksLikeRegex(query: string): boolean {
  return /(^|[^\\])(?:\.\*|\.\+|\[[^\]]+\]|\([^)]*\)|\{\d|\^|\$)|\\[dDsSwWbB]/.test(query);
}

interface Target {
  projectRoot: string;
  targetPath: string;
  displayPath: string;
  indexDir: string;
}

interface ProjectionMeta {
  version: number;
  sourcePath: string;
  sourceSize: number;
  sourceMtimeMs: number;
  projected: boolean;
}

interface WrapperIndexMeta {
  version: number;
  embedding: string;
}

type AnchorKind = "date" | "section" | "quote";
type EvidenceKind = "EXACT" | "EXACT_SECTION" | "HYBRID_VERIFIED" | "HYBRID_DISCOVERY";

interface HardAnchor {
  kind: AnchorKind;
  raw: string;
  canonical: string;
}

interface ZgHit {
  rank: number;
  sourcePath: string;
  sourceStart: number;
  sourceEnd: number;
  matchedStart?: number;
  matchedEnd?: number;
  matchedBy?: string;
  heading?: string;
  scope?: string;
  evidenceKind?: EvidenceKind;
  exactAnchors?: string[];
}

function inside(root: string, candidate: string): boolean {
  return candidate === root || candidate.startsWith(root + sep);
}

function resolveTarget(cwd: string, pathValue: string): Target {
  const requested = pathValue.trim();
  if (!requested) throw new Error("path must not be empty");
  if (isAbsolute(requested) || requested.startsWith("~")) throw new Error("path must be project-relative");

  const projectRoot = realpathSync(cwd);
  let targetPath: string;
  try {
    targetPath = realpathSync(resolve(projectRoot, requested));
  } catch {
    throw new Error("path does not exist");
  }
  if (!inside(projectRoot, targetPath)) throw new Error("path resolves outside the project");

  const stat = statSync(targetPath);
  if (!stat.isFile() && !stat.isDirectory()) throw new Error("path must resolve to a regular file or directory");

  const indexes = resolve(projectIndexesDir());
  const zhome = resolve(zvecHome());
  if (inside(indexes, targetPath) || inside(zhome, targetPath)) throw new Error("cannot search private zvec state");

  const displayPath = relative(projectRoot, targetPath).replaceAll("\\", "/") || ".";
  const key = createHash("sha256")
    .update(projectRoot)
    .update("\0")
    .update(targetPath)
    .digest("hex")
    .slice(0, 24);
  return { projectRoot, targetPath, displayPath, indexDir: join(projectIndexesDir(), key) };
}

function decodeHtml(text: string): string {
  const named: Record<string, string> = {
    amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", ndash: "–", mdash: "—",
  };
  return text
    .replace(/&#(\d+);/g, (_m, n) => String.fromCodePoint(Number(n)))
    .replace(/&#x([0-9a-f]+);/gi, (_m, n) => String.fromCodePoint(parseInt(n, 16)))
    .replace(/&([a-z]+);/gi, (m, name) => named[name.toLowerCase()] ?? m);
}

function htmlCellText(text: string): string {
  return decodeHtml(text
    .replace(/<sup\b[^>]*>(.*?)<\/sup>/gis, " [footnote $1] ")
    .replace(/<br\s*\/?\s*>/gi, " / ")
    .replace(/<[^>]+>/g, " "))
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeHtmlTables(markdown: string): string {
  let tableNo = 0;
  return markdown.replace(/<table\b[^>]*>[\s\S]*?<\/table>/gi, (table) => {
    tableNo += 1;
    const rows: string[] = [];
    const trRe = /<tr\b[^>]*>([\s\S]*?)<\/tr>/gi;
    let tr: RegExpExecArray | null;
    let rowNo = 0;
    while ((tr = trRe.exec(table)) !== null) {
      const cells: string[] = [];
      const cellRe = /<(?:th|td)\b[^>]*>([\s\S]*?)<\/(?:th|td)>/gi;
      let cell: RegExpExecArray | null;
      while ((cell = cellRe.exec(tr[1])) !== null) {
        const value = htmlCellText(cell[1]);
        if (value) cells.push(value);
      }
      if (cells.length) {
        rowNo += 1;
        const explicit = cells.map((value, index) => `CELL ${index + 1}: ${value}`).join(" | ");
        rows.push(`TABLE ${tableNo} ROW ${rowNo}: ${explicit}`);
      }
    }
    return rows.length ? `\n\n## Table ${tableNo} rows\n\n${rows.join("\n")}\n\n` : `\n\n${htmlCellText(table)}\n\n`;
  });
}

function wrapLongLines(text: string, width = 520): string {
  const out: string[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (line.length <= width || /^\s*(?:#|TABLE\s+\d+\s+ROW|[-*+]\s|\d+[.)]\s)/.test(line)) {
      out.push(line);
      continue;
    }
    let rest = line;
    while (rest.length > width) {
      const window = rest.slice(0, width + 120);
      const cuts = [window.lastIndexOf(". "), window.lastIndexOf("; "), window.lastIndexOf(": "), window.lastIndexOf(" ")];
      let cut = Math.max(...cuts);
      if (cut < Math.floor(width / 2)) cut = width;
      else cut += 1;
      out.push(rest.slice(0, cut).trimEnd());
      rest = rest.slice(cut).trimStart();
    }
    if (rest) out.push(rest);
  }
  return out.join("\n");
}

function splitPages(markdown: string): Array<{ page: number; text: string }> {
  const matches = [...markdown.matchAll(PAGE_MARKER_RE)];
  if (!matches.length) return [];
  const pages: Array<{ page: number; text: string }> = [];
  for (let i = 0; i < matches.length; i += 1) {
    const match = matches[i];
    const page = Number(match[1]);
    const start = (match.index ?? 0) + match[0].length;
    const end = i + 1 < matches.length ? (matches[i + 1].index ?? markdown.length) : markdown.length;
    const body = markdown.slice(start, end).trim();
    pages.push({ page, text: body });
  }
  return pages;
}

const POLISH_MONTHS: Record<string, number> = {
  stycznia: 1,
  lutego: 2,
  marca: 3,
  kwietnia: 4,
  maja: 5,
  czerwca: 6,
  lipca: 7,
  sierpnia: 8,
  września: 9,
  wrzesnia: 9,
  października: 10,
  pazdziernika: 10,
  listopada: 11,
  grudnia: 12,
};
const POLISH_MONTH_PATTERN = Object.keys(POLISH_MONTHS).join("|");
const NAMED_DATE_RE = new RegExp(`\\b(\\d{1,2})\\s+(${POLISH_MONTH_PATTERN})(?:\\s+(\\d{4})(?:\\s*r\\.?)?)?`, "giu");
const NUMERIC_DATE_RE = /\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b/g;
const SECTION_RE = /\b\d+(?:\.\d+){1,5}\.?\b/g;
const LAW_SECTION_RE = /§\s*\d+[a-ząćęłńóśźż]{0,6}\b/giu;
const LAW_ARTICLE_RE = /\b(?:art\.?|artykuł)\s*\d+[a-ząćęłńóśźż]{0,6}\b/giu;
const QUOTED_RE = /["„“”]([^"„“”]{4,160})["„“”]/gu;

function canonicalDate(day: string, month: string | number, year?: string): string {
  const monthNo = typeof month === "number" ? month : POLISH_MONTHS[month.toLocaleLowerCase("pl-PL")];
  const dd = String(Number(day)).padStart(2, "0");
  const mm = String(Number(monthNo)).padStart(2, "0");
  return year ? `date:${dd}-${mm}-${year}` : `date:${dd}-${mm}`;
}

function normalizeExactText(value: string): string {
  let text = decodeHtml(value.normalize("NFKC").replace(/\u00a0/g, " ").toLocaleLowerCase("pl-PL"));
  text = text.replace(NUMERIC_DATE_RE, (_m, day: string, month: string, year: string) => ` ${canonicalDate(day, Number(month), year)} `);
  text = text.replace(NAMED_DATE_RE, (_m, day: string, month: string, year?: string) => ` ${canonicalDate(day, month, year)} `);
  text = text
    .replace(/<sup\b[^>]*>(.*?)<\/sup>/gis, " footnote $1 ")
    .replace(/<[^>]+>/g, " ")
    .replace(/[\*_`~>#|]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return text;
}

function canonicalAnchor(kind: AnchorKind, raw: string): string {
  if (kind === "date") return normalizeExactText(raw);
  if (kind === "section") return raw.toLocaleLowerCase("pl-PL").replace(/\s+/g, "").replace(/\.$/, "");
  return normalizeExactText(raw);
}

function extractHardAnchors(query: string): HardAnchor[] {
  const anchors: HardAnchor[] = [];
  const seen = new Set<string>();
  const add = (kind: AnchorKind, raw: string) => {
    const cleaned = raw.trim();
    if (!cleaned) return;
    const canonical = canonicalAnchor(kind, cleaned);
    const key = `${kind}:${canonical}`;
    if (!canonical || seen.has(key)) return;
    seen.add(key);
    anchors.push({ kind, raw: cleaned, canonical });
  };

  for (const match of query.matchAll(NAMED_DATE_RE)) add("date", match[0]);
  for (const match of query.matchAll(NUMERIC_DATE_RE)) add("date", match[0]);
  for (const match of query.matchAll(LAW_SECTION_RE)) add("section", match[0]);
  for (const match of query.matchAll(LAW_ARTICLE_RE)) add("section", match[0]);
  for (const match of query.matchAll(SECTION_RE)) {
    const raw = match[0];
    const parts = raw.replace(/\.$/, "").split(".");
    const maybeDate = parts.length === 3 && Number(parts[0]) <= 31 && Number(parts[1]) <= 12 && parts[2].length === 4;
    if (!maybeDate) add("section", raw);
  }
  for (const match of query.matchAll(QUOTED_RE)) add("quote", match[1]);
  return anchors;
}

function regexEscape(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function anchorMatches(text: string, anchor: HardAnchor): boolean {
  if (anchor.kind === "section") {
    const raw = anchor.raw.trim().replace(/\.$/, "");
    const paragraph = /^§\s*(\d+[a-ząćęłńóśźż]{0,6})$/iu.exec(raw);
    if (paragraph) return new RegExp(`§\\s*${regexEscape(paragraph[1])}(?![\\p{L}\\p{N}])`, "iu").test(text);
    const article = /^(?:art\.?|artykuł)\s*(\d+[a-ząćęłńóśźż]{0,6})$/iu.exec(raw);
    if (article) return new RegExp(`\\b(?:art\\.?|artykuł)\\s*${regexEscape(article[1])}(?![\\p{L}\\p{N}])`, "iu").test(text);
    const dotted = /^(\d+(?:\.\d+){1,5})$/u.exec(raw);
    if (dotted) return new RegExp(`(?:^|[^\\d.])${regexEscape(dotted[1])}(?!\\d)`, "u").test(text);
    return false;
  }
  return normalizeExactText(text).includes(anchor.canonical);
}

interface SearchFile {
  logicalPath: string;
  actualPath: string;
}

function listPrivateSearchFiles(target: Target): SearchFile[] {
  const root = join(target.indexDir, "source");
  const files: SearchFile[] = [];
  const seenDirs = new Set<string>();
  const allowedIndexRoot = realpathSync(target.indexDir);

  const visit = (logicalPath: string) => {
    if (files.length >= MAX_EXACT_FILES) return;
    let actualPath: string;
    let stat: ReturnType<typeof statSync>;
    try {
      actualPath = realpathSync(logicalPath);
      stat = statSync(logicalPath);
    } catch {
      return;
    }
    if (!inside(target.projectRoot, actualPath) && !inside(allowedIndexRoot, actualPath)) return;
    if (stat.isDirectory()) {
      if (seenDirs.has(actualPath)) return;
      seenDirs.add(actualPath);
      let names: string[];
      try { names = readdirSync(logicalPath); } catch { return; }
      for (const name of names) {
        if (files.length >= MAX_EXACT_FILES) break;
        visit(join(logicalPath, name));
      }
      return;
    }
    if (!stat.isFile()) return;
    if (!/\.(?:md|mdx|txt|html?|xml|json|ya?ml|csv)$/i.test(logicalPath)) return;
    files.push({
      logicalPath: relative(target.indexDir, logicalPath).replaceAll("\\", "/"),
      actualPath,
    });
  };

  visit(root);
  return files;
}

function exactAnchorHits(target: Target, anchors: HardAnchor[]): ZgHit[] {
  if (!anchors.length) return [];
  const searchFiles = listPrivateSearchFiles(target);
  const groups: ZgHit[][] = [];

  for (const anchor of anchors) {
    const group: ZgHit[] = [];
    for (const file of searchFiles) {
      if (group.length >= MAX_EXACT_MATCHES_PER_ANCHOR) break;
      let text: string;
      try { text = readFileSync(file.actualPath, "utf8"); } catch { continue; }
      const lines = text.split(/\r?\n/);
      let lastMatchedEnd = 0;
      for (let i = 0; i < lines.length && group.length < MAX_EXACT_MATCHES_PER_ANCHOR; i += 1) {
        if (i + 1 <= lastMatchedEnd) continue;
        const end = Math.min(lines.length, i + 3);
        const window = lines.slice(i, end).join("\n");
        if (!anchorMatches(window, anchor)) continue;
        let matchStart = i;
        let matchEnd = end - 1;
        for (let j = i; j < end; j += 1) {
          if (anchorMatches(lines[j], anchor)) {
            matchStart = j;
            matchEnd = j;
            break;
          }
        }
        group.push({
          rank: 0,
          sourcePath: file.logicalPath,
          sourceStart: matchStart + 1,
          sourceEnd: matchEnd + 1,
          matchedStart: matchStart + 1,
          matchedEnd: matchEnd + 1,
          matchedBy: "exact",
          evidenceKind: "EXACT",
          exactAnchors: [anchor.raw],
        });
        lastMatchedEnd = matchEnd + 1;
      }
    }
    groups.push(group);
  }

  // Interleave anchors so a multi-date query cannot spend the entire bounded
  // evidence budget on repeated occurrences of the first date.
  const hits: ZgHit[] = [];
  for (let index = 0; ; index += 1) {
    let added = false;
    for (const group of groups) {
      if (group[index]) {
        hits.push(group[index]);
        added = true;
      }
    }
    if (!added) break;
  }
  return hits;
}

function readJsonObject(path: string): Record<string, unknown> | null {
  if (!existsSync(path)) return null;
  try {
    const value = JSON.parse(readFileSync(path, "utf8"));
    return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
  } catch {
    return null;
  }
}

function projectionMetaPath(target: Target): string {
  return join(target.indexDir, ".pi-content-projection.json");
}

function indexMetaPath(target: Target): string {
  return join(target.indexDir, ".pi-zvec-wrapper.json");
}

function sameProjectionMeta(meta: Record<string, unknown> | null, target: Target, stat: Stats): boolean {
  return Boolean(meta
    && meta.version === WRAPPER_INDEX_VERSION
    && meta.sourcePath === target.targetPath
    && meta.sourceSize === stat.size
    && typeof meta.sourceMtimeMs === "number"
    && Math.abs(meta.sourceMtimeMs - stat.mtimeMs) < 1);
}

function prepareIndex(target: Target): void {
  ensurePrivateDirectory(projectIndexesDir());
  ensurePrivateDirectory(target.indexDir);
  const sourceRoot = join(target.indexDir, "source");
  const stat = statSync(target.targetPath);
  if (!stat) throw new Error("target disappeared while preparing zvec index");
  const metaPath = projectionMetaPath(target);
  const oldMeta = readJsonObject(metaPath);

  if (sameProjectionMeta(oldMeta, target, stat) && existsSync(sourceRoot)) return;

  rmSync(sourceRoot, { recursive: true, force: true });
  ensurePrivateDirectory(sourceRoot);

  let projected = false;
  if (stat.isFile() && /\.md$/i.test(target.targetPath)) {
    const markdown = readFileSync(target.targetPath, "utf8");
    const pages = splitPages(markdown);
    if (pages.length >= 2) {
      projected = true;
      const pagesDir = join(sourceRoot, "pages");
      ensurePrivateDirectory(pagesDir);
      for (const item of pages) {
        const normalized = wrapLongLines(normalizeHtmlTables(item.text));
        const filename = `${String(item.page).padStart(4, "0")}.md`;
        writeFileSync(join(pagesDir, filename), `# Page ${item.page}\n\n${normalized.trim()}\n`, { encoding: "utf8", mode: 0o600 });
      }
    }
  }

  if (!projected) {
    const name = basename(target.targetPath) || "target";
    const link = join(sourceRoot, name);
    symlinkSync(target.targetPath, link, stat.isDirectory() ? "dir" : "file");
  }

  const meta: ProjectionMeta = {
    version: WRAPPER_INDEX_VERSION,
    sourcePath: target.targetPath,
    sourceSize: stat.size,
    sourceMtimeMs: stat.mtimeMs,
    projected,
  };
  writeFileSync(metaPath, JSON.stringify(meta, null, 2) + "\n", { encoding: "utf8", mode: 0o600 });
}

function wrapperIndexMatches(target: Target, config: ZvecContentConfig): boolean {
  const meta = readJsonObject(indexMetaPath(target));
  return Boolean(meta && meta.version === WRAPPER_INDEX_VERSION && meta.embedding === config.embedding);
}

function writeWrapperIndexMeta(target: Target, config: ZvecContentConfig): void {
  const meta: WrapperIndexMeta = { version: WRAPPER_INDEX_VERSION, embedding: config.embedding };
  writeFileSync(indexMetaPath(target), JSON.stringify(meta, null, 2) + "\n", { encoding: "utf8", mode: 0o600 });
}

async function ensureIndex(target: Target, config: ZvecContentConfig, signal?: AbortSignal): Promise<void> {
  prepareIndex(target);
  const hasIndex = existsSync(join(target.indexDir, ".zvec-grep"));
  const compatible = hasIndex && wrapperIndexMatches(target, config);
  if (compatible) return;

  const args = [
    "index",
    "--debug",
    ...(hasIndex ? ["--rebuild"] : []),
    "--embedding", config.embedding,
    "-L",
    "--no-ignore",
    "-g", "source",
    "-g", "source/**",
    "-g", "!source/.pi/zvec-project-indexes/**",
    "-g", "!source/.pi/zvec-grep/**",
    "--mode", "auto",
  ];
  await runZg("index", args, target.indexDir, config, signal);
  writeWrapperIndexMeta(target, config);
}

async function searchRegex(target: Target, query: string, config: ZvecContentConfig, signal?: AbortSignal): Promise<string> {
  prepareIndex(target);
  return runZg("rg", [
    "query",
    "--rg",
    "-i",
    "-C", "2",
    query,
    "source",
  ], target.indexDir, config, signal);
}

function parseRange(value: string): { start: number; end: number } | null {
  const match = /^(\d+)(?:-(\d+))?$/.exec(value.trim());
  if (!match) return null;
  const start = Number(match[1]);
  const end = Number(match[2] ?? match[1]);
  return start > 0 && end >= start ? { start, end } : null;
}

function parseZgHits(stdout: string): ZgHit[] {
  const lines = stdout.split(/\r?\n/);
  const hits: ZgHit[] = [];
  let current: ZgHit | null = null;
  for (const line of lines) {
    const header = /^#(\d+)\s+(.*?)\s+([^\s:]+(?:\/[^\s:]*)?):(\d+)-(\d+)\s*$/.exec(line.trim());
    if (header) {
      const attrs = header[2];
      const matchedBy = /(?:^|\s)matchedBy=([^\s]+)/.exec(attrs)?.[1];
      current = {
        rank: Number(header[1]),
        sourcePath: header[3],
        sourceStart: Number(header[4]),
        sourceEnd: Number(header[5]),
        ...(matchedBy ? { matchedBy } : {}),
      };
      hits.push(current);
      continue;
    }
    if (!current) continue;
    const trimmed = line.trim();
    if (trimmed.startsWith("matched:")) {
      const range = parseRange(trimmed.slice("matched:".length));
      if (range) {
        current.matchedStart = range.start;
        current.matchedEnd = range.end;
      }
    } else if (trimmed.startsWith("heading:")) {
      current.heading = trimmed.slice("heading:".length).trim();
    } else if (trimmed.startsWith("scope:")) {
      current.scope = trimmed.slice("scope:".length).trim();
    }
  }
  return hits;
}

function resolveHitFile(target: Target, sourcePath: string): string | null {
  const clean = sourcePath.replaceAll("\\", "/").replace(/^\.\//, "");
  if (!clean.startsWith("source/")) return null;
  const candidate = resolve(target.indexDir, clean);
  try {
    const actual = realpathSync(candidate);
    if (inside(target.projectRoot, actual) || inside(realpathSync(target.indexDir), actual)) return actual;
  } catch {}
  return null;
}

function numberedSectionId(line: string): string | null {
  const cleaned = line.replace(/^\s*#{1,6}\s*/, "").replace(/^\s*(?:\*\*|__)/, "");
  return /^(\d+(?:\.\d+){1,5})\.?(?:\*\*|__)?(?:\s+|$)/.exec(cleaned)?.[1] ?? null;
}

function legalSectionKey(line: string): string | null {
  const cleaned = line.replace(/^\s*#{1,6}\s*/, "").replace(/^\s*(?:\*\*|__)/, "");
  const paragraph = /^(§)\s*(\d+[a-ząćęłńóśźż]{0,6})\.?/iu.exec(cleaned);
  if (paragraph) return `§:${paragraph[2].toLocaleLowerCase("pl-PL")}`;
  const article = /^(?:art\.?|artykuł)\s*(\d+[a-ząćęłńóśźż]{0,6})\.?/iu.exec(cleaned);
  if (article) return `art:${article[1].toLocaleLowerCase("pl-PL")}`;
  return null;
}

function structuralSection(lines: string[], anchor: number): { start: number; end: number; structured: boolean } {
  const idx = Math.max(0, Math.min(lines.length - 1, anchor - 1));

  // Legal acts often render Art./§ clauses as plain text. Keep the exact
  // article/paragraph bounded until the next legal clause.
  for (let i = idx; i >= 0; i -= 1) {
    const key = legalSectionKey(lines[i]);
    if (!key) continue;
    let end = lines.length - 1;
    for (let j = i + 1; j < lines.length; j += 1) {
      const next = legalSectionKey(lines[j]);
      if (next && next !== key) { end = j - 1; break; }
    }
    return { start: i, end, structured: true };
  }

  // Procedural PDFs often render numbered clauses such as 3.4.16 as plain
  // text rather than Markdown headings. Prefer the nearest such clause when
  // present and stop at the next sibling/ancestor clause.
  for (let i = idx; i >= 0; i -= 1) {
    const id = numberedSectionId(lines[i]);
    if (!id) continue;
    const depth = id.split(".").length;
    let end = lines.length - 1;
    for (let j = i + 1; j < lines.length; j += 1) {
      const next = numberedSectionId(lines[j]);
      if (next && next !== id && next.split(".").length <= depth) {
        end = j - 1;
        break;
      }
    }
    return { start: i, end, structured: true };
  }

  let start = idx;
  let headingLevel = 7;
  for (let i = idx; i >= 0; i -= 1) {
    const match = /^(#{1,6})\s+/.exec(lines[i]);
    if (match) {
      start = i;
      headingLevel = match[1].length;
      break;
    }
  }
  let end = lines.length - 1;
  for (let i = Math.max(start + 1, idx + 1); i < lines.length; i += 1) {
    const match = /^(#{1,6})\s+/.exec(lines[i]);
    if (match && match[1].length <= headingLevel) {
      end = i - 1;
      break;
    }
  }
  return { start, end, structured: start !== idx };
}

function sliceAroundMatch(lines: string[], hit: ZgHit, charBudget: number): string {
  if (!lines.length) return "";
  const matchedStart = hit.matchedStart ?? hit.sourceStart;
  const matchedEnd = hit.matchedEnd ?? matchedStart;
  const section = structuralSection(lines, matchedStart);
  const sectionText = lines.slice(section.start, section.end + 1).join("\n").trim();
  if (sectionText.length <= charBudget) return sectionText;

  let start = Math.max(section.start, matchedStart - 1);
  let end = Math.min(section.end, matchedEnd - 1);
  let current = lines.slice(start, end + 1).join("\n").trim();
  let turn = 0;
  while (current.length < charBudget && (start > section.start || end < section.end)) {
    if ((turn % 2 === 0 && start > section.start) || end >= section.end) start -= 1;
    else if (end < section.end) end += 1;
    const next = lines.slice(start, end + 1).join("\n").trim();
    if (next.length > charBudget && current) break;
    current = next;
    turn += 1;
  }
  return current.slice(0, charBudget).trim();
}

function footnoteNumbers(text: string): string[] {
  const numbers = new Set<string>();
  for (const match of text.matchAll(/<sup\b[^>]*>\s*(\d+)\s*<\/sup>/gi)) numbers.add(match[1]);
  for (const match of text.matchAll(/\[footnote\s+(\d+)\]/gi)) numbers.add(match[1]);
  return [...numbers];
}

function findFootnoteContext(lines: string[], number: string): string {
  // Prefer definitions, not inline references such as "[footnote 11]" in a table cell.
  const patterns = [
    new RegExp(`^\\s*${number}[.)]\\s+`, "i"),
    new RegExp(`^\\s*<sup[^>]*>\\s*${number}\\s*<\\/sup>\\s+`, "i"),
    new RegExp(`^\\s*(?:przypis|footnote)\\s*${number}\\s*[:.)-]\\s*`, "i"),
  ];
  for (let i = 0; i < lines.length; i += 1) {
    if (patterns.some((pattern) => pattern.test(lines[i]))) {
      return lines.slice(i, Math.min(lines.length, i + 4)).join("\n").trim();
    }
  }
  return "";
}

function evidenceKey(target: Target, hit: ZgHit): string {
  const file = resolveHitFile(target, hit.sourcePath);
  if (!file) return `${hit.sourcePath}:${hit.matchedStart ?? hit.sourceStart}-${hit.matchedEnd ?? hit.sourceEnd}`;
  try {
    const lines = readFileSync(file, "utf8").split(/\r?\n/);
    const section = structuralSection(lines, hit.matchedStart ?? hit.sourceStart);
    return `${hit.sourcePath}:section:${section.start}-${section.end}`;
  } catch {
    return `${hit.sourcePath}:${hit.matchedStart ?? hit.sourceStart}-${hit.matchedEnd ?? hit.sourceEnd}`;
  }
}

function hitContainsAnchor(target: Target, hit: ZgHit, anchors: HardAnchor[]): boolean {
  if (!anchors.length) return true;
  const file = resolveHitFile(target, hit.sourcePath);
  if (!file) return false;
  let text: string;
  try { text = readFileSync(file, "utf8"); } catch { return false; }
  const lines = text.split(/\r?\n/);
  const start = Math.max(0, hit.sourceStart - 1);
  const end = Math.min(lines.length, Math.max(hit.sourceEnd, hit.matchedEnd ?? 0));
  const region = lines.slice(start, end).join("\n");
  const matched = anchors.filter((anchor) => anchorMatches(region, anchor));
  if (!matched.length) return false;
  hit.evidenceKind = "HYBRID_VERIFIED";
  hit.exactAnchors = matched.map((anchor) => anchor.raw);
  return true;
}

function anchorStatus(anchors: HardAnchor[], exactHits: ZgHit[], verifiedHits: ZgHit[]): string {
  if (!anchors.length) return "";
  const found = new Set<string>();
  for (const hit of [...exactHits, ...verifiedHits]) {
    for (const raw of hit.exactAnchors ?? []) found.add(raw);
  }
  return anchors.map((anchor) => `${JSON.stringify(anchor.raw)}=${found.has(anchor.raw) ? "found" : "not-found"}`).join(", ");
}

function hitLabel(target: Target, hit: ZgHit): string {
  const page = /(?:^|\/)pages\/(\d+)\.md$/.exec(hit.sourcePath)?.[1];
  if (page) return `${target.displayPath} (page ${Number(page)})`;
  return target.displayPath;
}

function buildEvidence(
  target: Target,
  hits: ZgHit[],
  maxChars: number,
  options: { anchors?: HardAnchor[]; exactHits?: number; semanticHits?: number } = {},
): string {
  const unique: ZgHit[] = [];
  const seen = new Set<string>();
  for (const hit of hits) {
    const key = evidenceKey(target, hit);
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(hit);
    if (unique.length >= MAX_EVIDENCE_HITS) break;
  }
  if (!unique.length) return "";

  const anchors = options.anchors ?? [];
  const route = anchors.length ? "exact+fused-hybrid+fts" : "fused-hybrid+fts";
  const anchorLine = anchors.length ? `\nanchors: ${anchorStatus(anchors, hits.filter((hit) => hit.evidenceKind === "EXACT" || hit.evidenceKind === "EXACT_SECTION"), hits.filter((hit) => hit.evidenceKind === "HYBRID_VERIFIED"))}` : "";
  const candidateLine = options.exactHits !== undefined
    ? `\ncandidates: exact=${options.exactHits} semantic=${options.semanticHits ?? 0}`
    : `\ncandidates: ${hits.length}`;
  const header = `Target: ${target.displayPath}\nroute: ${route}${anchorLine}${candidateLine}\n`;
  const remaining = Math.max(700, maxChars - header.length - 40);
  const perHit = Math.max(550, Math.floor(remaining / unique.length) - 120);
  const blocks: string[] = [];

  for (let i = 0; i < unique.length; i += 1) {
    const hit = unique[i];
    const file = resolveHitFile(target, hit.sourcePath);
    if (!file) continue;
    let text: string;
    try { text = readFileSync(file, "utf8"); } catch { continue; }
    const lines = text.split(/\r?\n/);
    const matchedStart = hit.matchedStart ?? hit.sourceStart;
    const section = structuralSection(lines, matchedStart);
    let excerpt = sliceAroundMatch(lines, hit, perHit);
    if (hit.evidenceKind === "EXACT" && section.structured) hit.evidenceKind = "EXACT_SECTION";
    const notes: string[] = [];
    for (const number of footnoteNumbers(excerpt)) {
      const note = findFootnoteContext(lines, number);
      if (note && !excerpt.includes(note)) notes.push(note);
    }
    if (notes.length) excerpt += `\n\nFootnote context:\n${notes.join("\n")}`;
    const evidenceKind = hit.evidenceKind ?? "HYBRID_DISCOVERY";
    const meta = [
      `#${i + 1}`,
      `evidence=${evidenceKind}`,
      hit.matchedBy ? `matchedBy=${hit.matchedBy}` : "",
      `source=${hitLabel(target, hit)}`,
      `lines=${hit.matchedStart ?? hit.sourceStart}-${hit.matchedEnd ?? hit.sourceEnd}`,
      hit.exactAnchors?.length ? `anchors=${hit.exactAnchors.map((value) => JSON.stringify(value)).join("+")}` : "",
    ].filter(Boolean).join(" ");
    blocks.push(`${meta}\n${excerpt.trim()}`);
  }

  const combined = `${header}\n${blocks.join("\n\n---\n\n")}`.trim();
  return capResult(combined, maxChars);
}

function capResult(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const suffix = "\n[results truncated]";
  if (maxChars <= suffix.length) return text.slice(0, maxChars);
  return text.slice(0, maxChars - suffix.length) + suffix;
}

function redactResult(text: string, target: Target): string {
  const slashRoot = target.projectRoot.replaceAll("\\", "/");
  const slashIndex = target.indexDir.replaceAll("\\", "/");
  return text
    .replaceAll(target.indexDir, "<project-index>")
    .replaceAll(slashIndex, "<project-index>")
    .replaceAll(target.projectRoot, "<project>")
    .replaceAll(slashRoot, "<project>");
}

export async function searchProjectPath(
  cwd: string,
  pathValue: string,
  query: string,
  config: ZvecContentConfig,
  signal?: AbortSignal,
): Promise<string> {
  const trimmed = query.trim();
  if (!trimmed) throw new Error("query must not be empty");
  const target = resolveTarget(cwd, pathValue);

  return withTargetLock(target.indexDir, signal, async () => {
    if (looksLikeRegex(trimmed)) {
      const stdout = await searchRegex(target, trimmed, config, signal);
      const body = redactResult(stdout.trim(), target);
      const combined = body
        ? `Target: ${target.displayPath}\nroute: regex\n${body}`
        : `Target: ${target.displayPath}\nroute: regex\nNo matching content.`;
      return capResult(combined, config.maxResultChars);
    }

    await ensureIndex(target, config, signal);
    const anchors = extractHardAnchors(trimmed);
    const exactHits = exactAnchorHits(target, anchors);
    const stdout = await runZg("query", [
      "query",
      "--debug",
      "--hybrid", trimmed,
      "--fts", trimmed,
      "--fuse",
      "--limit", String(config.limit),
      "--preview", "none",
      "--refresh", "wait",
      "--mode", "auto",
    ], target.indexDir, config, signal);

    const semanticHits = parseZgHits(stdout);
    for (const hit of semanticHits) hit.evidenceKind = "HYBRID_DISCOVERY";
    const verifiedSemanticHits = anchors.length
      ? semanticHits.filter((hit) => hitContainsAnchor(target, hit, anchors))
      : semanticHits;

    // Literal procedural anchors are constraints, not hints. Exact matches are
    // promoted ahead of semantic candidates; semantic candidates that omit all
    // requested hard anchors are excluded from model-visible evidence.
    const evidenceHits = anchors.length
      ? [...exactHits, ...verifiedSemanticHits]
      : verifiedSemanticHits;

    const evidence = buildEvidence(target, evidenceHits, config.maxResultChars, {
      anchors,
      exactHits: exactHits.length,
      semanticHits: semanticHits.length,
    });
    if (evidence) return evidence;

    if (anchors.length) {
      const status = anchorStatus(anchors, exactHits, verifiedSemanticHits);
      return capResult(
        `Target: ${target.displayPath}\nroute: exact+fused-hybrid+fts\nanchors: ${status}\nNo verified evidence contains the requested literal anchor(s).`,
        config.maxResultChars,
      );
    }

    // Defensive fallback for future zvec output-format changes: keep the tool
    // useful and bounded rather than returning an empty result.
    const body = redactResult(stdout.trim(), target);
    const combined = body
      ? `Target: ${target.displayPath}\nroute: fused-hybrid+fts\n${body}`
      : `Target: ${target.displayPath}\nroute: fused-hybrid+fts\nNo matching content.`;
    return capResult(combined, config.maxResultChars);
  });
}
