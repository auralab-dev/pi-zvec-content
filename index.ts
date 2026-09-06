import type { AgentToolResult, ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import {
  loadZvecContentConfig,
  searchProjectPath,
} from "./zvec.ts";

export default function (pi: ExtensionAPI) {
  const config = loadZvecContentConfig();

  pi.registerTool({
    name: "file_content_search",
    label: "File Content Search",
    description: "Search contents of a project-relative file or directory and return bounded excerpts. Use paths from download_file or find. Use words, phrases, or natural-language concepts; intentional regex-like queries are handled as regex. Use read instead when the full file or a specific line range is needed.",
    promptSnippet: "Use file_content_search for targeted excerpts and bounded semantic search; use read for direct workspace file reads.",
    executionMode: "sequential",
    parameters: Type.Object({
      path: Type.String({ description: "Project-relative file or directory path from download_file or find. Absolute paths are rejected." }),
      query: Type.String({ description: "Words, phrase, concept, or intentional regex to search for." }),
    }),

    async execute(_toolCallId, rawParams, signal, _onUpdate, ctx): Promise<AgentToolResult<Record<string, unknown>>> {
      const path = typeof rawParams.path === "string" ? rawParams.path.trim() : "";
      const query = typeof rawParams.query === "string" ? rawParams.query.trim() : "";
      if (!path) throw new Error("file_content_search requires a project-relative path");
      if (!query) throw new Error("file_content_search requires a query");

      const text = await searchProjectPath(ctx.cwd, path, query, config, signal);
      return { content: [{ type: "text", text }], details: { path } };
    },
  });
}
