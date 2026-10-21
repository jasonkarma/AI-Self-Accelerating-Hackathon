import { Fragment, useMemo } from "react";
import type { ReactNode } from "react";

/**
 * A deliberately small Markdown renderer.
 *
 * The council memos are model output, and model output is where a full Markdown
 * library earns its keep — but the subset that actually appears in these seven
 * memos is narrow and fixed: `##`/`###` headings, pipe tables, `-` bullet lists,
 * `1.` ordered lists, fenced blocks, `---` rules, blockquotes, `**bold**`,
 * `` `code` `` and plain paragraphs. Rendering exactly that set (and escaping
 * everything else) keeps the bundle honest: no `dangerouslySetInnerHTML`, no
 * sanitiser, no HTML injection surface from a payload.
 *
 * The alternative — piping through `marked` and re-sanitising — was rejected
 * because a memo is *untrusted text from a model*, and a renderer that can only
 * emit the elements it knows is a stronger guarantee than one that sanitises a
 * broader grammar.
 */

type Block =
  | { kind: "heading"; level: 2 | 3 | 4; text: string }
  | { kind: "paragraph"; text: string }
  | { kind: "list"; ordered: boolean; items: string[] }
  | { kind: "table"; head: string[]; rows: string[][] }
  | { kind: "code"; text: string }
  | { kind: "rule" }
  | { kind: "quote"; text: string };

function splitRow(line: string): string[] {
  return line
    .replace(/^\s*\|/, "")
    .replace(/\|\s*$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isDivider(line: string): boolean {
  return /^\s*\|?(\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$/.test(line);
}

export function parseMarkdown(source: string): Block[] {
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index] ?? "";

    if (line.trim() === "") {
      index += 1;
      continue;
    }

    if (/^\s*---+\s*$/.test(line)) {
      blocks.push({ kind: "rule" });
      index += 1;
      continue;
    }

    const fence = /^\s*```/.exec(line);
    if (fence) {
      const body: string[] = [];
      index += 1;
      while (index < lines.length && !/^\s*```/.test(lines[index] ?? "")) {
        body.push(lines[index] ?? "");
        index += 1;
      }
      index += 1; // closing fence
      blocks.push({ kind: "code", text: body.join("\n") });
      continue;
    }

    const heading = /^(#{2,4})\s+(.*)$/.exec(line);
    if (heading) {
      const level = Math.min(4, Math.max(2, heading[1]!.length)) as 2 | 3 | 4;
      blocks.push({ kind: "heading", level, text: heading[2]!.trim() });
      index += 1;
      continue;
    }

    // A table starts with a pipe row immediately followed by a divider row.
    if (line.includes("|") && isDivider(lines[index + 1] ?? "")) {
      const head = splitRow(line);
      index += 2;
      const rows: string[][] = [];
      while (index < lines.length && (lines[index] ?? "").includes("|") && (lines[index] ?? "").trim() !== "") {
        const cells = splitRow(lines[index] ?? "");
        // Normalise ragged rows so the table always has `head.length` columns.
        while (cells.length < head.length) cells.push("");
        rows.push(cells.slice(0, head.length));
        index += 1;
      }
      blocks.push({ kind: "table", head, rows });
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index] ?? "")) {
        items.push((lines[index] ?? "").replace(/^\s*[-*]\s+/, ""));
        index += 1;
      }
      blocks.push({ kind: "list", ordered: false, items });
      continue;
    }

    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*\d+[.)]\s+/.test(lines[index] ?? "")) {
        items.push((lines[index] ?? "").replace(/^\s*\d+[.)]\s+/, ""));
        index += 1;
      }
      blocks.push({ kind: "list", ordered: true, items });
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const parts: string[] = [];
      while (index < lines.length && /^\s*>\s?/.test(lines[index] ?? "")) {
        parts.push((lines[index] ?? "").replace(/^\s*>\s?/, ""));
        index += 1;
      }
      blocks.push({ kind: "quote", text: parts.join(" ") });
      continue;
    }

    // Paragraph: consume until a blank line or a line that starts another block.
    const paragraph: string[] = [line];
    index += 1;
    while (index < lines.length) {
      const next = lines[index] ?? "";
      if (
        next.trim() === "" ||
        /^\s*(#{2,4}|```|>|---+)\s*/.test(next) ||
        /^\s*[-*]\s+/.test(next) ||
        /^\s*\d+[.)]\s+/.test(next) ||
        (next.includes("|") && isDivider(lines[index + 1] ?? ""))
      ) {
        break;
      }
      paragraph.push(next);
      index += 1;
    }
    blocks.push({ kind: "paragraph", text: paragraph.join(" ") });
  }

  return blocks;
}

/** Inline spans: `**bold**`, `` `code` ``, and `*emphasis*`. */
export function renderInline(text: string): ReactNode {
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*)/g).filter(Boolean);
  return tokens.map((token, index) => {
    if (token.startsWith("**") && token.endsWith("**")) {
      return (
        <strong key={index} className="font-semibold ink">
          {token.slice(2, -2)}
        </strong>
      );
    }
    if (token.startsWith("`") && token.endsWith("`")) {
      return (
        <code
          key={index}
          className="mono rounded px-1 py-[1px] text-[11px]"
          style={{ background: "var(--panel-2)", color: "var(--ink)" }}
        >
          {token.slice(1, -1)}
        </code>
      );
    }
    if (token.startsWith("*") && token.endsWith("*") && token.length > 2) {
      return (
        <em key={index} className="italic">
          {token.slice(1, -1)}
        </em>
      );
    }
    return <Fragment key={index}>{token}</Fragment>;
  });
}

/**
 * Renders a council memo.
 *
 * `dense` is used for the thinking trace and the raw-ledger view: same grammar,
 * smaller type, no tables spanning the full width. The block spacing is tight
 * because a memo is evidence a reader scans, not an article they read linearly.
 */
export function Markdown({ source, dense = false }: { source: string; dense?: boolean }) {
  const blocks = useMemo(() => parseMarkdown(source), [source]);
  const text = dense ? "text-[11.5px] leading-relaxed" : "text-[12.5px] leading-relaxed";

  return (
    <div className={`space-y-1.5 ${dense ? "ink-dim" : "ink"}`}>
      {blocks.map((block, index) => {
        switch (block.kind) {
          case "heading":
            return (
              <h4
                key={index}
                className={`${block.level === 2 ? "text-[12.5px]" : "text-[12px]"} font-semibold text-[color:var(--ink)] ${
                  index === 0 ? "" : "pt-1.5"
                }`}
              >
                {renderInline(block.text)}
              </h4>
            );
          case "paragraph":
            return (
              <p key={index} className={text}>
                {renderInline(block.text)}
              </p>
            );
          case "list":
            return block.ordered ? (
              <ol key={index} className={`ml-4 list-decimal space-y-0.5 ${text}`}>
                {block.items.map((item, itemIndex) => (
                  <li key={itemIndex}>{renderInline(item)}</li>
                ))}
              </ol>
            ) : (
              <ul key={index} className={`ml-4 list-disc space-y-0.5 ${text}`}>
                {block.items.map((item, itemIndex) => (
                  <li key={itemIndex}>{renderInline(item)}</li>
                ))}
              </ul>
            );
          case "table":
            return (
              <div key={index} className="scroll-thin my-1 overflow-x-auto rounded-md border" style={{ borderColor: "var(--border)" }}>
                <table className="w-full border-collapse text-[11px]">
                  <thead>
                    <tr style={{ background: "var(--panel-2)" }}>
                      {block.head.map((cell, cellIndex) => (
                        <th
                          key={cellIndex}
                          className="whitespace-nowrap px-2 py-1 text-left font-semibold"
                          style={{ color: "var(--ink-dim)" }}
                        >
                          {renderInline(cell)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, rowIndex) => (
                      <tr key={rowIndex} style={{ borderTop: "1px solid var(--border)" }}>
                        {row.map((cell, cellIndex) => (
                          <td key={cellIndex} className="px-2 py-0.5 align-top">
                            {renderInline(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          case "code":
            return (
              <pre
                key={index}
                className="mono scroll-thin overflow-x-auto rounded-md px-2.5 py-1.5 text-[10.5px] leading-relaxed"
                style={{ background: "var(--panel-2)", color: "var(--ink-dim)", border: "1px solid var(--border)" }}
              >
                {block.text}
              </pre>
            );
          case "quote":
            return (
              <blockquote
                key={index}
                className="border-l-2 pl-2.5 text-[11.5px] italic ink-dim"
                style={{ borderColor: "#0EA5E9" }}
              >
                {renderInline(block.text)}
              </blockquote>
            );
          case "rule":
            return <hr key={index} style={{ borderColor: "var(--border)" }} />;
          default:
            return null;
        }
      })}
    </div>
  );
}
