import { parseMarkdown, walk } from "./author-markdown.mjs";
import { applyEdits } from "./authoring.mjs";

export function splitArticle(source) {
  const header = source.match(/^---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/)?.[0];
  if (!header) throw new Error("文章缺少有效元信息。");
  return { header, body: source.slice(header.length) };
}

// A public annotation is content, not a remote collaboration thread. Each
// highlighted inline run may share an ID; a single footnote owns its Markdown.
export function transformHighlights(tree) {
  function visit(parent) {
    if (!parent.children) return;
    const result = [], stack = [result];
    for (const child of parent.children) {
      const open = child.type === "html" && child.value.match(/^<mark class="author-highlight"(?: data-author-note="(author-\d+)")?>$/);
      if (open) {
        const mark = { type: "authorHighlight", note: open[1] || "", children: [] };
        stack.at(-1).push(mark); stack.push(mark.children);
      } else if (child.type === "html" && child.value === "</mark>" && stack.length > 1) stack.pop();
      else { visit(child); stack.at(-1).push(child); }
    }
    if (stack.length !== 1) throw new Error("高亮标签不完整，无法安全编辑。");
    parent.children = result;
  }
  visit(tree);
  // Repeating the same emphasis mark has no extra visual meaning. Flatten it
  // before ProseMirror parsing so closing the inner mark cannot close the outer.
  function normalize(parent, active = new Set()) {
    if (!parent.children) return;
    const result = [];
    for (const child of parent.children) {
      const format = ["strong", "emphasis", "delete"].includes(child.type);
      normalize(child, format ? new Set([...active, child.type]) : active);
      for (const item of format && active.has(child.type) ? child.children : [child]) {
        if (item.type === "text") item.value = item.value.replace(/\r?\n/g, " ");
        if (item.type === "text" && result.at(-1)?.type === "text") result.at(-1).value += item.value;
        else result.push(item);
      }
    }
    parent.children = result;
  }
  normalize(tree);
  return tree;
}

export function remarkPublicAnnotations() {
  const data = this.data();
  const extensions = data.toMarkdownExtensions || (data.toMarkdownExtensions = []);
  extensions.push({ handlers: {
    authorHighlight(node, _parent, state, info) {
      const note = /^author-\d+$/.test(node.note || "") ? ` data-author-note="${node.note}"` : "";
      return `<mark class="author-highlight"${note}>${state.containerPhrasing(node, info)}</mark>`;
    },
  } });
  return transformHighlights;
}

// Ignore syntax choices (bullet marker, fence width, heading style), but not
// document content. Loading must pass this before the editor can replace prose.
export function documentMeaning(markdown) {
  const tree = transformHighlights(parseMarkdown(markdown));
  function clean(node) {
    const result = {};
    for (const [key, value] of Object.entries(node)) {
      if (["position", "data", "spread", "label"].includes(key) || value == null || value === "") continue;
      result[key] = key === "children" ? value.map(clean) : value;
    }
    return result;
  }
  return JSON.stringify(clean(tree));
}

export function assertRoundTrip(before, after) {
  const a = documentMeaning(before), b = documentMeaning(after);
  if (a !== b) {
    let offset = 0; while (a[offset] === b[offset]) offset++;
    throw new Error("这篇文章包含尚未完整兼容的格式，已保留原文并停止进入编辑。", { cause: { offset, before: a.slice(Math.max(0, offset-80), offset+160), after: b.slice(Math.max(0, offset-80), offset+160) } });
  }
}

// The server independently validates the complete document. Metadata remains
// server-owned, except for the existing narrowly scoped intro operation.
export function applyDocument(source, body, intro) {
  if (typeof body !== "string" || !body.trim() || body.length > 1000000) throw new Error("正文为空或超过大小限制。");
  const original = splitArticle(source);
  const allowedHtml = new Map();
  walk(parseMarkdown(original.body), node => {
    if (node.type === "html") allowedHtml.set(node.value, (allowedHtml.get(node.value) || 0) + 1);
  });
  const definitions = new Set(), references = [], noteMarks = [];
  const tree = parseMarkdown(body);
  walk(tree, node => {
    if (node.type === "html" && !/^<mark class="author-highlight"(?: data-author-note="author-\d+")?>$|^<\/mark>$|^<br\s*\/?\s*>$/i.test(node.value)) {
      const remaining = allowedHtml.get(node.value) || 0;
      if (!remaining) throw new Error("正文包含不受支持的新 HTML，请使用编辑器的格式工具。");
      allowedHtml.set(node.value, remaining - 1);
    }
    if (["link", "image", "definition"].includes(node.type)) {
      // URL scheme checks intentionally strip control characters before validation.
      // eslint-disable-next-line no-control-regex
      const url = node.url.replace(/[\s\u0000-\u001f\u007f]/g, "");
      if (/^[a-z][a-z\d+.-]*:/i.test(url) && !/^(https?:|mailto:)/i.test(url)) throw new Error("链接仅支持网页、邮箱或站内地址，图片请先保存为网站资源。");
    }
    if (node.type === "footnoteDefinition") {
      if (definitions.has(node.identifier)) throw new Error("存在重复的批注编号。");
      definitions.add(node.identifier);
    }
    if (node.type === "footnoteReference") references.push(node.identifier);
    if (node.type === "html") {
      const id = node.value.match(/data-author-note="(author-\d+)"/)?.[1];
      if (id) noteMarks.push(id);
    }
  });
  transformHighlights(tree);
  for (const id of [...references, ...noteMarks]) if (!definitions.has(id)) throw new Error("批注引用缺少正文，请恢复批注或删除对应引用。");
  // Do not allow a body payload to become frontmatter when composed.
  const next = original.header + "\n" + body.trim() + "\n";
  return intro === undefined ? next : applyEdits(next, [], intro);
}
