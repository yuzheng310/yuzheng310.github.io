import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { JSDOM } from "jsdom";
import { splitArticle, documentMeaning, applyDocument } from "../src/lib/author-document.mjs";

const dom = new JSDOM("<!doctype html><html><body></body></html>", { url: "http://localhost", pretendToBeVisual: true });
for (const name of ["window","document","navigator","Node","HTMLElement","SVGElement","Element","Text","DocumentFragment","DOMParser","MutationObserver","getComputedStyle","DOMRect","KeyboardEvent","MouseEvent","Event"]) Object.defineProperty(globalThis, name, { configurable: true, value: dom.window[name] });
globalThis.requestAnimationFrame = dom.window.requestAnimationFrame.bind(dom.window);
globalThis.cancelAnimationFrame = dom.window.cancelAnimationFrame.bind(dom.window);
globalThis.CustomEvent = dom.window.CustomEvent;
for (const name of ["addEventListener", "removeEventListener", "dispatchEvent"]) globalThis[name] = dom.window[name].bind(dom.window);
globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
globalThis.IntersectionObserver = class { observe() {} unobserve() {} disconnect() {} };
dom.window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
dom.window.Range.prototype.getClientRects = () => [];
dom.window.Range.prototype.getBoundingClientRect = () => new dom.window.DOMRect();
const { createArticleEditor } = await import("../src/editor/milkdown/engine.mjs");
const { TextSelection } = await import("@milkdown/kit/prose/state");
async function mount(markdown) {
  const root = document.createElement("div"); document.body.append(root);
  const editor = await createArticleEditor({ root, markdown });
  return { ...editor, destroy: async () => { await editor.destroy(); root.remove(); } };
}

for (const slug of readdirSync(new URL("../src/content/blog/", import.meta.url)).filter(name => name.endsWith(".md")).map(name => name.slice(0, -3))) test(`real article survives Milkdown import/export: ${slug}`, async () => {
  const source = readFileSync(new URL(`../src/content/blog/${slug}.md`, import.meta.url), "utf8");
  const { body } = splitArticle(source);
  const editor = await mount(body);
  try {
    assert.equal(documentMeaning(editor.getMarkdown()), documentMeaning(body));
    assert.equal(editor.changed(), false);
    assert.ok(applyDocument(source, editor.getMarkdown()).startsWith(splitArticle(source).header));
    // Exercise the post-transaction document too (trailing paragraphs, etc.).
    editor.action(view => { view.dispatch(view.state.tr.insertText("测试", 1)); });
    assert.doesNotThrow(() => applyDocument(source, editor.getMarkdown()));
    editor.undo();
  } finally { await editor.destroy(); }
});

test("five list items share one public note; edit, reload, delete and undo preserve content", async () => {
  const original = "- 切分了什么状态？\n- 保留了什么？\n- 传输了什么？\n- 频率是多少？\n- 改善了什么？\n\n公式 $x^2$。\n\n```py\nprint(1)\n```\n";
  const editor = await mount(original);
  try {
    editor.action(view => { const firstList = view.state.doc.firstChild; view.dispatch(view.state.tr.setSelection(TextSelection.create(view.state.doc, 3, firstList.nodeSize - 3))); });
    editor.highlight();
    assert.equal((editor.getMarkdown().match(/<mark /g) || []).length, 5);
    editor.undo(); assert.equal(documentMeaning(editor.getMarkdown()), documentMeaning(original));
    editor.saveNote("一个**公开批注**，包含 $a+b$。");
    let markdown = editor.getMarkdown();
    assert.equal((markdown.match(/data-author-note="author-1"/g) || []).length, 5);
    assert.equal(editor.notes().length, 1);
    assert.equal(editor.notes()[0].markdown, "一个**公开批注**，包含 $a+b$。");
    editor.saveNote("修改后的批注\n\n$$\na+b=c\n$$", "author-1");
    markdown = editor.getMarkdown();
    const reopened = await mount(markdown);
    try {
      assert.match(reopened.notes()[0].markdown, /修改后的批注/);
      reopened.deleteNote("author-1");
      assert.equal(reopened.notes().length, 0);
      assert.equal(documentMeaning(reopened.getMarkdown()), documentMeaning(original));
      reopened.undo(); assert.equal(reopened.notes().length, 1);
    } finally { await reopened.destroy(); }
  } finally { await editor.destroy(); }
});

test("whole-document writes preserve metadata and reject executable URLs, new HTML and broken notes", () => {
  const source = "---\ntitle: 原标题\n---\n原文";
  assert.match(applyDocument(source, "新正文", "导语"), /title: 原标题\nintro: "导语"/);
  for (const body of ["<script>alert(1)</script>", "[链接](javascript:alert%281%29)", "![图](blob:temporary)", "<mark class=\"author-highlight\" data-author-note=\"author-1\">正文</mark>", ""]) assert.throws(() => applyDocument(source, body));
});

test("formatted real-world list selection publishes marks, math, images and a nearby annotation card", async () => {
  const source = readFileSync(new URL("../src/content/blog/llm-inference-parallelism.md", import.meta.url), "utf8");
  const editor = await mount(splitArticle(source).body);
  try {
    editor.action(view => {
      let range;
      view.state.doc.descendants((node, pos) => {
        if (!range && node.type.name === "bullet_list" && node.textContent.startsWith("切分了什么状态")) range = [pos + 3, pos + node.nodeSize - 3];
      });
      assert.ok(range);
      view.dispatch(view.state.tr.setSelection(TextSelection.create(view.state.doc, ...range)));
    });
    editor.saveNote("跨五项的**公开批注**。\n\n$$\na+b=c\n$$");
    const saved = applyDocument(source, editor.getMarkdown());
    const { createRequire } = await import("node:module");
    const require = createRequire(import.meta.resolve("astro"));
    const { createMarkdownProcessor } = await import(require.resolve("@astrojs/markdown-remark"));
    const { default: math } = await import("remark-math");
    const { default: katex } = await import("rehype-katex");
    const { default: notes } = await import("../src/plugins/rehype-author-notes.mjs");
    const { default: diagrams } = await import("../src/plugins/rehype-translated-diagrams.mjs");
    const renderer = await createMarkdownProcessor({ remarkPlugins: [math], rehypePlugins: [katex, diagrams, notes] });
    const html = (await renderer.render(splitArticle(saved).body)).code;
    assert.equal((html.match(/class="translated-diagram"/g) || []).length, 25);
    assert.match(html, /data-author-note-card="author-2"/);
    assert.match(html, /跨五项的/);
    assert.match(html, /katex-display/);
    assert.match(html, /tensor_parallel_output/);
    const reopened = await mount(splitArticle(saved).body);
    try { assert.equal(reopened.notes().length, 2); } finally { await reopened.destroy(); }
  } finally { await editor.destroy(); }
});
