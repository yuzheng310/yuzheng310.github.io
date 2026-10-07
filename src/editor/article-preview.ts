import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeRaw from "rehype-raw";
import rehypeKatex from "rehype-katex";
import rehypeStringify from "rehype-stringify";
import rehypeAuthorNotes from "../plugins/rehype-author-notes.mjs";
import { articleImage } from "./milkdown/surface";
// Call only after applyDocument validates the body against the fetched source.
const renderer = unified().use(remarkParse).use(remarkGfm).use(remarkMath)
  .use(remarkRehype, { allowDangerousHtml: true }).use(rehypeRaw)
  .use(rehypeKatex).use(rehypeAuthorNotes).use(rehypeStringify);
export async function previewArticle(container: HTMLElement, markdown: string) {
  container.innerHTML = String(await renderer.process(markdown));
  for (const img of container.querySelectorAll<HTMLImageElement>("img")) {
    img.replaceWith(articleImage({ attrs: { src: img.getAttribute("src") || "", alt: img.alt, title: img.title } }).dom);
  }
  document.dispatchEvent(new Event("article-font-change"));
}
