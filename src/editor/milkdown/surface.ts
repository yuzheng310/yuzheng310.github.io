import "@milkdown/crepe/theme/common/style.css";
import "../../styles/inline-editor.css";
import { createArticleEditor } from "./engine.mjs";
import translations from "../../data/translated-diagrams.json";

export function articleImage(node: { attrs: { src: string; alt: string; title: string } }) {
  const { src, alt, title } = node.attrs;
  const translated = (translations as Record<string, { src: string; width: number; height: number }>)[src];
  const dom = document.createElement("span");
  dom.className = "inline-article-image"; dom.contentEditable = "false";
  const img = document.createElement("img");
  img.src = translated?.src || src; img.alt = alt; img.title = title;
  img.loading = "lazy"; img.decoding = "async";
  if (translated) {
    img.width = translated.width; img.height = translated.height;
    const controls = document.createElement("span"); controls.className = "inline-image-switch";
    controls.setAttribute("role", "group"); controls.setAttribute("aria-label", "图片语言");
    for (const [label, url] of [["中文", translated.src], ["原图", src]]) {
      const button = document.createElement("button"); button.type = "button"; button.textContent = label;
      button.setAttribute("aria-pressed", String(url === translated.src));
      button.addEventListener("click", () => {
        img.src = url;
        for (const item of controls.querySelectorAll("button")) item.setAttribute("aria-pressed", String(item === button));
      });
      controls.append(button);
    }
    dom.append(controls);
  }
  dom.append(img);
  return { dom, stopEvent: () => true, ignoreMutation: () => true };
}

export async function mountArticleEditor(root: HTMLElement, markdown: string, onChange: () => void, onNote: () => void) {
  const htmlView = (node: { attrs: { value: string } }) => {
    const raw = node.attrs.value;
    const heading = raw.match(/^<h([1-6]) id="([\w.-]+)">([^<]*)<\/h\1>$/);
    const dom = document.createElement(heading ? `h${heading[1]}` : "span");
    dom.contentEditable = "false";
    if (heading) {
      dom.textContent = new DOMParser().parseFromString(heading[3], "text/html").body.textContent;
      dom.title = "原标题按原样保留";
    } else if (/^<\/?(?:span|a)(?:\s[^<>]*)?>$/.test(raw)) {
      // Existing source anchors stay in Markdown, not as visible HTML tokens.
      dom.setAttribute("aria-hidden", "true");
    } else if (/^<br\s*\/?\s*>$/i.test(raw)) dom.append(document.createElement("br"));
    else { dom.textContent = raw; dom.className = "author-preserved-html"; }
    return { dom, ignoreMutation: () => true };
  };
  const editor = await createArticleEditor({ root, markdown, onChange, onNote, imageView: articleImage, htmlView });
  const prose = root.querySelector<HTMLElement>(".ProseMirror")!;
  prose.classList.add("prose");
  prose.setAttribute("aria-label", "文章正文编辑器");
  prose.setAttribute("role", "textbox");
  prose.setAttribute("aria-multiline", "true");
  return editor;
}
