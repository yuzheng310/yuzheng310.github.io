import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

const parser = unified().use(remarkParse).use(remarkGfm).use(remarkMath);
export const parseMarkdown = source => parser.parse(source);
export function walk(node, visit, parents = []) {
  visit(node, parents);
  for (const child of node.children || []) walk(child, visit, [...parents, node]);
}

// Accept both common TeX delimiter styles, while leaving literal code untouched.
export function normalizeNote(value) {
  let source = value.replace(/\r\n?/g, '\n').trim();
  const protectedRanges = [];
  walk(parseMarkdown(source), node => {
    if (['code', 'inlineCode', 'math', 'inlineMath'].includes(node.type)) protectedRanges.push([node.position.start.offset, node.position.end.offset]);
  });
  const convert = text => text.replace(/\\\[([\s\S]*?)\\\]/g, (_, body) => `\n\n$$\n${body.trim()}\n$$\n\n`).replace(/\\\(([^\n]*?)\\\)/g, (_, body) => `$${body.trim()}$`);
  let offset = 0, result = '';
  for (const [start, end] of protectedRanges.sort((a,b) => a[0]-b[0])) {
    result += convert(source.slice(offset, start)) + source.slice(start, end); offset = end;
  }
  source = (result + convert(source.slice(offset))).trim();
  const escapes = [];
  walk(parseMarkdown(source), node => {
    if (node.type === 'html') escapes.push({ start: node.position.start.offset, end: node.position.end.offset, value: node.value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;') });
    if (['link', 'image', 'definition'].includes(node.type)) {
      const url = node.url.replace(/[\s\u0000-\u001f]/g, '');
      if (/^[a-z][a-z\d+.-]*:/i.test(url) && !/^(https?:|mailto:)/i.test(url)) throw new Error('批注中的链接仅支持网页地址或邮箱。');
    }
  });
  for (const patch of escapes.sort((a,b) => b.start-a.start)) source=source.slice(0,patch.start)+patch.value+source.slice(patch.end);
  return source;
}

// Group all pieces of one visual paragraph, including emphasis and link labels.
// Validate against the parsed document, never trust client-provided source offsets.
export function patchParagraph(source, edit, replacements) {
  const tree = parseMarkdown(source);
  let paragraph;
  walk(tree, (node, parents) => {
    if (node.type === 'paragraph' && !parents.some(p => p.type === 'footnoteDefinition') && node.position.start.offset <= edit.start && node.position.end.offset >= edit.end) paragraph = node;
  });
  if (!paragraph) throw new Error('请在同一段落内选择文字。');
  const leaves = [];
  walk(paragraph, node => { if (['text','inlineCode','inlineMath'].includes(node.type)) leaves.push(node); });
  const ranges = edit.ranges;
  const touched = leaves.filter(node => node.position.start.offset < edit.end && node.position.end.offset > edit.start);
  if (touched.length !== ranges.length) throw new Error('选区包含无法识别的内容，请重新选择。');
  for (let i = 0; i < ranges.length; i++) {
    const node = touched[i], range = ranges[i];
    const begin = node.position.start.offset, end = node.position.end.offset;
    if (range.start < begin || range.end > end || (node.type !== 'text' && (range.start !== begin || range.end !== end))) throw new Error('公式或行内代码需要完整选中。');
    if ((i > 0 && range.start !== begin) || (i < ranges.length - 1 && range.end !== end)) throw new Error('选区不是连续的文字。');
  }
  function render(node) {
    const begin = node.position.start.offset, end = node.position.end.offset;
    if (!node.children) {
      let raw = source.slice(begin,end);
      for (let i=ranges.length-1;i>=0;i--) {
        const range=ranges[i];
        if (range.start>=begin && range.end<=end) raw=raw.slice(0,range.start-begin)+replacements[i]+raw.slice(range.end-begin);
      }
      return raw;
    }
    let cursor = begin, raw = '', content = '';
    for (const child of node.children) {
      const rendered = render(child);
      raw += source.slice(cursor,child.position.start.offset)+rendered;
      content += rendered; cursor=child.position.end.offset;
    }
    if (!content && ['strong','emphasis','delete','link','linkReference'].includes(node.type)) return '';
    return raw+source.slice(cursor,end);
  }
  return { start: paragraph.position.start.offset, end: paragraph.position.end.offset, value: render(paragraph) };
}
