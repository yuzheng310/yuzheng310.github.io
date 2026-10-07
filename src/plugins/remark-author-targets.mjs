import { readFileSync } from 'node:fs';

// Source coordinates belong to a paragraph, not to individual formatting runs.
// Leaves still carry exact coordinates so edits preserve surrounding Markdown.
export default function remarkAuthorTargets() {
  return (tree, file) => {
    const path = String(file.path || '');
    if (!/[/\\]content[/\\]blog[/\\][\w-]+\.md$/.test(path)) return;
    const source = readFileSync(path, 'utf8');
    const delta = source.indexOf(String(file.value));
    if (delta < 0) return;
    function visit(node, protectedText = false, inParagraph = false) {
      const protect = protectedText || ['heading','image','code','math','footnoteDefinition'].includes(node.type);
      if (!node.children) return;
      const paragraph = inParagraph || node.type === 'paragraph';
      if (['paragraph','listItem'].includes(node.type) && !protect) {
        node.data ||= {}; node.data.hProperties ||= {};
        node.data.hProperties['data-author-block'] = node.position.start.offset + delta;
      }
      node.children = node.children.map(child => {
        if (!protect && paragraph && ['text','inlineCode','inlineMath'].includes(child.type) && child.position) {
          const start = child.position.start.offset + delta, end = child.position.end.offset + delta;
          const atomic = child.type !== 'text' || source.slice(start,end) !== child.value;
          const properties = { 'data-author-start':start, 'data-author-end':end, ...(atomic ? {'data-author-atomic':'true','data-author-label':child.value} : {}) };
          if (child.type === 'text') return {type:'authorText', data:{hName:'span',hProperties:properties,hChildren:[{type:'text',value:child.value}]}};
          return {type:'authorAtomic',children:[child],data:{hName:'span',hProperties:properties}};
        }
        visit(child, protect, paragraph); return child;
      });
    }
    visit(tree);
  };
}
