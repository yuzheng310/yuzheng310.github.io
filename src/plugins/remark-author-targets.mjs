import { readFileSync } from 'node:fs';

// Map editable prose directly to source offsets; never round-trip the document
// through a rich-text serializer. Escaped text and generated math stay protected.
export default function remarkAuthorTargets() {
  return (tree, file) => {
    const path = String(file.path || '');
    if (!/[/\\]content[/\\]blog[/\\][\w-]+\.md$/.test(path)) return;
    const source = readFileSync(path, 'utf8');
    const input = String(file.value);
    const delta = source.indexOf(input);
    if (delta < 0) return;
    function visit(node, protectedText = false) {
      const protect = protectedText || ['heading', 'link', 'image', 'code', 'inlineCode', 'math', 'inlineMath', 'footnoteDefinition'].includes(node.type);
      if (!node.children) return;
      node.children = node.children.map(child => {
        if (!protect && child.type === 'text' && child.value.trim() && child.position) {
          const start = child.position.start.offset + delta;
          const end = child.position.end.offset + delta;
          if (source.slice(start, end) === child.value) {
            return { type: 'authorText', data: { hName: 'span', hProperties: { 'data-author-start': start, 'data-author-end': end }, hChildren: [{ type: 'text', value: child.value }] } };
          }
        }
        visit(child, protect);
        return child;
      });
    }
    visit(tree);
  };
}
