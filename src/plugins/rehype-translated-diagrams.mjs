import { readFileSync } from 'node:fs';

const translations = JSON.parse(readFileSync(new URL('../data/translated-diagrams.json', import.meta.url), 'utf8'));
const element = (tagName, properties, children = []) => ({ type: 'element', tagName, properties, children });
const text = value => ({ type: 'text', value });

// Keep Markdown and author-edit source coordinates intact. Translate only
// standalone images with an explicitly registered, reviewed counterpart.
export default function rehypeTranslatedDiagrams() {
  return tree => {
    let count = 0;
    function visit(node) {
      if (!node.children) return;
      node.children = node.children.map(child => {
        const image = child.tagName === 'p' && child.children?.length === 1 ? child.children[0] : null;
        const original = image?.tagName === 'img' ? image.properties.src : null;
        const translated = translations[original];
        if (!translated) { visit(child); return child; }
        const id = `translated-diagram-${++count}`;
        const alt = image.properties.alt || '文章配图';
        Object.assign(image.properties, {
          id, src: translated.src, width: translated.width, height: translated.height,
          alt: `${alt}（中文版）`, loading: 'lazy', decoding: 'async',
        });
        const buttons = ['中文', '原图'].map((label, i) => element('button', {
          type: 'button', 'data-diagram-version': i ? 'original' : 'zh',
          'aria-pressed': String(!i), 'aria-controls': id,
        }, [text(label)]));
        return element('figure', {
          className: ['translated-diagram'], 'data-zh': translated.src,
          'data-original': original, 'data-alt': alt,
        }, [
          element('div', { className: ['diagram-toolbar'], 'data-pagefind-ignore': true }, [
            element('div', { className: ['diagram-switch'], role: 'group', 'aria-label': `${alt}：图片语言`, hidden: true }, buttons),
            element('a', { href: original, target: '_blank', rel: 'noopener', className: ['diagram-original-link'] }, [text('查看英文原图 ↗')]),
          ]),
          element('a', {
            className: ['diagram-stage'], href: translated.src, target: '_blank', rel: 'noopener',
            'aria-label': `放大：${alt}（中文版）`, style: `aspect-ratio: ${translated.width} / ${translated.height}`,
          }, [image]),
          element('div', { className: ['diagram-feedback'], role: 'status', 'aria-live': 'polite', 'data-pagefind-ignore': true }),
        ]);
      });
    }
    visit(tree);
  };
}
