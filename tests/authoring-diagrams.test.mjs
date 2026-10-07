import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import diagrams from '../src/plugins/rehype-translated-diagrams.mjs';
import dimensions from '../src/plugins/rehype-image-dimensions.mjs';
const require = createRequire(import.meta.resolve('astro'));
const { createMarkdownProcessor } = await import(require.resolve('@astrojs/markdown-remark'));

test('translated figures render by default with original fallback, stable sizing and unique controls', async () => {
  const processor = await createMarkdownProcessor({ rehypePlugins: [diagrams, dimensions] });
  const source = readFileSync('src/content/blog/llm-inference-parallelism.md', 'utf8');
  const html = (await processor.render(source)).code;
  assert.equal((html.match(/class="translated-diagram"/g) || []).length, 25);
  const ids = [...html.matchAll(/id="(translated-diagram-\d+)"/g)].map(m => m[1]);
  assert.equal(new Set(ids).size, 25);
  assert.equal((html.match(/aria-controls="translated-diagram-/g) || []).length, 50);
  assert.match(html, /src="\/translations\/images\/14c97c813ebe2c6b.jpg"/);
  assert.match(html, /查看英文原图/);
  assert.match(html, /style="aspect-ratio:/);
  for (const [, uri] of html.matchAll(/(?:data-original|data-zh|src)="(\/translations\/[^" ]+)"/g)) {
    assert.ok(existsSync(`public${uri}`), uri);
  }
});

test('unregistered images and surrounding author edit coordinates remain unchanged', () => {
  const img = { type: 'element', tagName: 'img', properties: { src: '/other.png', alt: '其他文章' } };
  const paragraph = { type: 'element', tagName: 'p', properties: { 'data-author-block': 42 }, children: [{ type: 'text', value: '正文' }] };
  const tree = { children: [{ type: 'element', tagName: 'p', children: [img] }, paragraph] };
  const before = structuredClone(tree);
  diagrams()(tree);
  assert.deepEqual(tree, before);
});
