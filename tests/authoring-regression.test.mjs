import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { applyEdits } from '../src/lib/authoring.mjs';
const require = createRequire(import.meta.resolve('astro'));
const { createMarkdownProcessor } = await import(require.resolve('@astrojs/markdown-remark'));
const processor = await createMarkdownProcessor();

test('one selection can replace text across bold and links without deleting unrelated prose', async () => {
  const source = '段落开头 **粗体内容**，以及[链接文字](/about/)和结尾。';
  const ranges = ['开头 ', '粗体内容', '，以及', '链接文字'].map(quote => ({ start: source.indexOf(quote), end: source.indexOf(quote) + quote.length, quote }));
  const start=ranges[0].start, end=ranges.at(-1).end;
  const edited = applyEdits(source,[{start,end,quote:source.slice(start,end),ranges,kind:'replace',value:'更新后的说明'}]);
  assert.equal((await processor.render(edited)).code, '<p>段落更新后的说明和结尾。</p>');
});

test('one highlight across formatted text preserves bold and link semantics', async () => {
  const source = '前文**重点**与[链接](/about/)后文。';
  const ranges = ['文','重点','与','链接'].map((quote,i) => ({ start: i===0?1:source.indexOf(quote), end:(i===0?1:source.indexOf(quote))+quote.length, quote }));
  const start=ranges[0].start,end=ranges.at(-1).end;
  const html = (await processor.render(applyEdits(source,[{start,end,quote:source.slice(start,end),ranges,kind:'highlight',value:''}]))).code;
  assert.match(html, /<strong><mark[^>]*>重点<\/mark><\/strong>/);
  assert.match(html, /<a href="\/about\/"><mark[^>]*>链接<\/mark><\/a>/);
});

test('author notes retain paragraphs, Markdown emphasis, tables and TeX delimiters', async () => {
  const {default: math} = await import('remark-math');
  const {default: katex} = await import('rehype-katex');
  const renderer = await createMarkdownProcessor({remarkPlugins:[math],rehypePlugins:[katex]});
  const value = '**要点**\n\n第二段。\n\n\\[ Y=\\sigma(XA)B \\]\n\n| GPU | 输出 |\n|---|---|\n| 0 | \\(P_0\\) |';
  const html=(await renderer.render(applyEdits('正文重点',[{start:2,end:4,quote:'重点',kind:'note',value}]))).code;
  assert.match(html, /<strong>要点<\/strong>/);
  assert.match(html, /<p>第二段。<\/p>/);
  assert.match(html, /class="katex-display"/);
  assert.match(html, /<table>/);
});

test('public author cards stay beside the passage, preserve old anchors and leave citations alone', async () => {
  const {default: notes} = await import('../src/plugins/rehype-author-notes.mjs');
  const renderer = await createMarkdownProcessor({rehypePlugins:[notes]});
  const source='原文重点。\n\n后文引用[^cite]。\n\n[^cite]: 文献';
  const updated=applyEdits(source,[{start:2,end:4,quote:'重点',kind:'note',value:'**解释**\n\n第二段'}]);
  const html=(await renderer.render(updated)).code;
  assert.equal((html.match(/id="user-content-fn-author-1"/g)||[]).length,1);
  assert.ok(html.indexOf('<details')<html.indexOf('后文引用'));
  assert.match(html, /<strong>解释<\/strong>/);
  assert.match(html, /<p>第二段/);
  assert.match(html, /id="user-content-fn-cite"/);
  assert.ok(!html.includes('href="#user-content-fnref-author-1"'));
});

test('unsafe note links fail closed, literal code stays literal and HTML is escaped', async () => {
  const {normalizeNote}=await import('../src/lib/author-markdown.mjs');
  assert.throws(()=>normalizeNote('[危险](javascript:alert%281%29)'));
  assert.throws(()=>normalizeNote('[危险](jav&#x61;script:alert%281%29)'));
  assert.equal(normalizeNote('`\\[x\\]`'), '`\\[x\\]`');
  assert.equal(normalizeNote('<script>alert(1)</script>'), '&lt;script&gt;alert(1)&lt;/script&gt;');
});

test('grouped selections reject missing fragments and crossing paragraph boundaries', () => {
  const source='开始**中间**结束\n\n另一段';
  const edit={start:0,end:10,quote:source.slice(0,10),kind:'replace',value:'新文',ranges:[{start:0,end:2,quote:'开始'}]};
  assert.throws(()=>applyEdits(source,[edit]));
  assert.throws(()=>applyEdits(source,[{...edit,end:source.length,quote:source,ranges:[{start:0,end:source.length,quote:source}]}]));
});
