import type { CollectionEntry } from 'astro:content';

const documents = import.meta.glob('../documents/translations/*.html', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;
const escape = (value: string) => value.replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]!);

/** Preserve standalone document styles and interactions while sharing site navigation. */
export function renderTranslation(item: CollectionEntry<'blog'>, site: URL | undefined): string {
  const { data } = item;
  const document = documents[`../documents/translations/${data.htmlFile}`];
  if (!document) throw new Error(`Missing translation document: ${data.htmlFile}`);
  const title = escape(`${data.title} · yuzheng310`);
  const canonical = new URL(`/blog/${item.slug}/`, site).href;
  const metadata = `<title>${title}</title>
<meta name="description" content="${escape(data.description)}">
<link rel="canonical" href="${canonical}">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="alternate" type="application/rss+xml" title="yuzheng310 · 项目与笔记" href="/rss.xml">
<meta property="og:title" content="${title}"><meta property="og:description" content="${escape(data.description)}">
<meta property="og:url" content="${canonical}"><meta property="og:type" content="article">
<link rel="stylesheet" href="/translations/site-reader.css">`;
  const tags = data.tags.map(tag => `<a href="/blog/tags/${encodeURIComponent(tag)}/">${escape(tag)}</a>`).join('');
  const sourceLabel = item.slug === 'llm-inference-explained' ? '原作者主页 ↗' : '阅读原文 ↗';
  const navigation = `<div id="site-reader-bar"><nav aria-label="网站导航"><a href="/">yuzheng310</a><a href="/blog/">← 全部笔记</a><a href="${escape(data.sourceURL || '')}" rel="noreferrer">${sourceLabel}</a></nav><div class="reader-tags" aria-label="文章标签">${tags}</div><p>原作者：${escape(data.sourceAuthor || '')} · 本站收录：<time datetime="${data.date.toISOString().slice(0,10)}">${data.date.toISOString().slice(0,10)}</time></p><p>${escape(data.translationScope || '中文译文')}</p></div>`;
  return document.replace(/<title>[\s\S]*?<\/title>/i, '').replace('</head>', `${metadata}</head>`).replace(/<body([^>]*)>/i, `<body$1>${navigation}`);
}
