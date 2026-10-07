import { getCollection } from 'astro:content';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import type { APIRoute } from 'astro';
export async function getStaticPaths() {
  // Local interaction testing only. No source endpoint is emitted in production.
  if (!import.meta.env.DEV) return [];
  const items = await getCollection('blog', ({ data }) => !data.draft);
  return items.map(item => ({ params: { slug: item.slug } }));
}
export const GET: APIRoute = async ({ params }) => {
  if (!import.meta.env.DEV || !/^[a-z0-9-]+$/.test(params.slug || '')) return new Response(null, { status: 404 });
  const source = await readFile(`src/content/blog/${params.slug}.md`, 'utf8');
  return Response.json({ source, sha: 'local-preview', revision: createHash('sha256').update(source).digest('base64url') });
};
