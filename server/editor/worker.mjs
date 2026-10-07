import { applyEdits } from '../../src/lib/authoring.mjs';
import { applyDocument } from '../../src/lib/author-document.mjs';

const encoder = new TextEncoder();
const decoder = new TextDecoder();
const repo = 'yuzheng310/yuzheng310.github.io';
const branch = 'main';
const encode = bytes => btoa(String.fromCharCode(...bytes)).replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');
const decode = value => Uint8Array.from(atob(value.replaceAll('-', '+').replaceAll('_', '/')), c => c.charCodeAt(0));
export const digest = async text => encode(new Uint8Array(await crypto.subtle.digest('SHA-256', encoder.encode(text))));
async function key(secret) {
  if (!secret || secret.length < 43) throw new Error('Editor service is not configured');
  return crypto.subtle.importKey('raw', await crypto.subtle.digest('SHA-256', encoder.encode(secret)), 'AES-GCM', false, ['encrypt', 'decrypt']);
}
export async function seal(payload, secret) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  return encode(iv) + '.' + encode(new Uint8Array(await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, await key(secret), encoder.encode(JSON.stringify(payload)))));
}
export async function unseal(token, secret, type) {
  const parts = token.split('.');
  if (parts.length !== 2) throw new Error('Invalid session');
  const payload = JSON.parse(decoder.decode(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: decode(parts[0]) }, await key(secret), decode(parts[1]))));
  if (payload.type !== type || !Number.isFinite(payload.exp) || payload.exp <= Date.now()) throw new Error('Expired session');
  return payload;
}
class HttpError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}
function slugOf(url) {
  const slug = url.searchParams.get('slug');
  if (!slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug) || slug.length > 100) throw new HttpError(400, '无效的文章。');
  return slug;
}
const json = (data, status = 200, headers = {}) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', ...headers } });
async function github(path, token, init = {}) {
  const response = await fetch(`https://api.github.com${path}`, { ...init, headers: { Accept: 'application/vnd.github+json', Authorization: `Bearer ${token}`, 'User-Agent': 'yuzheng-article-editor', 'X-GitHub-Api-Version': '2022-11-28', ...init.headers } });
  if (!response.ok) {
    if ([409, 422].includes(response.status)) throw new HttpError(409, '文章有了新版本，请保留草稿并重新载入后编辑。');
    throw new HttpError(response.status === 401 ? 401 : 502, response.status === 401 ? '登录已失效，请重新登录。' : '读取或保存 GitHub 文章失败，请稍后重试。');
  }
  return response.status === 204 ? null : response.json();
}
async function assertOwner(token, env) {
  const user = await github('/user', token);
  // Numeric GitHub IDs survive account renames. The allowlist is enforced here,
  // never inferred from a client-side flag or a supplied username.
  if (!/^\d+$/.test(env.OWNER_ID || '') || String(user.id) !== env.OWNER_ID) throw new HttpError(403, '只有网站作者可以编辑。');
  return user;
}
async function authenticated(request, env) {
  let session;
  try { session = await unseal(request.headers.get('Authorization')?.replace(/^Bearer /, '') || '', env.SESSION_SECRET, 'session'); }
  catch { throw new HttpError(401, '请先使用作者账号登录。'); }
  await assertOwner(session.token, env);
  return session.token;
}
function cookie(value, maxAge) { return `__Host-author-state=${value}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${maxAge}`; }
function callbackPage(env, message) {
  const nonce = encode(crypto.getRandomValues(new Uint8Array(16)));
  const data = JSON.stringify(message).replaceAll('<', '\\u003c');
  const origin = JSON.stringify(env.SITE_ORIGIN).replaceAll('<', '\\u003c');
  return new Response(`<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="referrer" content="no-referrer"><title>作者登录</title><body><p>登录处理完成，可以返回文章页面。</p><script nonce="${nonce}">if(window.opener){window.opener.postMessage(${data},${origin});window.close()}</script></body></html>`, { headers: { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer', 'Content-Security-Policy': `default-src 'none'; script-src 'nonce-${nonce}'; frame-ancestors 'none'; base-uri 'none'`, 'Set-Cookie': cookie('', 0) } });
}
export async function handle(request, env) {
  const url = new URL(request.url);
  if (!env.GITHUB_CLIENT_ID || !env.GITHUB_CLIENT_SECRET || !env.SESSION_SECRET || !/^\d+$/.test(env.OWNER_ID || '') || !/^https:\/\/[a-z0-9.-]+$/.test(env.SITE_ORIGIN || '')) return json({ error: '作者服务尚未配置。' }, 503);
  const origin = request.headers.get('Origin');
  const cors = origin === env.SITE_ORIGIN ? { 'Access-Control-Allow-Origin': origin, Vary: 'Origin' } : {};
  try {
    if (url.pathname === '/login' && request.method === 'GET') {
      const state = encode(crypto.getRandomValues(new Uint8Array(32)));
      const verifier = encode(crypto.getRandomValues(new Uint8Array(32)));
      const saved = await seal({ type: 'oauth', state, verifier, exp: Date.now() + 600000 }, env.SESSION_SECRET);
      const target = new URL('https://github.com/login/oauth/authorize');
      target.search = new URLSearchParams({ client_id: env.GITHUB_CLIENT_ID, redirect_uri: url.origin + '/callback', state, code_challenge: await digest(verifier), code_challenge_method: 'S256', allow_signup: 'false' }).toString();
      // Register a GitHub App with Contents: read/write on this repository only.
      // No broad OAuth `repo` scope and no installation-wide private key required.
      return new Response(null, { status: 302, headers: { Location: target.href, 'Set-Cookie': cookie(saved, 600), 'Cache-Control': 'no-store' } });
    }
    if (url.pathname === '/callback' && request.method === 'GET') {
      try {
        const saved = request.headers.get('Cookie')?.match(/(?:^|;\s*)__Host-author-state=([^;]+)/)?.[1];
        const flow = await unseal(saved || '', env.SESSION_SECRET, 'oauth');
        if (!url.searchParams.get('code') || flow.state !== url.searchParams.get('state')) throw new Error('State mismatch');
        const response = await fetch('https://github.com/login/oauth/access_token', { method: 'POST', headers: { Accept: 'application/json', 'Content-Type': 'application/json' }, body: JSON.stringify({ client_id: env.GITHUB_CLIENT_ID, client_secret: env.GITHUB_CLIENT_SECRET, code: url.searchParams.get('code'), code_verifier: flow.verifier, redirect_uri: url.origin + '/callback' }) });
        const result = await response.json();
        if (!response.ok || !result.access_token) throw new Error('OAuth denied');
        await assertOwner(result.access_token, env);
        const session = await seal({ type: 'session', token: result.access_token, exp: Date.now() + Math.min(3600, result.expires_in || 3600) * 1000 }, env.SESSION_SECRET);
        return callbackPage(env, { type: 'author-login', session });
      } catch { return callbackPage(env, { type: 'author-login', error: '登录失败或账号没有编辑权限，请使用网站作者账号重试。' }); }
    }
    if (origin !== env.SITE_ORIGIN) throw new HttpError(403, '请求来源不受信任。');
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: { ...cors, 'Access-Control-Allow-Methods': 'GET, POST, OPTIONS', 'Access-Control-Allow-Headers': 'Authorization, Content-Type', 'Access-Control-Max-Age': '600' } });
    const token = await authenticated(request, env);
    if (url.pathname === '/session' && request.method === 'GET') return json({ owner: true }, 200, cors);
    const slug = slugOf(url);
    const path = `/repos/${repo}/contents/src/content/blog/${slug}.md`;
    if (!((url.pathname === '/article' && request.method === 'GET') || (url.pathname === '/publish' && request.method === 'POST'))) throw new HttpError(404, '没有这个接口。');
    const file = await github(`${path}?ref=${branch}`, token);
    if (file.type !== 'file' || typeof file.content !== 'string') throw new HttpError(400, '无法编辑这篇文章。');
    const source = decoder.decode(decode(file.content.replace(/\s/g, '')));
    if (url.pathname === '/article') return json({ source, sha: file.sha, revision: await digest(source) }, 200, cors);
    if (!request.headers.get('Content-Type')?.startsWith('application/json')) throw new HttpError(415, '请使用 JSON 请求。');
    const raw = await request.text();
    if (raw.length > 1200000) throw new HttpError(413, '修改内容过大。');
    const body = JSON.parse(raw);
    if (typeof body.sha !== 'string' || body.sha !== file.sha) throw new HttpError(409, '文章已被修改，请保留草稿并重新载入。');
    let updated;
    try {
      if (body.format === 'markdown-v1') {
        if (body.edits !== undefined) throw new Error('不能混用整篇保存和选区修改。');
        updated = applyDocument(source, body.markdown, body.intro);
      } else updated = applyEdits(source, body.edits, body.intro);
    }
    catch (error) { throw new HttpError(400, error.message); }
    if (updated === source) throw new HttpError(400, '没有需要发布的修改。');
    // UTF-8 base64 without spreading large article buffers into function arguments.
    let binary = ''; for (const byte of encoder.encode(updated)) binary += String.fromCharCode(byte);
    const result = await github(path, token, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: `Edit article: ${slug}`, content: btoa(binary), sha: file.sha, branch }) });
    return json({ commit: result.commit.sha, url: result.commit.html_url, published: false, message: '修改已提交，网站正在自动构建。' }, 200, cors);
  } catch (error) { return json({ error: error instanceof HttpError ? error.message : '请求处理失败，请重试。' }, error instanceof HttpError ? error.status : 500, cors); }
}
export default { fetch: handle };
