import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { applyEdits } from '../src/lib/authoring.mjs';
import { digest, handle, seal, unseal } from '../server/editor/worker.mjs';
const original = readFileSync(new URL('../src/content/blog/alisa-llm-infrastructure.md', import.meta.url), 'utf8');
const quote = '读大模型系统资料时';
const start = original.indexOf(quote);
const change = { start, end: start + quote.length, quote, kind: 'replace', value: '学习大模型系统时' };
const env = { GITHUB_CLIENT_ID: 'test-client', GITHUB_CLIENT_SECRET: 'test-secret', SESSION_SECRET: 'test-session-secret-that-is-at-least-43-characters-long', OWNER_ID: '1234', SITE_ORIGIN: 'https://yuzheng310.github.io' };
const service = 'https://editor.example.workers.dev';

test('editing changes only the selected bytes, preserving every equation and code block', () => {
  const result = applyEdits(original, [change]);
  assert.equal(result, original.slice(0, start) + change.value + original.slice(change.end));
});
test('highlight, annotation and intro serialize safely; annotations survive subsequent edits', () => {
  const result = applyEdits(original, [{ ...change, kind: 'note', value: '这里有 <script> 和 $符号，需保留为文字。' }], '读前提示："测试"\n第二行');
  assert.ok(result.includes('<mark class="author-highlight">' + quote + '</mark>[^author-1]'));
  assert.ok(result.includes('&lt;script&gt;'));
  assert.ok(result.includes('intro: "读前提示：\\"测试\\"\\n第二行"'));
  assert.ok(result.includes('\\$符号'));
  const secondStart = result.indexOf('阅读时可以抓住三个问题');
  const second = applyEdits(result, [{ start: secondStart, end: secondStart + 2, quote: '阅读', kind: 'note', value: '第二条' }]);
  assert.ok(second.includes('[^author-2]'));
  assert.ok(second.includes('[^author-1]: 作者批注：'));
  const highlight = applyEdits(original, [{ ...change, kind: 'highlight', value: '' }]);
  assert.ok(highlight.includes('<mark class="author-highlight">' + quote + '</mark>'));
});
test('stale quotes, overlaps, invalid types, invalid ranges and multiline YAML fields fail closed', () => {
  assert.throws(() => applyEdits(original, [{ ...change, quote: '过时内容' }]));
  assert.throws(() => applyEdits(original, [change, change]));
  assert.throws(() => applyEdits(original, [{ ...change, start: -1 }]));
  assert.throws(() => applyEdits(original, [{ ...change, kind: 'html' }]));
  assert.throws(() => applyEdits('---\nintro: |\n  old\n---\nbody', [], 'new'));
});
test('empty intro is supported; text is not interpreted as HTML or Markdown', () => {
  const result = applyEdits('---\ntitle: test\nintro: "old"\n---\n正文', [], '');
  assert.ok(result.includes('intro: ""'));
  const escaped = applyEdits(original, [{ ...change, value: '**<img src=x onerror=alert(1)>**' }]);
  assert.ok(escaped.includes('\\*\\*&lt;img'));
});
test('sessions reject tampering, expiration and token-type confusion', async () => {
  const session = await seal({ type: 'session', token: 'github-test', exp: Date.now() + 10000 }, env.SESSION_SECRET);
  assert.equal((await unseal(session, env.SESSION_SECRET, 'session')).token, 'github-test');
  await assert.rejects(unseal(session.slice(0, -4) + 'aaaa', env.SESSION_SECRET, 'session'));
  await assert.rejects(unseal(session, env.SESSION_SECRET, 'oauth'));
  await assert.rejects(unseal(await seal({ type: 'session', exp: 1 }, env.SESSION_SECRET), env.SESSION_SECRET, 'session'));
});
test('service rejects missing configuration, anonymous writes, wrong origin and forged sessions', async () => {
  assert.equal((await handle(new Request(service + '/login'), {})).status, 503);
  for (const path of ['/article?slug=session-kv', '/publish?slug=session-kv']) {
    const method = path.startsWith('/publish') ? 'POST' : 'GET';
    assert.equal((await handle(new Request(service + path, { method, headers: { Origin: env.SITE_ORIGIN } }), env)).status, 401);
    assert.equal((await handle(new Request(service + path, { method, headers: { Origin: 'https://attacker.example', Authorization: 'Bearer fake' } }), env)).status, 403);
  }
});
test('GitHub owner check, source conflicts and the fixed repository/path are enforced', async t => {
  const savedFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = savedFetch; });
  const session = await seal({ type: 'session', token: 'github-test', exp: Date.now() + 100000 }, env.SESSION_SECRET);
  const headers = { Origin: env.SITE_ORIGIN, Authorization: `Bearer ${session}`, 'Content-Type': 'application/json' };
  let owner = false, writes = [];
  globalThis.fetch = async (url, init) => {
    assert.ok(String(url).startsWith('https://api.github.com/'));
    if (String(url).endsWith('/user')) return Response.json({ id: owner ? 1234 : 9999 });
    assert.ok(String(url).startsWith('https://api.github.com/repos/yuzheng310/yuzheng310.github.io/contents/src/content/blog/alisa-llm-infrastructure.md'));
    if (init.method === 'PUT') { writes.push(JSON.parse(init.body)); return Response.json({ commit: { sha: 'new-commit', html_url: 'https://github.com/example/commit/new' } }); }
    return Response.json({ type: 'file', sha: 'current-sha', content: Buffer.from(original).toString('base64') });
  };
  const url = service + '/publish?slug=alisa-llm-infrastructure';
  const request = sha => new Request(url, { method: 'POST', headers, body: JSON.stringify({ sha, edits: [change] }) });
  assert.equal((await handle(request('current-sha'), env)).status, 403);
  owner = true;
  assert.equal((await handle(request('stale-sha'), env)).status, 409);
  assert.equal(writes.length, 0);
  assert.equal((await handle(new Request(service + '/article?slug=../secrets', { headers }), env)).status, 400);
  const result = await handle(request('current-sha'), env);
  assert.equal(result.status, 200);
  assert.equal((await result.json()).published, false);
  assert.equal(writes.length, 1);
  assert.equal(writes[0].sha, 'current-sha');
  assert.equal(writes[0].branch, 'main');
  assert.equal(Buffer.from(writes[0].content, 'base64').toString(), applyEdits(original, [change]));
});
test('OAuth uses PKCE, secure state cookies and rejects callbacks without matching state', async () => {
  const result = await handle(new Request(service + '/login'), env);
  assert.equal(result.status, 302);
  const url = new URL(result.headers.get('Location'));
  assert.equal(url.searchParams.get('code_challenge_method'), 'S256');
  const cookie = result.headers.get('Set-Cookie');
  assert.match(cookie, /HttpOnly; Secure; SameSite=Lax/);
  const flow = await unseal(cookie.split(';')[0].split('=')[1], env.SESSION_SECRET, 'oauth');
  assert.equal(url.searchParams.get('code_challenge'), await digest(flow.verifier));
  const denied = await handle(new Request(service + '/callback?code=invalid&state=mismatch', { headers: { Cookie: cookie.split(';')[0] } }), env);
  const html = await denied.text();
  assert.ok(html.includes('登录失败'));
  assert.ok(!html.includes('"session":'));
  assert.match(denied.headers.get('Content-Security-Policy'), /frame-ancestors 'none'/);
});

test('saved annotations render through Astro with public footnotes and unchanged math/code', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.resolve('astro'));
  const { createMarkdownProcessor } = await import(require.resolve('@astrojs/markdown-remark'));
  const { default: remarkMath } = await import('remark-math');
  const { default: rehypeKatex } = await import('rehype-katex');
  const processor = await createMarkdownProcessor({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] });
  const source = '正文重点 $x^2$\n\n$$\ny = x + 1\n$$\n\n```js\nconst n = 1;\n```';
  const updated = applyEdits(source, [{ start: 2, end: 4, quote: '重点', kind: 'note', value: '解释 <script> 和 $x$' }]);
  const before = (await processor.render(source)).code;
  const after = (await processor.render(updated)).code;
  assert.ok(after.includes('<mark class="author-highlight">重点</mark>'));
  assert.ok(after.includes('data-footnotes'));
  assert.match(after, /作者批注：解释 (?:&#x3C;|&lt;)script(?:>|&gt;) 和 \$x\$/);
  assert.ok(!after.includes('<script>'));
  assert.deepEqual([...after.matchAll(/<span class="katex[\s\S]*?<\/math>/g)].map(x => x[0]), [...before.matchAll(/<span class="katex[\s\S]*?<\/math>/g)].map(x => x[0]));
  assert.equal(after.match(/<pre[\s\S]*?<\/pre>/)?.[0], before.match(/<pre[\s\S]*?<\/pre>/)?.[0]);
});

test('successful OAuth callback seals the owner token and rejects a different user', async t => {
  const savedFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = savedFetch; });
  const login = await handle(new Request(service + '/login'), env);
  const cookie = login.headers.get('Set-Cookie').split(';')[0];
  const state = new URL(login.headers.get('Location')).searchParams.get('state');
  let owner = true;
  globalThis.fetch = async (url, init) => {
    if (String(url).endsWith('/access_token')) {
      assert.ok(JSON.parse(init.body).code_verifier);
      return Response.json({ access_token: 'test-github-token', expires_in: 28800 });
    }
    return Response.json({ id: owner ? 1234 : 9 });
  };
  const callback = () => handle(new Request(service + '/callback?code=test&state=' + state, { headers: { Cookie: cookie } }), env);
  const html = await (await callback()).text();
  assert.ok(!html.includes('test-github-token'));
  const sealed = html.match(/"session":"([^"]+)"/)[1];
  const session = await unseal(sealed, env.SESSION_SECRET, 'session');
  assert.equal(session.token, 'test-github-token');
  assert.ok(session.exp <= Date.now() + 3600000);
  owner = false;
  assert.ok(!(await (await callback()).text()).includes('"session":'));
});
