import { applyEdits } from '../lib/authoring.mjs';
type Edit = { start: number; end: number; quote: string; kind: 'replace' | 'highlight' | 'note'; value: string };
type Target = { span: HTMLElement; start: number; end: number; quote: string };
const root = document.querySelector<HTMLElement>('.author-editor');
if (root) setup(root);
function setup(root: HTMLElement) {
  const query = <T extends HTMLElement>(selector: string) => root.querySelector<T>(selector)!;
  const entry = query<HTMLButtonElement>('.author-entry');
  const bar = query('.author-bar');
  const selectionBar = query('.author-selection');
  const status = query('.author-status');
  const dialog = query<HTMLDialogElement>('.author-dialog');
  const title = query('.author-dialog-title');
  const quote = query('.author-quote');
  const label = query('.author-field-label');
  const input = query<HTMLTextAreaElement>('#author-input');
  const review = query('.author-review');
  const error = query('.author-dialog-error');
  const apply = query<HTMLButtonElement>('.author-apply');
  const introBox = document.querySelector<HTMLElement>('.author-intro')!;
  const initialIntro: string = JSON.parse(query('.author-initial-intro').textContent || '""');
  const api = root.dataset.api!;
  const slug = root.dataset.slug!;
  const local = import.meta.env.DEV && !api && ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
  let session = ''; // Memory only: never store credentials in URLs or localStorage.
  let source = '', sha = '', editing = false, busy = false;
  let target: Target | undefined;
  let mode: 'replace' | 'highlight' | 'note' | 'intro' | 'review' | 'read-note' | 'exit' = 'replace';
  let edits: Edit[] = [];
  let intro = initialIntro;
  const history: { edits: Edit[]; intro: string }[] = [];
  const originals = new Map<HTMLElement, string>();
  let hideStatus: number | undefined;
  if (local) entry.textContent = '体验页内编辑（本地）';
  const dirty = () => edits.length > 0 || intro !== initialIntro;
  function notify(message: string) {
    clearTimeout(hideStatus); status.textContent = message; status.hidden = false;
    hideStatus = window.setTimeout(() => { status.hidden = true; }, 8500);
  }
  async function request(path: string, options: RequestInit = {}) {
    const response = await fetch(`${api}${path}`, { ...options, headers: { Authorization: `Bearer ${session}`, ...options.headers } });
    const result = await response.json();
    if (response.status === 401 && editing) query<HTMLButtonElement>('[data-action=login]').hidden = false;
    if (!response.ok) throw new Error(result.error || '请求失败，请重试。');
    return result;
  }
  async function start() {
    busy = true; entry.disabled = true;
    try {
      const article = local ? await fetch(`/author/source/${slug}.json`).then(r => r.json()) : await request(`/article?slug=${slug}`);
      if (article.revision !== root.dataset.revision) throw new Error('这篇文章已有新版本，请等网站更新后刷新再编辑。');
      source = article.source; sha = article.sha; editing = true;
      query<HTMLButtonElement>('[data-action=login]').hidden = true;
      document.body.classList.add('author-editing'); bar.hidden = false; entry.hidden = true;
      root.querySelector('.author-mode')!.textContent = local ? '本地体验 · 不会发布' : '作者编辑';
      notify('选中一段普通文字，即可修改、高亮或批注。公式、代码和标题保持原样。');
    } catch (e) { notify((e as Error).message); }
    finally { busy = false; entry.disabled = false; }
  }
  entry.addEventListener('click', () => {
    if (busy) return;
    if (local) { void start(); return; }
    let endpoint: URL;
    try { endpoint = new URL(api); if (endpoint.protocol !== 'https:') throw new Error(); }
    catch { notify('作者登录服务尚未接通。'); return; }
    const popup = window.open(`${api}/login`, 'yuzheng-author-login', 'popup,width=540,height=700');
    if (!popup) { notify('请允许登录弹窗，然后重试。'); return; }
    entry.disabled = true;
    const cleanup = () => { window.removeEventListener('message', receive); clearInterval(closed); clearTimeout(timeout); entry.disabled = false; };
    const receive = (event: MessageEvent) => {
      if (event.origin !== endpoint.origin || event.source !== popup || event.data?.type !== 'author-login') return;
      cleanup();
      if (typeof event.data.session === 'string') { session = event.data.session; void start(); }
      else notify(event.data.error || '登录失败。');
    };
    window.addEventListener('message', receive);
    const closed = window.setInterval(() => { if (popup.closed) cleanup(); }, 1000);
    const timeout = window.setTimeout(() => { cleanup(); notify('登录等待超时，请重试。'); }, 600000);
  });
  function render() {
    for (const [span, text] of originals) {
      const start = Number(span.dataset.authorStart);
      const changes = edits.filter(edit => edit.start >= start && edit.end <= Number(span.dataset.authorEnd)).sort((a, b) => a.start - b.start);
      span.replaceChildren();
      let offset = 0;
      for (const edit of changes) {
        span.append(document.createTextNode(text.slice(offset, edit.start - start)));
        if (edit.kind === 'replace') span.append(document.createTextNode(edit.value.replace(/\r?\n/g, ' ')));
        else {
          const mark = document.createElement('mark'); mark.className = 'author-highlight'; mark.textContent = edit.quote; span.append(mark);
          if (edit.kind === 'note') {
            const note = document.createElement('button'); note.type = 'button'; note.className = 'author-note-preview'; note.textContent = '批注';
            note.addEventListener('click', () => { show('read-note'); quote.textContent = edit.quote; input.value = edit.value; }); span.append(note);
          }
        }
        offset = edit.end - start;
      }
      span.append(document.createTextNode(text.slice(offset)));
      span.dataset.authorDirty = String(changes.length > 0);
    }
    introBox.hidden = !intro.trim(); introBox.querySelector('p')!.textContent = intro;
    const count = edits.length + Number(intro !== initialIntro);
    query('.author-count').textContent = count ? `${count} 处未发布` : '未修改';
    query<HTMLButtonElement>('[data-action=undo]').disabled = history.length === 0;
    query<HTMLButtonElement>('[data-action=review]').disabled = !dirty();
    document.dispatchEvent(new Event('article-font-change'));
  }
  function record() { history.push({ edits: edits.map(edit => ({ ...edit })), intro }); }
  function show(next: typeof mode) {
    mode = next; error.textContent = ''; review.replaceChildren(); quote.textContent = target?.quote || '';
    const names = { replace: '修改选中文字', highlight: '高亮', note: '添加公开批注', intro: '编辑文章导语', review: '预览改动', 'read-note': '作者批注', exit: '退出编辑' };
    title.textContent = names[next];
    input.hidden = label.hidden = ['review', 'exit'].includes(next);
    quote.hidden = ['intro', 'review', 'exit'].includes(next);
    label.textContent = next === 'note' ? '批注内容（发布后所有读者可见）' : next === 'intro' ? '导语（显示在正文之前）' : '内容';
    input.readOnly = next === 'read-note';
    input.maxLength = next === 'intro' ? 3000 : 10000;
    input.value = next === 'intro' ? intro : next === 'replace' ? target?.quote || '' : '';
    apply.textContent = next === 'review' ? (local ? '本地体验无法发布' : '确认发布') : next === 'read-note' ? '关闭' : next === 'exit' ? '放弃修改并退出' : '应用修改';
    apply.disabled = next === 'review' && local;
    if (next === 'exit') { const p = document.createElement('p'); p.textContent = '还有未发布的修改。退出会丢弃这些修改，你也可以取消并继续编辑。'; review.append(p); }
    dialog.showModal(); if (!input.hidden && !input.readOnly) input.focus();
  }
  query('.author-cancel').addEventListener('click', () => dialog.close());
  function saveSelection() {
    if (!editing || busy || dialog.open) return;
    const selected = window.getSelection();
    if (!selected || selected.isCollapsed || !selected.rangeCount) { selectionBar.hidden = true; return; }
    const range = selected.getRangeAt(0);
    const startElement = range.startContainer.nodeType === Node.TEXT_NODE ? range.startContainer.parentElement : range.startContainer as Element;
    const endElement = range.endContainer.nodeType === Node.TEXT_NODE ? range.endContainer.parentElement : range.endContainer as Element;
    const span = startElement?.closest<HTMLElement>('.prose [data-author-start]');
    if (!span || span !== endElement?.closest('.prose [data-author-start]') || span.dataset.authorDirty === 'true') {
      selectionBar.hidden = true;
      if (startElement?.closest('.prose')) notify('请在同一段普通文字内选择，避开公式、代码、链接和格式边界。已有草稿的段落可先撤销再改。');
      return;
    }
    const prefix = document.createRange(); prefix.selectNodeContents(span); prefix.setEnd(range.startContainer, range.startOffset);
    const start = Number(span.dataset.authorStart) + prefix.toString().length;
    const text = range.toString();
    if (!text.trim() || source.slice(start, start + text.length) !== text) { selectionBar.hidden = true; return; }
    target = { span, start, end: start + text.length, quote: text };
    const bounds = range.getBoundingClientRect();
    selectionBar.hidden = false;
    selectionBar.style.left = `${Math.max(8, Math.min(innerWidth - selectionBar.offsetWidth - 8, bounds.left))}px`;
    selectionBar.style.top = `${Math.max(8, Math.min(innerHeight - selectionBar.offsetHeight - 8, bounds.top - selectionBar.offsetHeight - 10))}px`;
  }
  document.addEventListener('pointerup', event => { if (!root.contains(event.target as Node)) window.setTimeout(saveSelection, 0); });
  document.addEventListener('keyup', event => { if (event.shiftKey) saveSelection(); if (event.key === 'Escape') selectionBar.hidden = true; });
  window.addEventListener('scroll', () => { selectionBar.hidden = true; }, { passive: true });
  selectionBar.addEventListener('pointerdown', event => event.preventDefault());
  selectionBar.querySelectorAll<HTMLButtonElement>('[data-kind]').forEach(button => button.addEventListener('click', () => {
    if (!target) return;
    selectionBar.hidden = true;
    if (button.dataset.kind === 'highlight') stage('highlight', '');
    else show(button.dataset.kind as 'replace' | 'note');
  }));
  function stage(kind: Edit['kind'], value: string) {
    if (!target) return;
    record();
    if (!originals.has(target.span)) originals.set(target.span, target.span.textContent || '');
    edits.push({ start: target.start, end: target.end, quote: target.quote, kind, value });
    render(); window.getSelection()?.removeAllRanges(); notify('修改已应用到当前预览，尚未公开发布。');
  }
  bar.addEventListener('click', event => {
    const action = (event.target as HTMLElement).closest<HTMLElement>('[data-action]')?.dataset.action;
    if (busy) return;
    if (action === 'login') entry.click();
    if (action === 'intro') show('intro');
    if (action === 'undo') { const previous = history.pop(); if (previous) { edits = previous.edits; intro = previous.intro; render(); } }
    if (action === 'exit') { if (dirty()) show('exit'); else exit(); }
    if (action === 'review') {
      show('review');
      const heading = document.createElement('p'); heading.textContent = local ? '这是本地体验，不能写入线上文章。' : '发布后，导语、高亮和批注都将公开。修改会记录到 GitHub，网站构建完成后生效。'; review.append(heading);
      const changes = [...edits.map(edit => ({ kind: { replace: '修改文字', highlight: '高亮', note: '作者批注' }[edit.kind], before: edit.quote, after: edit.kind === 'highlight' ? '标为重点' : edit.value })), ...(intro !== initialIntro ? [{ kind: '导语', before: initialIntro, after: intro }] : [])];
      for (const change of changes) {
        const article = document.createElement('article'); const strong = document.createElement('strong'); strong.textContent = change.kind;
        const old = document.createElement('p'); old.textContent = `原文：${change.before || '（空）'}`;
        const next = document.createElement('p'); next.textContent = `修改：${change.after || '（删除）'}`;
        article.append(strong, old, next); review.append(article);
      }
    }
  });
  function exit() {
    edits = []; intro = initialIntro; history.length = 0; render(); editing = false; session = '';
    document.body.classList.remove('author-editing'); bar.hidden = true; selectionBar.hidden = true; entry.hidden = false; dialog.close();
  }
  dialog.querySelector('form')!.addEventListener('submit', async event => {
    event.preventDefault(); if (busy) return;
    if (mode === 'exit') { exit(); return; }
    if (mode === 'read-note') { dialog.close(); return; }
    if (mode === 'intro') { record(); intro = input.value.trim(); render(); dialog.close(); return; }
    if (mode === 'replace' || mode === 'note') {
      if (mode === 'note' && !input.value.trim()) { error.textContent = '请填写批注内容。'; return; }
      stage(mode, input.value); dialog.close(); return;
    }
    if (mode !== 'review' || local) return;
    busy = true; apply.disabled = true; error.textContent = '';
    try {
      applyEdits(source, edits, intro !== initialIntro ? intro : undefined);
      const result = await request(`/publish?slug=${slug}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sha, edits, ...(intro !== initialIntro ? { intro } : {}) }) });
      // The commit is saved, but the build may still fail. Say exactly what is known.
      edits = []; history.length = 0; editing = false; session = ''; document.body.classList.remove('author-editing'); bar.hidden = true; dialog.close();
      intro = initialIntro; // Clear dirty tracking; keep the visual preview in place.
      entry.hidden = false; entry.disabled = true; entry.textContent = '修改已提交，等待网站更新';
      notify(result.message || '修改已提交，网站正在自动构建。');
      const link = document.createElement('a'); link.href = 'https://github.com/yuzheng310/yuzheng310.github.io/actions'; link.textContent = '查看发布状态 ↗'; link.target = '_blank'; link.rel = 'noopener noreferrer'; entry.after(link);
    } catch (e) { error.textContent = (e as Error).message; }
    finally { busy = false; apply.disabled = false; }
  });
  window.addEventListener('beforeunload', event => { if (dirty()) { event.preventDefault(); } });
}
