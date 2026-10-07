import { applyDocument, splitArticle } from "../lib/author-document.mjs";
type Surface = Awaited<ReturnType<typeof import("./milkdown/surface").mountArticleEditor>>;
export function setup(root: HTMLElement, preparedPopup: Window | null = null) {
  const q = <T extends HTMLElement>(selector: string) => root.querySelector<T>(selector)!;
  const entry = q<HTMLButtonElement>(".author-entry"), bar = q(".author-bar");
  const status = q(".author-status"), dialog = q<HTMLDialogElement>(".author-dialog");
  const input = q<HTMLTextAreaElement>("#author-input"), apply = q<HTMLButtonElement>(".author-apply");
  const error = q(".author-dialog-error"), review = q(".author-review");
  const published = document.querySelector<HTMLElement>(".article-main > .prose")!;
  published.classList.add("article-published-body");
  const host = document.createElement("div"); host.className = "inline-editor-host"; host.hidden = true;
  const preview = document.createElement("div"); preview.className = "prose article-draft-preview"; preview.hidden = true;
  published.after(host, preview);
  const introBox = document.querySelector<HTMLElement>(".author-intro")!;
  const initialIntro: string = JSON.parse(q(".author-initial-intro").textContent || "\"\"");
  const api = root.dataset.api!, slug = root.dataset.slug!;
  const local = import.meta.env.DEV && !api && ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
  let session = "", source = "", sha = "", intro = initialIntro;
  let surface: Surface | undefined;
  let busy = false, editing = false, previewing = false, bodyDirty = false;
  let mode: "note" | "intro" | "notes" | "review" | "exit" | "delete" = "note";
  let noteId: string | undefined, timer: number | undefined, previewVersion = 0;
  entry.disabled = false;
  if (local) entry.textContent = "体验原位编辑（本地）";
  const dirty = () => bodyDirty || intro !== initialIntro;
  function notify(message: string) {
    clearTimeout(timer); status.textContent = message; status.hidden = false;
    timer = window.setTimeout(() => { status.hidden = true; }, 9000);
  }
  function renderState() {
    bodyDirty = Boolean(surface?.changed());
    q(".author-count").textContent = dirty() ? "有未发布修改" : "未修改";
    q<HTMLButtonElement>("[data-action=review]").disabled = !dirty() || busy;
    q<HTMLButtonElement>("[data-action=undo]").disabled = !editing || previewing;
    q<HTMLButtonElement>("[data-action=redo]").disabled = !editing || previewing;
    q<HTMLButtonElement>("[data-action=preview]").textContent = previewing ? "继续编辑" : "阅读预览";
    introBox.hidden = !intro.trim(); introBox.querySelector("p")!.textContent = intro;
  }
  async function request(path: string, options: RequestInit = {}) {
    const response = await fetch(`${api}${path}`, { ...options, headers: { Authorization: `Bearer ${session}`, ...options.headers } });
    const result = await response.json();
    if (response.status === 401) q<HTMLButtonElement>("[data-action=login]").hidden = false;
    if (!response.ok) throw new Error(result.error || "请求失败，请重试。");
    return result;
  }
  async function start() {
    busy = true; entry.disabled = true;
    try {
      if (editing) {
        await request("/session"); q<HTMLButtonElement>("[data-action=login]").hidden = true;
        notify("登录已恢复，当前草稿已保留。"); return;
      }
      const article = local ? await fetch(`/author/source/${slug}.json`).then(r => { if (!r.ok) throw new Error("无法读取本地文章。"); return r.json(); }) : await request(`/article?slug=${slug}`);
      if (article.revision !== root.dataset.revision) throw new Error("这篇文章已有新版本，请等网站更新后刷新再编辑。");
      source = article.source; sha = article.sha;
      const { mountArticleEditor } = await import("./milkdown/surface");
      host.hidden = false;
      surface = await mountArticleEditor(host, splitArticle(source).body, renderState, () => openNote());
      published.hidden = true; editing = true; bodyDirty = false;
      bar.hidden = false; entry.hidden = true; q<HTMLButtonElement>("[data-action=login]").hidden = true;
      document.body.classList.add("author-editing");
      q(".author-mode").textContent = local ? "本地体验 · 不会发布" : "作者编辑";
      renderState(); notify("可以直接在正文输入。选中文字后可高亮或批注，也可以跨段落和列表选择。");
    } catch (e) { if (!editing) { host.hidden = true; host.replaceChildren(); } notify((e as Error).message); }
    finally { busy = false; entry.disabled = false; }
  }
  function login() {
    if (busy) return;
    if (local) { void start(); return; }
    let endpoint: URL;
    try { endpoint = new URL(api); if (endpoint.protocol !== "https:") throw new Error(); }
    catch { notify("作者登录服务尚未接通。"); return; }
    const popup = preparedPopup && !preparedPopup.closed ? preparedPopup : window.open(`${api}/login`, "yuzheng-author-login", "popup,width=540,height=700");
    if (preparedPopup && popup) popup.location.href = `${api}/login`;
    preparedPopup = null;
    if (!popup) { notify("请允许登录弹窗，然后重试。"); return; }
    busy = true; entry.disabled = true;
    const cleanup = () => { window.removeEventListener("message", receive); clearInterval(closed); clearTimeout(timeout); busy = false; entry.disabled = false; };
    const receive = (event: MessageEvent) => {
      if (event.origin !== endpoint.origin || event.source !== popup || event.data?.type !== "author-login") return;
      cleanup();
      if (typeof event.data.session === "string") { session = event.data.session; void start(); }
      else notify(event.data.error || "登录失败。");
    };
    window.addEventListener("message", receive);
    const closed = window.setInterval(() => { if (popup.closed) cleanup(); }, 1000);
    const timeout = window.setTimeout(() => { cleanup(); notify("登录等待超时，请重试。"); }, 600000);
  }
  entry.addEventListener("click", login);
  const draftBody = () => surface?.changed() ? surface.getMarkdown() : splitArticle(source).body;
  function validatedDraft() { return applyDocument(source, draftBody(), intro !== initialIntro ? intro : undefined); }

  function show(next: typeof mode) {
    mode = next; error.textContent = ""; review.replaceChildren(); review.className = "author-review";
    q(".author-quote").hidden = true;
    input.hidden = q(".author-field-label").hidden = !["note", "intro"].includes(next);
    input.readOnly = false; input.value = ""; input.maxLength = next === "intro" ? 3000 : 10000;
    q(".author-note-help").hidden = next !== "note"; q(".author-note-rendered").hidden = next !== "note";
    q(".author-dialog-title").textContent = { note: noteId ? "修改公开批注" : "添加公开批注", intro: "编辑导语", notes: "管理公开批注", review: "发布修改", exit: "退出编辑", delete: "删除批注" }[next];
    q(".author-field-label").textContent = next === "intro" ? "导语（正文前展示）" : "批注内容（支持 Markdown 和公式）";
    apply.disabled = next === "review" && local;
    apply.textContent = { note: "应用批注", intro: "应用导语", notes: "关闭", review: local ? "本地体验无法发布" : "确认发布", exit: "放弃草稿并退出", delete: "确认删除" }[next];
    if (!dialog.open) dialog.showModal();
  }
  function openNote(id?: string) {
    if (!surface || busy) return;
    const selection = surface.selection();
    if (!id && (!selection.text.trim() || previewing)) { notify("请先在编辑中的正文选中文字。"); return; }
    noteId = id; show("note");
    input.value = id ? surface.notes().find((note: { id: string }) => note.id === id)?.markdown || "" : "";
    if (!id) { q(".author-quote").hidden = false; q(".author-quote").textContent = selection.text; }
    void previewNote(); input.focus();
  }
  async function previewNote() {
    const version = ++previewVersion;
    try {
      const { renderNote } = await import("./note-preview");
      const html = input.value.trim() ? await renderNote(input.value) : "<p>批注预览会显示在这里。</p>";
      if (version === previewVersion && mode === "note") q(".author-note-rendered").innerHTML = html;
    } catch (e) { if (version === previewVersion) error.textContent = (e as Error).message; }
  }
  input.addEventListener("input", () => { if (mode === "note") void previewNote(); });
  q(".author-cancel").addEventListener("click", () => dialog.close());
  function manageNotes() {
    show("notes"); review.classList.add("author-notes-list");
    const notes = surface?.notes() || [];
    if (!notes.length) review.textContent = "还没有公开批注。选中正文后，点击浮动工具栏中的“批注”。";
    for (const note of notes) {
      const card = document.createElement("article"), title = document.createElement("strong"), text = document.createElement("p");
      title.textContent = `作者批注 ${note.id.replace("author-", "")}`; text.textContent = note.markdown;
      const edit = document.createElement("button"); edit.type = "button"; edit.textContent = "修改批注"; edit.addEventListener("click", () => openNote(note.id));
      const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "删除批注";
      remove.addEventListener("click", () => { noteId = note.id; show("delete"); review.textContent = "将删除这条批注及其关联高亮，原文保留。发布前可以撤销。"; });
      card.append(title, text, edit, remove); review.append(card);
    }
  }
  host.addEventListener("click", event => {
    const element = (event.target as Element).closest<HTMLElement>("sup[data-type=footnote_reference]");
    const id = element?.dataset.authorNote || element?.dataset.label;
    if (id && /^author-\d+$/.test(id)) openNote(id);
  });
  for (const name of ["drop", "paste"]) host.addEventListener(name, event => {
    const files = name === "paste" ? (event as ClipboardEvent).clipboardData?.files : (event as DragEvent).dataTransfer?.files;
    if (files?.length) { event.preventDefault(); notify("图片请先存入网站资源，再使用链接插入；本地临时图片无法发布。"); }
  }, true);
  async function setPreview(value: boolean) {
    if (!surface) return;
    if (value) {
      validatedDraft();
      const { previewArticle } = await import("./article-preview");
      await previewArticle(preview, draftBody());
    }
    previewing = value; host.hidden = value; preview.hidden = !value; surface.setReadonly(value); renderState();
  }
  function exportDraft() {
    const blob = new Blob([validatedDraft()], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = `${slug}-draft.md`; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function exit() {
    editing = false; bodyDirty = false; intro = initialIntro; session = ""; previewing = false;
    await surface?.destroy(); surface = undefined;
    host.replaceChildren(); host.hidden = true; preview.hidden = true; published.hidden = false;
    bar.hidden = true; entry.hidden = false; dialog.close(); document.body.classList.remove("author-editing"); renderState();
  }
  bar.addEventListener("click", async event => {
    const action = (event.target as HTMLElement).closest<HTMLElement>("[data-action]")?.dataset.action;
    if (busy || !action) return;
    if (action === "login") { login(); return; }
    try {
      if (action === "intro") { show("intro"); input.value = intro; input.focus(); }
      if (action === "notes") manageNotes();
      if (action === "undo") { surface?.undo(); renderState(); }
      if (action === "redo") { surface?.redo(); renderState(); }
      if (action === "preview") { busy = true; await setPreview(!previewing); }
      if (action === "export") exportDraft();
      if (action === "exit") { if (dirty()) { show("exit"); review.textContent = "还有未发布的修改。可以取消并继续编辑，或先下载草稿。"; } else await exit(); }
      if (action === "review") {
        busy = true; await setPreview(true); show("review");
        review.textContent = local ? "当前为本地体验。你可以关闭此窗口检查阅读预览，或下载 Markdown 草稿。" : "页面已切换为阅读预览。确认后，正文、导语和批注将提交到 GitHub，并在网站构建成功后公开。";
      }
    } catch (e) { notify((e as Error).message); }
    finally { busy = false; renderState(); }
  });
  dialog.querySelector("form")!.addEventListener("submit", async event => {
    event.preventDefault(); if (busy || !surface) return;
    error.textContent = "";
    try {
      if (mode === "exit") { await exit(); return; }
      if (mode === "intro") { intro = input.value.trim(); renderState(); dialog.close(); return; }
      if (mode === "note") { surface.saveNote(input.value, noteId); renderState(); if (previewing) await setPreview(true); dialog.close(); return; }
      if (mode === "delete") { surface.deleteNote(noteId!); renderState(); if (previewing) await setPreview(true); manageNotes(); return; }
      if (mode === "notes") { dialog.close(); return; }
      if (mode !== "review" || local) return;
      validatedDraft(); busy = true; apply.disabled = true;
      const result = await request(`/publish?slug=${slug}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ format: "markdown-v1", sha, markdown: draftBody(), ...(intro !== initialIntro ? { intro } : {}) }) });
      bodyDirty = false; intro = initialIntro; editing = false; session = "";
      await surface.destroy(); surface = undefined; host.hidden = true;
      bar.hidden = true; dialog.close(); entry.hidden = false; entry.disabled = true; entry.textContent = "修改已提交，等待网站更新";
      document.body.classList.remove("author-editing"); notify(result.message);
      const link = document.createElement("a"); link.href = "https://github.com/yuzheng310/yuzheng310.github.io/actions"; link.textContent = "查看发布状态 ↗"; link.target = "_blank"; link.rel = "noopener noreferrer"; entry.after(link);
    } catch (e) { error.textContent = (e as Error).message; }
    finally { busy = false; apply.disabled = mode === "review" && local; }
  });
  window.addEventListener("beforeunload", event => { if (dirty()) event.preventDefault(); });
}
