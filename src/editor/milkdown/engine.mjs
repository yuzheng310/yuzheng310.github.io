import { CrepeBuilder } from '@milkdown/crepe/builder';
import { codeMirror } from '@milkdown/crepe/feature/code-mirror';
import { latex } from '@milkdown/crepe/feature/latex';
import { toolbar } from '@milkdown/crepe/feature/toolbar';
import { linkTooltip } from '@milkdown/crepe/feature/link-tooltip';
import { editorViewCtx, parserCtx, serializerCtx } from '@milkdown/kit/core';
import { uploadConfig } from '@milkdown/kit/plugin/upload';
import { undo, redo } from '@milkdown/kit/prose/history';
import { $view } from '@milkdown/kit/utils';
import { imageSchema, htmlSchema } from '@milkdown/kit/preset/commonmark';
import { annotationMarkdown, highlightSchema, addNote, updateNote, deleteNote, listNotes, toggleHighlight } from './annotations.mjs';
import { assertRoundTrip } from '../../lib/author-document.mjs';
import { normalizeNote } from '../../lib/author-markdown.mjs';

export async function createArticleEditor({ root, markdown, onChange = () => {}, onNote = () => {}, imageView, htmlView }) {
  const editor = new CrepeBuilder({ root, defaultValue: markdown })
    .addFeature(codeMirror, { previewToggleText: preview => preview ? '编辑公式' : '查看效果' })
    .addFeature(latex)
    .addFeature(linkTooltip, { inputPlaceholder: '链接地址' })
    .addFeature(toolbar, {
      boldLabel: '粗体', italicLabel: '斜体', codeLabel: '行内代码', linkLabel: '链接', strikethroughLabel: '删除线', latexLabel: '行内公式',
      buildToolbar(builder) {
        builder.addGroup('author', '作者注释')
          .addItem('highlight', { icon: '<span class="author-tool-text">高亮</span>', label: '高亮或取消高亮', active: () => false, onRun: ctx => toggleHighlight(ctx.get(editorViewCtx)) })
          .addItem('note', { icon: '<span class="author-tool-text">批注</span>', label: '添加公开批注', active: () => false, onRun: () => onNote() });
      },
    });
  // Local blob uploads cannot survive publishing. Images remain repository URLs.
  editor.editor.config(ctx => ctx.update(uploadConfig.key, prev => ({ ...prev, uploader: async () => [] }))).use(annotationMarkdown).use(highlightSchema);
  // CommonMark represents an omitted image title as null; this upstream schema
  // version requires strings. Normalize only these optional attributes.
  editor.editor.use(imageSchema.extendSchema(prev => ctx => ({
    ...prev(ctx),
    parseMarkdown: {
      match: node => node.type === 'image',
      runner: (state, node, type) => state.addNode(type, { src: node.url, alt: node.alt || '', title: node.title || '' }),
    },
  })));
  if (imageView) editor.editor.use($view(imageSchema.node, () => node => imageView(node)));
  if (htmlView) editor.editor.use($view(htmlSchema.node, () => node => htmlView(node)));
  let ready = false;
  editor.on(api => api.updated(() => { if (ready) onChange(); }));
  try {
    await editor.create();
    const baseline = editor.getMarkdown();
    assertRoundTrip(markdown, baseline);
    ready = true;
    const action = fn => editor.editor.action(ctx => fn(ctx.get(editorViewCtx), ctx));
    return {
      getMarkdown: () => editor.getMarkdown(), baseline,
      changed: () => editor.getMarkdown() !== baseline,
      destroy: () => editor.destroy(),
      setReadonly: value => editor.setReadonly(value),
      undo: () => action(view => undo(view.state, view.dispatch)),
      redo: () => action(view => redo(view.state, view.dispatch)),
      highlight: () => action(view => toggleHighlight(view)),
      selection: () => action(view => ({ from: view.state.selection.from, to: view.state.selection.to, text: view.state.doc.textBetween(view.state.selection.from, view.state.selection.to, '\n') })),
      notes: () => action((view, ctx) => listNotes(view.state.doc).map(note => ({
        id: note.id,
        markdown: ctx.get(serializerCtx)(ctx.get(parserCtx)('占位').copy(note.node.content)).replace(/^作者批注[：:]\s*/, '').trim(),
      }))),
      saveNote: (value, id) => action((view, ctx) => {
        const normalized = normalizeNote(value);
        if (!normalized.trim()) throw new Error('请填写批注内容。');
        if (normalized.length > 10000) throw new Error('批注最多 10000 字。');
        const doc = ctx.get(parserCtx)(normalized);
        return id ? updateNote(view, id, doc.content) : addNote(view, doc.content);
      }),
      deleteNote: id => action(view => deleteNote(view, id)),
      // Tests use the real view and transactions; consumers use the public API.
      action,
    };
  } catch (error) { if (editor.editor.status === 'Created') await editor.destroy(); throw error; }
}
