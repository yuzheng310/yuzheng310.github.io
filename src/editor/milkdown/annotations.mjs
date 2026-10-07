import { $markSchema, $remark } from '@milkdown/kit/utils';
import { closeHistory } from '@milkdown/kit/prose/history';
import { remarkPublicAnnotations } from '../../lib/author-document.mjs';

export const annotationMarkdown = $remark('publicAnnotations', () => remarkPublicAnnotations);
export const highlightSchema = $markSchema('author_highlight', () => ({
  attrs: { note: { default: '' } },
  inclusive: false,
  parseDOM: [{ tag: 'mark.author-highlight', getAttrs: dom => ({ note: /^author-\d+$/.test(dom.dataset.authorNote || '') ? dom.dataset.authorNote : '' }) }],
  toDOM: mark => ['mark', { class: 'author-highlight', ...(mark.attrs.note ? { 'data-author-note': mark.attrs.note } : {}) }, 0],
  parseMarkdown: {
    match: node => node.type === 'authorHighlight',
    runner: (state, node, type) => { state.openMark(type, { note: node.note || '' }); state.next(node.children); state.closeMark(type); },
  },
  toMarkdown: {
    match: mark => mark.type.name === 'author_highlight',
    runner: (state, mark) => { state.withMark(mark, 'authorHighlight', undefined, { note: mark.attrs.note }); },
  },
}));

export function listNotes(doc) {
  const notes = [];
  doc.descendants((node, pos) => {
    if (node.type.name === 'footnote_definition' && /^author-\d+$/.test(node.attrs.label)) {
      notes.push({ id: node.attrs.label, node, pos }); return false;
    }
  });
  return notes;
}

export function toggleHighlight(view) {
  const { state } = view, { from, to, empty } = state.selection;
  if (empty) return;
  const type = state.schema.marks.author_highlight;
  let hasPlain = false;
  state.doc.nodesBetween(from, to, node => { if (node.marks.some(mark => mark.type === type && !mark.attrs.note)) hasPlain = true; });
  const tr = closeHistory(state.tr);
  state.doc.nodesBetween(from, to, (node, pos) => {
    if (!node.isInline || node.marks.some(mark => mark.type === type && mark.attrs.note)) return;
    if (hasPlain) tr.removeMark(Math.max(from, pos), Math.min(to, pos + node.nodeSize), type);
    else tr.addMark(Math.max(from, pos), Math.min(to, pos + node.nodeSize), type.create());
  });
  view.dispatch(tr); view.focus();
}

export function addNote(view, content) {
  const { state } = view, { from, to, empty } = state.selection;
  if (empty || !state.doc.textBetween(from, to).trim()) throw new Error('请先选中需要批注的正文。');
  let overlaps = false;
  state.doc.nodesBetween(from, to, node => {
    if (node.type.name === 'footnote_definition' || node.type.name === 'footnote_reference' || node.marks.some(mark => mark.type.name === 'author_highlight' && mark.attrs.note)) overlaps = true;
  });
  if (overlaps) throw new Error('选区包含已有批注，请在“管理批注”中修改，或选择未批注的正文。');
  const used = new Set(listNotes(state.doc).map(note => note.id));
  let number = 1; while (used.has(`author-${number}`)) number++;
  const id = `author-${number}`;
  const mark = state.schema.marks.author_highlight.create({ note: id });
  const tr = closeHistory(state.tr).addMark(from, to, mark);
  tr.insert(to, state.schema.nodes.footnote_reference.create({ label: id }));
  tr.insert(tr.doc.content.size, state.schema.nodes.footnote_definition.create({ label: id }, content));
  view.dispatch(tr); view.focus();
  return id;
}

export function updateNote(view, id, content) {
  const note = listNotes(view.state.doc).find(note => note.id === id);
  if (!note) throw new Error('批注已不存在。');
  view.dispatch(closeHistory(view.state.tr).replaceWith(note.pos + 1, note.pos + note.node.nodeSize - 1, content));
}

export function deleteNote(view, id) {
  const { state } = view, tr = closeHistory(state.tr), removals = [];
  state.doc.descendants((node, pos) => {
    if ((node.type.name === 'footnote_definition' || node.type.name === 'footnote_reference') && node.attrs.label === id) {
      removals.push([pos, pos + node.nodeSize]); return false;
    }
    for (const mark of node.marks) if (mark.type.name === 'author_highlight' && mark.attrs.note === id) tr.removeMark(pos, pos + node.nodeSize, mark);
  });
  for (const [from, to] of removals.sort((a, b) => b[0] - a[0])) tr.delete(from, to);
  view.dispatch(tr);
}
