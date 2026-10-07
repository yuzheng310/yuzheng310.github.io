// Shared by the browser and publishing service. All edits address the original
// Markdown, so unrelated formatting, code and equations remain byte-for-byte intact.
export function escapeText(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/([\\`*_{}[\]()#+.!|~$^-])/g, '\\$1').replace(/\r?\n/g, ' ');
}
export function validateEdits(source, edits) {
  if (!Array.isArray(edits) || edits.length > 100) throw new Error('一次最多发布 100 处修改。');
  let end = -1;
  for (const edit of [...edits].sort((a, b) => a.start - b.start)) {
    if (!edit || !Number.isInteger(edit.start) || !Number.isInteger(edit.end) || edit.start < 0 || edit.end <= edit.start || edit.end > source.length || edit.start < end) throw new Error('修改范围重叠或已经失效，请重新选择。');
    if (typeof edit.quote !== 'string' || source.slice(edit.start, edit.end) !== edit.quote) throw new Error('原文已发生变化，请重新载入文章。');
    if (!['replace', 'highlight', 'note'].includes(edit.kind)) throw new Error('不支持的修改类型。');
    if (typeof edit.value !== 'string' || edit.value.length > 10000 || (edit.kind === 'note' && !edit.value.trim())) throw new Error('修改内容无效。');
    end = edit.end;
  }
}
export function applyEdits(source, edits, intro) {
  validateEdits(source, edits);
  let result = source;
  const notes = [];
  const existing = new Set([...source.matchAll(/\[\^(author-\d+)\]/g)].map(match => match[1]));
  let counter = 1;
  for (const edit of [...edits].sort((a, b) => b.start - a.start)) {
    let replacement;
    if (edit.kind === 'replace') replacement = escapeText(edit.value);
    else if (edit.kind === 'highlight') replacement = `<mark class="author-highlight">${edit.quote}</mark>`;
    else {
      while (existing.has(`author-${counter}`)) counter++;
      const id = `author-${counter++}`;
      existing.add(id);
      replacement = `<mark class="author-highlight">${edit.quote}</mark>[^${id}]`;
      notes.push(`[^${id}]: 作者批注：${escapeText(edit.value.trim())}`);
    }
    result = result.slice(0, edit.start) + replacement + result.slice(edit.end);
  }
  if (notes.length) result = result.trimEnd() + '\n\n' + notes.reverse().join('\n\n') + '\n';
  if (intro !== undefined) {
    if (typeof intro !== 'string' || intro.length > 3000) throw new Error('导语最多 3000 字。');
    const frontmatter = result.match(/^---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/);
    if (!frontmatter) throw new Error('文章缺少有效的元信息。');
    const header = frontmatter[0];
    // This field is maintained as a JSON-quoted YAML scalar. Refuse to overwrite
    // an unfamiliar multiline field rather than corrupting its continuation lines.
    const current = header.match(/^intro:([^\r\n]*)/m);
    if (current && !/^\s*"(?:[^"\\]|\\.)*"\s*$/.test(current[1])) throw new Error('导语格式需要先转换为单行字符串。');
    const field = `intro: ${JSON.stringify(intro.trim())}`;
    const next = current ? header.replace(/^intro:[^\r\n]*/m, () => field) : header.replace(/\r?\n---(?:\r?\n|$)$/, match => '\n' + field + match);
    result = next + result.slice(header.length);
  }
  return result;
}
