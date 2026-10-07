export type Piece = { span: HTMLElement; start: number; end: number; quote: string; label: string };
export type SelectionTarget = { block: HTMLElement; pieces: Piece[]; start: number; end: number; quote: string; selection: string };
export function collectSelection(range: Range, source: string): SelectionTarget | undefined {
  const parent = (node: Node) => node.nodeType === Node.TEXT_NODE ? node.parentElement : node as Element;
  const block = parent(range.startContainer)?.closest<HTMLElement>('.prose [data-author-block]');
  if (!block || block !== parent(range.endContainer)?.closest('.prose [data-author-block]')) throw new Error('请在同一段落内选择；可以跨越粗体、链接和完整公式。');
  if (block.dataset.authorDirty === 'true') throw new Error('这段已有未发布修改，可先撤销再调整，或发布后继续编辑。');
  const pieces: Piece[] = [];
  for (const span of block.querySelectorAll<HTMLElement>('[data-author-start]')) {
    if (!range.intersectsNode(span)) continue;
    const full=document.createRange(); full.selectNodeContents(span);
    const part=range.cloneRange();
    if (range.compareBoundaryPoints(Range.START_TO_START,full)<0) part.setStart(full.startContainer,full.startOffset);
    if (range.compareBoundaryPoints(Range.END_TO_END,full)>0) part.setEnd(full.endContainer,full.endOffset);
    if (part.collapsed || !part.toString()) continue;
    const atomic=span.dataset.authorAtomic==='true';
    let start=Number(span.dataset.authorStart), end=Number(span.dataset.authorEnd);
    if (atomic) {
      if (range.compareBoundaryPoints(Range.START_TO_START,full)>0 || range.compareBoundaryPoints(Range.END_TO_END,full)<0) throw new Error('公式和行内代码请完整选中，也可以连同整段一起选择。');
    } else {
      const prefix=document.createRange();prefix.selectNodeContents(span);prefix.setEnd(part.startContainer,part.startOffset);
      start += prefix.toString().length; end=start+part.toString().length;
      if(source.slice(start,end)!==part.toString()) throw new Error('选区已变化，请重新选择。');
    }
    pieces.push({span,start,end,quote:source.slice(start,end),label:atomic?(span.dataset.authorLabel||part.toString()):part.toString()});
  }
  if (!pieces.length || !pieces.some(piece=>piece.label.trim())) return;
  const start=pieces[0].start,end=pieces[pieces.length-1].end;
  return {block,pieces,start,end,quote:source.slice(start,end),selection:pieces.map(piece=>piece.label).join('')};
}
