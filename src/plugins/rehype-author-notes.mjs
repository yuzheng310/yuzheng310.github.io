const element = (tagName, properties={}, children=[]) => ({type:'element',tagName,properties,children});
const text = value => ({type:'text',value});
const plain = node => node.type==='text' ? node.value : (node.children||[]).map(plain).join('');
const visit = (node,fn,parents=[]) => { fn(node,parents); for(const child of [...node.children||[]]) visit(child,fn,[...parents,node]); };

// Turn only author notes into nearby, native disclosure cards. Ordinary citations
// retain the site's footnote behavior, and old links continue to resolve.
export default function rehypeAuthorNotes() {
  return tree => {
    const notes = new Map(), refs=[];
    visit(tree,(node,parents) => {
      const id=node.properties?.id;
      if(node.tagName==='li' && /^user-content-fn-author-\d+$/.test(id||'')) notes.set(id,{node,parents});
      if(node.tagName==='a' && /^#user-content-fn-author-\d+$/.test(node.properties?.href||'')) refs.push({node,parents});
    });
    for(const [id,entry] of notes) {
      const references=refs.filter(ref=>ref.node.properties.href===`#${id}`);
      if(!references.length) continue;
      const first=references[0], number=id.split('-').at(-1);
      const meaningful=entry.node.children.filter(n=>n.type!=='text'||n.value.trim());
      visit(entry.node,node=>{ if(node.children) node.children=node.children.filter(child=>!Object.hasOwn(child.properties||{},'dataFootnoteBackref')); });
      const firstParagraph=meaningful.find(n=>n.tagName==='p');
      if(firstParagraph?.children[0]?.type==='text') firstParagraph.children[0].value=firstParagraph.children[0].value.replace(/^作者批注[：:]\s*/, '');
      const bodyChildren=entry.node.children.filter(n=>n.tagName!=='p'||plain(n).trim()||n.children.some(c=>c.type==='element'));
      const content=element('div',{className:['author-note-body']},bodyChildren);
      const summary=element('summary',{},[element('span',{className:['author-note-label']},[text(`作者批注 ${number}`)]),element('span',{className:['author-note-hint']},[text('展开阅读')])]);
      const card=element('details',{id,className:['author-note-card'],'data-author-note-card':`author-${number}`},[summary,content]);
      // Place a list note after the list, so long mathematics has the full reading width.
      const list=[...first.parents].reverse().find(n=>['ul','ol'].includes(n.tagName));
      const block=list || [...first.parents].reverse().find(n=>['p','blockquote','div'].includes(n.tagName));
      let parent;
      visit(tree,node=>{if(node.children?.includes(block)) parent=node;});
      if(!parent) continue;
      let index=parent.children.indexOf(block)+1;
      while(parent.children[index]?.properties?.className?.includes('author-note-card')) index++;
      parent.children.splice(index,0,card);
      for(const ref of references) {
        ref.node.children=[text(`批注 ${number}`)];
        ref.node.properties={...ref.node.properties,className:['author-note-ref'],'aria-controls':id,'aria-label':`阅读作者批注 ${number}`};
        delete ref.node.properties.ariaDescribedBy;
        const local=ref.parents.at(-1)?.tagName==='sup' ? ref.parents.at(-2) : ref.parents.at(-1);
        visit(local||block,node=>{if(node.tagName==='mark' && !node.properties['data-author-note'] && !node.properties.dataAuthorNote) node.properties['data-author-note']=`author-${number}`;});
      }
      const oldParent=entry.parents.at(-1); oldParent.children=oldParent.children.filter(n=>n!==entry.node);
    }
    // Remove an empty author-only footnote section, but never ordinary citations.
    visit(tree,node=>{
      if(node.children) node.children=node.children.filter(child=>{
        if(child.tagName!=='section'||!Object.hasOwn(child.properties||{},'dataFootnotes')) return true;
        const list=child.children?.find(n=>n.tagName==='ol');
        return list?.children?.some(n=>n.tagName==='li');
      });
    });
  };
}
