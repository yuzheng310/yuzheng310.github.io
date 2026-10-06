"""Import the workspace's HTML translations without changing their source files.
Run: uv run --with beautifulsoup4 python scripts/import-translations.py /path/to/workspace
"""
from pathlib import Path
import base64
import json
import re
import shutil
import sys
from bs4 import BeautifulSoup

root = Path(sys.argv[1]).resolve()
site = Path(__file__).resolve().parents[1]
asset_map = {item['url']: item['path'] for item in json.loads((site / 'scripts/translation-assets.json').read_text()) if 'path' in item}
(site / 'src/documents/translations').mkdir(parents=True, exist_ok=True)
entries = [
 ('cuda-matmul', 'cuda-mmm-annotated.html', 'CUDA 矩阵乘法优化：从朴素 Kernel 到接近 cuBLAS', '沿 Simon Boehm 的工作日志，理解合并访存、共享内存、寄存器分块与线程束级优化。', ['翻译','CUDA','GPU 优化','矩阵乘法'], 'https://siboehm.com/articles/22/CUDA-MMM', 'Simon Boehm', '译文与学习注释'),
 ('llm-inference-explained', 'llm-inference-annotated.html', '图解大模型推理：从 Prefill 到 Decode', '沿分词、注意力、KV Cache 和解码的执行路径，理解推理延迟与吞吐瓶颈。', ['翻译','大模型推理','KV Cache','性能分析'], 'https://x.com/_avichawla', 'Avi Chawla', '译文与学习注释；来源文件仅保留作者主页，原文章直链待补'),
 ('cuda-from-zero-to-hero', 'cuda_from_zero_to_hero_zh.html', 'CUDA 从零到精通 #1', '从 GPU 硬件、线程层次与内存结构出发，理解 CUDA 编程的基本执行模型。', ['翻译','CUDA','GPU 架构'], 'https://x.com/goyal__pramod/status/2103565642800431533', 'Pramod Goyal', '中文译文'),
 ('ultrascale-playbook', 'ultrascale_playbook_zh.html', 'Ultra-Scale Playbook：在 GPU 集群上训练大语言模型', '覆盖显存核算、数据并行、张量并行、流水线并行、上下文并行与 GPU 性能分析。', ['翻译','分布式训练','并行计算','GPU 优化'], 'https://huggingface.co/spaces/nanotron/ultrascale-playbook', 'Hugging Face / Nanotron', '中文翻译与精解'),
 ('alisa-llm-infrastructure', 'alisa_book_llms_infra_zh.html', 'Alisa’s Book of LLMs：大模型 Infra 核心章节', '围绕 Transformer 计算核算、推理、数值精度、GPU 与分布式并行整理的中文学习材料。', ['翻译','Transformer','大模型推理','分布式训练'], 'https://alisawuffles.notion.site/alisa-s-book-of-llms', 'Alisa Liu', 'Infra 相关章节的中文翻译与精解'),
 ('mooncake', 'mooncake_paper_translation.html', 'Mooncake：以 KV Cache 为中心的解耦推理架构', '围绕 Prefill / Decode 分离、分布式 KV Cache 存储池与请求调度的论文中文精读。', ['翻译','论文','KV Cache','分离式推理'], 'https://arxiv.org/abs/2407.00079', 'Ruoyu Qin 等', '论文中文精读，包含整理者的解释与延伸，并非逐句全译'),
 ('continuum', 'continuum_paper_translation.html', 'Continuum：多轮 Agent 调度与 KV Cache TTL', '理解工具调用期间的 KV 驻留、多轮任务连续性，以及 TTL 与调度策略之间的取舍。', ['翻译','论文','Agent','KV Cache'], 'https://arxiv.org/abs/2511.02230', 'Hanchen Li 等', '论文中文精读，包含整理者的解释与延伸，并非逐句全译'),
 ('py-kvcache', 'personnalinfra/docs/py-kvcache-study/translation-zh.html', 'py-kvcache：基于 NVMe SSD 的外部 KV 缓存', '论文 §5、§6.4–6.6、§7–8 选章中译，保留双语对照、原始图表、实验条件与 MathML 公式。', ['翻译','论文','KV Cache','NVMe','vLLM'], 'https://arxiv.org/abs/2609.11744v1', 'Joseph Kanichai、Tiziano De Matteis、Animesh Trivedi', '论文选章翻译：§5、§6.4–6.6、§7–8；中文整理于 2026-09-24'),
]

for slug, filename, title, description, tags, source, author, scope in entries:
    path = root / filename
    text = path.read_text()
    # Repair source-generation escape damage, rather than removing formula symbols.
    text = text.replace('\x07lpha', r'\alpha').replace('\x08eta', r'\beta')
    if slug == 'continuum':
        text = text.replace('ICLR 2025', 'arXiv 预印本').replace('$\nightarrow$', r'$\rightarrow$')
    soup = BeautifulSoup(text, 'html.parser')
    for node in list(soup.find_all(string=True)):
        if node.find_parent(['script', 'style', 'pre', 'code']): continue
        repaired = re.sub(r'\$\$[\s\S]*?\$\$|(?<!\$)\$[^$]+\$(?!\$)', lambda match: match.group(0).replace('\t', r'\t').replace('\nightarrow', r'\rightarrow'), str(node))
        if repaired != str(node): node.replace_with(repaired)
    assets = site / 'public' / 'translations' / slug
    assets.mkdir(parents=True, exist_ok=True)
    # Extract inline images unchanged, so pages do not carry large base64 payloads.
    for index, img in enumerate(soup.find_all('img')):
        src = img.get('src', '')
        if src in asset_map:
            img['src'] = asset_map[src]
            img.attrs.pop('onerror', None)
        if src.startswith('data:image/'):
            mime, content = src.split(';base64,', 1)
            ext = mime.split('/')[-1].replace('jpeg', 'jpg')
            name = f'figure-{index + 1}.{ext}'
            (assets / name).write_bytes(base64.b64decode(content))
            img['src'] = f'/translations/{slug}/{name}'
        img['loading'] = 'lazy'
        img['decoding'] = 'async'
    # Source documents contain local authoring tools; the public page is a reader.
    if slug in ('cuda-matmul', 'llm-inference-explained'):
        for node in soup.select('.top-nav, #selection-popover, #comment-modal, #annotation-drawer'):
            node.decompose()
        for node in soup.find_all('script'):
            node.decompose()
        for node in soup.select('[contenteditable]'):
            del node['contenteditable']
        script = soup.new_tag('script', src='/translations/reader.js', defer=True)
        soup.body.append(script)
        style = soup.new_tag('style')
        style.string = 'body{padding-top:0!important}.page-layout,.page-wrapper{margin-top:32px!important}'
        soup.head.append(style)
    if slug == 'cuda-from-zero-to-hero':
        for node in soup.find_all('script', src=True):
            if 'tailwind' in node['src']:
                node.decompose()
        soup.head.append(soup.new_tag('link', rel='stylesheet', href='/translations/cuda-from-zero-to-hero/tailwind.css'))
    if slug == 'ultrascale-playbook':
        shutil.copyfile(root / 'ultra-cheatsheet.svg', assets / 'ultra-cheatsheet.svg')
        for img in soup.find_all('img', src='ultra-cheatsheet.svg'):
            img['src'] = f'/translations/{slug}/ultra-cheatsheet.svg'
    if slug == 'py-kvcache':
        for node in soup.find_all(['a','img']):
            attr = 'src' if node.name == 'img' else 'href'
            value = node.get(attr, '')
            if value.startswith(('translated-assets/', 'original/latex/images/')):
                local = path.parent / value
                if not local.is_file(): raise FileNotFoundError(local)
                target = assets / local.name
                shutil.copyfile(local, target)
                node[attr] = f'/translations/{slug}/{target.name}'
            elif value.startswith('original/paper-offline.html'):
                node[attr] = 'https://arxiv.org/html/2609.11744v1' + value.removeprefix('original/paper-offline.html')
            elif value == 'original/latex/main.tex' or value == 'original/2609.11744v1-source.tar.gz':
                node[attr] = 'https://arxiv.org/src/2609.11744v1'
            elif value == 'index.html':
                node[attr] = '/blog/'
        for text_node in soup.find_all(string=True):
            if '依据你提供的本地 LaTeX 源码' in str(text_node):
                text_node.replace_with(str(text_node).replace('依据你提供的本地 LaTeX 源码', '依据论文 LaTeX 源码').replace('本地英文全文','英文全文'))
    # Separate private project coaching from the paper's own claims.
    if slug in ('continuum', 'mooncake'):
        section = soup.select_one('#section-connection')
        if section:
            number = '7' if slug == 'continuum' else '8'
            section.clear()
            section.append(BeautifulSoup(f'<h2>{number}. 个人思考：与会话 KV 管理的联系</h2><p>以下是阅读后的个人思考，不属于论文原文。</p><p>在 Qwen3 Runtime 中，我关注工具等待期间的会话状态保存、前缀校验与增量恢复。只有前缀匹配且缓存有效时，后续请求才能复用历史 KV；显存压力、回收策略和传输成本都会影响收益。</p><p>这与论文讨论的缓存生命周期问题有关，但实现范围和评测条件不同。Qwen3 Runtime 面向单 GPU 的 Agentic Rollout 推理层；论文中的集群架构、调度算法与性能结果不能直接视为该项目已实现或已验证的能力。</p><p>具体实现与实验边界见 <a href="/projects/qwen3-runtime/">Qwen3 Runtime 项目记录</a>。</p>', 'html.parser'))
            link = soup.select_one('a[href="#section-connection"]')
            if link: link.string = f'{number}. 个人思考：与会话 KV 管理的联系'
        for heading in soup.find_all('h3'):
            if '与我们项目设计的惊人一致性' in heading.get_text(): heading.string = '系统核心机制'
        for paragraph in soup.find_all(['p', 'li']):
            if '这完全等价于你代码中' in paragraph.get_text():
                paragraph.clear()
                paragraph.append('运行时按 TTL 显式保留工具调用期间的 KV 状态，在租约到期后允许回收。')
        for node in soup.find_all(string=True):
            text = str(node)
            if '这完全等价于你代码中' in text:
                node.replace_with(text[:text.index('这完全等价于你代码中')])
            elif '权威中文精读' in text:
                node.replace_with(text.replace('权威中文精读', '中文精读'))
    # Bundle math/code dependencies locally; the full HTML layout is retained.
    for node in soup.find_all(['script','link']):
        attr = 'src' if node.name == 'script' else 'href'
        url = node.get(attr, '')
        url = url.replace('https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/', '/vendor/katex/')
        url = url.replace('https://cdn.jsdelivr.net/npm/prismjs@1.29.0/', '/vendor/prism/')
        if url: node[attr] = url
    if slug in ('cuda-matmul', 'llm-inference-explained', 'cuda-from-zero-to-hero', 'continuum', 'mooncake'):
        soup.head.append(soup.new_tag('link', rel='stylesheet', href='/vendor/katex/katex.min.css'))
        for src in ['/vendor/katex/katex.min.js','/vendor/katex/contrib/auto-render.min.js','/translations/math.js']:
            soup.body.append(soup.new_tag('script', src=src, defer=True))
    (site / 'src/documents/translations' / f'{slug}.html').write_text(str(soup))
    data = {'title':title,'description':description,'date':'2026-10-06','tags':tags,'htmlFile':slug+'.html','sourceURL':source,'sourceAuthor':author,'translationScope':scope}
    frontmatter='---\n'+'\n'.join(f'{k}: {json.dumps(v,ensure_ascii=False)}' for k,v in data.items())+'\n---\n'
    (site / 'src/content/blog' / f'{slug}.md').write_text(frontmatter)
    print(f'Imported {slug}: {len(soup.find_all("img"))} images')
