# 原位编辑器候选调研

调研日期：2026-10-07。范围：Vditor、Milkdown/Crepe、Tiptap、CloudCannon、TinaCMS、Keystatic、CKEditor 5、TinyMCE、MDXEditor、BlockNote、Hypothesis。核对官方文档/官方仓库和本站代码；尝试打开官方演示，但没有完成候选编辑器的运行验证，没有安装或修改网站应用。下文「事实」来自文档，「判断」是面向本站的选型推断，「待验证」不能视为已实现。

## 结论

**优先验证 Vditor 与 Milkdown/Crepe，Tiptap 作为深度定制备选；若更愿意付费换取完整 CMS，则考察 CloudCannon。** 当前证据不足以宣布任何方案开箱满足全部需求。用户需要的是“真实页面内编辑 + 公开作者批注 + Markdown 技术文章 + 仅本人可写”，不能只按工具栏是否漂亮排名。

Milkdown/Crepe 以 Markdown 为中心，又已有选区工具栏、列表、代码块、行内/块级数学支持，核心为 MIT。它能减少编辑器基础工作，但公开批注、本站 HTML 标记和脚注仍需扩展及往返验证，不能称为即插即用。[Crepe API](https://milkdown.dev/docs/api/crepe)、[官方仓库](https://github.com/Milkdown/milkdown)、[转换 API](https://milkdown.dev/docs/api/transformer)

**CKEditor 5 是愿意接受商业插件及 HTML 数据适配时的成熟备选**；**TinyMCE 的 inline + 官方 Comments 组合存在明确功能冲突**；**BlockNote 更适合接受 JSON 为主存储的块编辑器项目**；**MDXEditor 值得用于 Markdown 编辑，但数学和本站公开批注需进一步实证**。这些是下面事实支持的选型判断，不是性能测试排名。

## 本站需要保留的内容

本地核查：`package.json` 当前是 Astro + remark-gfm + remark-math + rehype-katex；`src/content/blog/llm-inference-parallelism.md` 包含 `<mark class="author-highlight">`、`[^author-1]` 作者脚注以及脚注内多段正文和块级数学。`src/editor/` 与 `server/editor/` 已有客户端和 Worker 保存服务。因此验证对象必须是**本站真实 Markdown**，不能仅看一个编辑器空白示例。

「能输出 Markdown」不等于「能保留本站全部 Markdown」。尤其不能把渲染后的 KaTeX DOM 再直接当作数学源码编辑。

现有 Worker 保存接口处理选区和段落补丁，不是通用全文保存接口。复用现有身份认证与 GitHub 发布机制是可行的架构方向，但接入整篇编辑器需要新增或改造保存协议，并保留版本冲突检测、路径限制和服务端身份校验。下文“接现有 Worker”均指此类适配，不代表直接发送 HTML/JSON 就能保存。

本地选区逻辑要求选区两端属于同一个 `data-author-block`，后端段落修改也有相应边界，这解释了跨多个列表项时工具栏消失。成熟编辑器能提供连续文档的选区模型，但仍须实际测试高亮和作者批注跨块的行为。

## 候选对比

### Vditor

- **事实：** 提供所见即所得、即时渲染与分屏 Markdown 模式；支持数学、代码、脚注等。MIT 许可，提供普通 JavaScript 接入，不必为编辑器引入完整 React 应用。[官方介绍](https://b3log.org/vditor/)、[API 与功能清单](https://github.com/Vanessa219/vditor/blob/master/README_en_US.md)、[许可证](https://github.com/Vanessa219/vditor/blob/master/LICENSE)
- **事实：** 官方有“划词评论”演示，提供评论 ID、添加/移除回调以及关联选区高亮 API，**仅支持 wysiwyg 模式**。演示自行创建评论 UI 并使用 localStorage，不是生产级 GitHub 保存或公开发布系统。[划词评论演示及源码](https://b3log.org/vditor/demo/advanced-comment.html)
- **判断：** 与本站需求接近，值得优先做小规模验证；它减少基础编辑和选区评论交互的开发，但仍需调整正文样式、公开作者批注的数据表示、保存及重新加载。
- **待验证：** 原生评论锚点是否能经 Markdown 导出/导入稳定恢复；跨列表批注、现有 mark 与脚注结构、公式和自定义 HTML 是否保留。不能把评论演示当成无损保存证明。应使用本站渲染器核对公开文章，不能用编辑器预览代替发布结果。

### Tiptap

- **事实：** 提供选区浮动菜单、高亮、行内/块级数学扩展；可以定制 UI，但并非完整 CMS。[Bubble Menu](https://tiptap.dev/docs/editor/extensions/functionality/bubble-menu)、[Highlight](https://tiptap.dev/docs/editor/extensions/marks/highlight)、[Mathematics](https://tiptap.dev/docs/editor/extensions/nodes/mathematics)
- **事实：** Markdown 扩展当前标注 Beta，并明确 Comments 暂不支持 Markdown 往返；用 Markdown 替换带批注的文档可能丢失批注。[Markdown 限制](https://tiptap.dev/docs/editor/markdown)
- **事实：** 官方 Comments 安装说明要求 Start 计划和 Document server；开源编辑器核心与商业评论服务需区分。[Comments 安装](https://tiptap.dev/docs/comments/getting-started/install)、[功能及许可对照](https://tiptap.dev/feature-comparison)
- **事实：** 严格 schema 会过滤未声明的内容或属性，因此任意本站 HTML 不会天然完整保留。[Schema](https://tiptap.dev/docs/editor/core-concepts/schema)
- **判断：** 可实现很贴合网站的原位体验，但作者公开批注、已有 Markdown 扩展和可靠导出仍需开发；不适合宣称为最省开发的一键替换。自定义作者注并不一定需要购买协作 Comments，但也不能省略对应开发成本。

### CloudCannon

- **事实：** 完整 Git CMS，支持页面预览中的原位富文本、纯文本及图片编辑，有 Astro 集成指南。[Visual Editing](https://cloudcannon.com/documentation/guides/set-up-visual-editing/)、[Astro 指南](https://docs.astro.build/en/guides/cms/cloudcannon/)
- **事实：** Standard 价格为每月 55 美元，年付折合每月 49 美元，含 Git 同步和可视化编辑。10 美元 Lite 为 Partners Only，不应作为普通个人用户的默认价格。[价格](https://cloudcannon.com/pricing/)
- **事实：** 富文本配置对 custom markup 提醒：不能重建的标记可能在编辑时被移除。允许自定义标记不等于任意内容无损。[Block editable 配置](https://cloudcannon.com/documentation/developer-reference/configuration-file/types/_editables/block/)
- **判断：** 本次候选中较接近“购买完整成品管理体验”的方向；但没有核实到本站这种公开作者批注、数学和图像切换的开箱解决方案。订阅费用与站点适配成本需一起考虑，不能因为收费便认为无须开发。

### TinaCMS / Keystatic

- **事实：** Tina 的 `tinaField` 将网页元素点击关联到侧栏字段；它的 Astro 可视化预览要求支持服务端渲染的 adapter/动态预览端点，即便配置静态输出也有此要求。这与“直接在原文光标处编辑”的体验有区别。[TinaField](https://tina.io/docs/contextual-editing/tinafield)、[Astro 预览](https://tina.io/docs/contextual-editing/astro)
- **事实：** Tina 富文本已有格式与高亮配置，不应误称没有高亮能力。[Rich Text](https://tina.io/docs/reference/types/rich-text)
- **事实：** Keystatic GitHub 模式要求可运行其 API 的服务端；仓库写权限决定可编辑成员，并不天然只限仓库所有者一人。[GitHub 模式](https://keystatic.com/docs/github-mode)、[Astro 指南](https://docs.astro.build/en/guides/cms/keystatic/)
- **判断：** 两者适合内容后台、文章元信息和发布管理；当“直接在正文里划词编辑”是第一需求时，优先级低于原位编辑器。这不是说它们不能编辑文章，而是默认交互不完全对应用户优先级。

### Hypothesis

- **事实：** 支持网页注释，但不直接改写文章正文；没有附注内容的单纯高亮默认为私有。普通公共层、私有组与受限组的可见性/写权限不同，公开读但限制写的 Restricted Groups 属合作方案能力。[注释基础](https://web.hypothes.is/help/annotation-basics/)、[组权限](https://web.hypothes.is/help/annotating-with-groups/)
- **判断：** 更适合独立阅读批注层，不能单独满足修改正文、公开高亮与作者独占编辑的整套要求。

### Milkdown / Crepe

- **事实：** Crepe 提供选中文字的格式工具栏；支持有序/无序/任务列表、CodeMirror 代码块、KaTeX 行内和块级数学；支持指定 DOM 根节点、`getMarkdown()` 和只读状态。[Crepe API](https://milkdown.dev/docs/api/crepe)
- **事实：** Milkdown 基于 ProseMirror 与 remark，MIT 许可。扩展 schema 可定义 `parseMarkdown` 与 `toMarkdown`，并非只能将 HTML 转成 Markdown。[官方仓库](https://github.com/Milkdown/milkdown)、[Transformer API](https://milkdown.dev/docs/api/transformer)
- **判断：** 可以将其挂到当前文章正文位置，沿用外部网站布局，加载原始 Markdown，再把 Markdown 提交给现有 Worker。该 API 路径支持这种集成，但不是官方替本站提供的 GitHub 登录或发布功能。[Crepe API](https://milkdown.dev/docs/api/crepe)
- **待验证：** 跨五个列表项应用自定义高亮、复杂脚注、脚注内公式、`mark.author-highlight`、图片双语切换数据能否无损往返。官方功能清单中没有可直接等同本站“作者公开批注”的成品功能；需以公开内容节点/mark/脚注适配，而非假设已有 Comments。自定义高亮按钮示例也不等于已有完整高亮持久化实现。[Crepe API](https://milkdown.dev/docs/api/crepe)、[Transformer API](https://milkdown.dev/docs/api/transformer)

### CKEditor 5

- **事实：** Inline Editor 可以直接在目标位置编辑，聚焦出现浮动工具栏。Highlight 使用 `<mark>`，官方示例包含列表。[Inline 示例](https://ckeditor.com/docs/ckeditor5/latest/examples/builds/inline-editor.html)、[Highlight](https://ckeditor.com/docs/ckeditor5/latest/features/highlight.html)
- **事实：** Markdown 插件提供 GFM 输入/输出，但官方提醒部分富文本能力只有 HTML 输出能保留，并通常推荐 HTML 作为数据格式。GHS 可以白名单保留额外元素、属性、class 和 style，但只提供有限编辑支持，不能自动得到自定义组件的交互界面。[Markdown](https://ckeditor.com/docs/ckeditor5/latest/features/markdown.html)、[General HTML Support](https://ckeditor.com/docs/ckeditor5/latest/features/html/general-html-support.html)
- **事实：** OSS 分发采用 GPL 2+ / 商业双许可；Comments 属于高级功能。官方 MathType 集成为 Wiris 提供的额外付费功能。[许可](https://ckeditor.com/docs/ckeditor5/latest/getting-started/licensing/license-and-legal.html)、[Comments](https://ckeditor.com/docs/ckeditor5/latest/features/collaboration/comments/comments.html)、[数学许可说明](https://support.ckeditor.com/hc/en-us/articles/10514337276572-Does-CKEditor-5-support-math-equations-and-chemical-formulas)
- **事实：** Comments 的文档标记只保存 thread ID，评论正文不是普通文章数据；官方推荐 adapter 保存，也提供同时加载/保存内容与评论的路线。[Comments 数据设计](https://ckeditor.com/docs/ckeditor5/latest/features/collaboration/comments/comments.html)
- **判断：** 可用成熟编辑交互加自己的保存接口，但本站公开批注仍需另做发布渲染/序列化；不应认为购买 Comments 就自动导出本站脚注。跨列表高亮可以列为原型验收项，文档示例不构成本站实测通过。

### TinyMCE

- **事实：** Inline 模式直接编辑页面元素，使用宿主页面 CSS，无 iframe；但是官方明确 **Comments 只支持 classic 模式**，因为 inline 和 distraction-free 不支持侧栏。[Inline 模式与限制](https://www.tiny.cloud/docs/tinymce/latest/use-tinymce-inline/)
- **事实：** Math 是付费插件；输出以 `tiny-math-inline` / `tiny-math-block` 保存公式。Markdown 插件提供把当前 HTML 转为 Markdown 的 API，但不是原始 Markdown 字节保真承诺。[Math](https://www.tiny.cloud/docs/tinymce/latest/math/)、[Markdown](https://www.tiny.cloud/docs/tinymce/latest/markdown/)
- **事实：** TinyMCE 8 自托管须使用 `gpl` 或有效商业 license key；GPL 路线为 GPLv2+。不能将“可自托管”解读为无许可条件。[许可配置](https://www.tiny.cloud/docs/tinymce/latest/license-key/)
- **判断：** 单看原位文字编辑很贴合；合并“原位 + 原生 Comments + Markdown 数学博客”后不省适配。可自定义公开批注，但这正是用户希望少做的工作。`getContent` / `setContent` 可接自有保存服务，框架本身不强制替换 GitHub 身份系统。[Editor API](https://www.tiny.cloud/docs/tinymce/latest/apis/tinymce.editor/)

### MDXEditor

- **事实：** React 组件，以 Markdown 字符串输入输出，基于 Lexical 和 MDAST 双向转换；核心 MIT。API 有现成 HighlightToggle，代码块有专门编辑支持。[架构概述](https://mdxeditor.dev/editor/docs/overview)、[LICENSE](https://github.com/mdx-editor/editor/blob/main/LICENSE)、[HighlightToggle](https://mdxeditor.dev/editor/api/functions/HighlightToggle)、[代码块](https://mdxeditor.dev/editor/docs/code-blocks)
- **事实：** 普通 HTML 可以变成 generic HTML nodes 并编辑内部内容；默认没有新增/删除/修改 HTML 属性的完整 UI，官方更建议 directive 或 JSX 扩展。JSX 可以注册自己的编辑组件。[HTML 支持](https://mdxeditor.dev/editor/docs/html-support)、[JSX](https://mdxeditor.dev/editor/docs/jsx)
- **事实：** 可通过插件、Lexical 节点以及 Markdown import/export visitors 扩展；因此“支持 MDX”不能推出“任意现有语法自动保留”。[扩展接口](https://mdxeditor.dev/editor/docs/extending-the-editor)
- **待验证：** 当前没有从本次查阅资料确认开箱即用的数学/复杂作者脚注方案。高亮跨多个列表项、已有 mark class 的保留也必须用本站数据验证，不能把 Lexical 的一般能力当作具体集成保证。
- **判断：** 保存 Markdown 到现有 Worker 的形式很自然，但 React 接入、保持当前正文样式、公式和批注适配的成本未知。[架构概述](https://mdxeditor.dev/editor/docs/overview)

### BlockNote

- **事实：** 有现成块编辑 UI 与 Comments。核心 MPL 2.0；XL 功能采用 GPL-3.0 / 商业双许可。Comments 列在 Community 功能中，不应误称为一律付费。[定价及许可](https://www.blocknotejs.org/pricing)
- **事实：** 官方对 Markdown 导入和导出都标注为 **lossy**，建议 `JSON.stringify(editor.document)` 用作无损存储。导入仅覆盖常见 Markdown 子集，支持所有扩展语法不是项目目标。[导入](https://www.blocknotejs.org/docs/features/import/markdown)、[导出](https://www.blocknotejs.org/docs/features/export/markdown)
- **事实：** 当前已有 `@blocknote/math-block`，支持行内/块级 LaTeX；数学可导出 `$...$` / `$$`，该数学组件依赖 React。代码块可配置语法高亮。[Math](https://www.blocknotejs.org/docs/features/blocks/math)、[代码块](https://www.blocknotejs.org/docs/features/blocks/code-blocks)
- **事实：** Comments 接入要求用户解析、ThreadStore 和实时协作，并非一个仅把注释放入普通 Markdown 的按钮。[Comments](https://www.blocknotejs.org/docs/features/collaboration/comments)
- **判断：** 可继续用自己的后端存储，但若坚持 Markdown 为唯一事实来源、公开批注静态发布，需要转换/扩展；为单作者博客引入协作数据体系未必划算。自定义 block/inline/style 能承载站点组件，但也意味着额外 schema 与序列化开发。[自定义 Schema](https://www.blocknotejs.org/docs/features/custom-schemas)

## 共同边界与后续验收建议

以下是本次调研形成的工程判断，不是厂商承诺：

1. **原位不是自动接管任何网页 DOM。** 需要用一个编辑器根覆盖整个连续文章内容；若仍逐段建立互不关联的编辑器，跨列表/跨段落选择问题可能重现。
2. **公开作者批注作为内容建模。** 保留类似 mark + 脚注的稳定表示，编辑时提供弹窗/侧栏，发布时静态渲染。协作 Comments 线程和作者注释是不同产品语义。
3. **暂不改变身份系统。** 候选都提供内容 API，原则上可沿用现有 Worker 验证 GitHub 身份；编辑器只负责本地编辑。前端隐藏工具栏不能代替后端权限检查。
4. **先验证保存的可靠性。** 将真实文章导入后不修改便导出，检查语义/结构而不只比字节；再执行跨列表高亮、批注新增/修改、撤销、公式编辑、图注编辑。检查原图链接、译图映射、脚注引用、代码语言、LaTeX、frontmatter 是否保留。
5. **评估页面一致性。** 用现有站点正文 CSS 验证桌面/窄屏以及只读和编辑态；不能把“能嵌入当前页面”说成“默认就和现有排版完全一致”。

这份调研仅足以缩小候选名单。没有完成运行时 PoC，因此不声称任何候选已经解决本站跨列表选择或实现完全无损往返。

## 建议的验证顺序和交付体验

先用《LLM 推理并行策略》原文副本对比 Vditor 与 Crepe。主要门槛是保存可靠性，而不是空白演示的外观：

1. 一次选中截图里的五个列表项，新增高亮和同一条公开批注，能够修改、删除和撤销。
2. 连续编辑普通段落、列表和中英文混排，输入法行为正常；不再逐段切成独立编辑区域。
3. 保存后重开文章，公式源码、代码语言与缩进、原图链接、译图关联、已有作者注释、文章元信息保持正确。
4. 批注锚点经附近正文修改后仍指向正确内容；用户能看到保存失败或冲突，不覆盖新版本。
5. 发布渲染与当前文章布局一致；匿名读者看见正文、高亮、作者批注，不能保存编辑。

预期交付体验：本人登录 → 在真实文章页面启用编辑 → 正文原位修改，导语作为独立可编辑区域 → 划词出现高亮/批注操作 → 本地即时看到草稿效果 → 明确保存/发布后更新公开站点。实时预览不等于每次按键立刻公开发布。

架构判断：采用浏览器编辑器时，原则上可继续使用 Astro 静态公开站点 + 现有 Cloudflare Worker，不必仅为了原位输入而把整个站点迁移成服务端网站。前提是补齐内容转换和安全保存接口；Tina 的服务端预览要求是其集成路线的要求，不能泛化到所有原位编辑方案。

这轮没有购买服务、授权新平台或改动线上内容。下一阶段应依据上述真实文章验证结果选择，而不是仅凭产品介绍直接迁移全部文章。
