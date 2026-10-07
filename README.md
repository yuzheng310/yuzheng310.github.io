# yuzheng 个人主页

网站：**[yuzheng310.github.io](https://yuzheng310.github.io/)**

基于 [Astro Nano](https://github.com/markhorn-dev/astro-nano)（MIT）改写。整体结构与克制交互参考 [Astro Sphere](https://github.com/markhorn-dev/astro-sphere)，中文长文的字体、行宽与层级参考 [Retypeset](https://github.com/radishzzz/astro-theme-retypeset)。保留独立 Astro 布局，没有混入第二套主题框架。

- 首页：`src/pages/index.astro`
- 项目记录：`src/content/projects/`
- 技术笔记：`src/content/blog/`
- 关于：`src/pages/about/index.astro`
- 公共布局：`src/layouts/Portfolio.astro`
- 样式：`src/styles/portfolio.css`

## 本地运行

使用兼容现有锁文件的 pnpm 8：

```sh
npx pnpm@8.15.9 install --frozen-lockfile
npm run dev
npm run build
```

项目介绍根据现有材料整理。技术笔记包含项目记录、翻译与论文精读。以公开源码中的实验条件与限制为准。主页使用 GitHub 昵称；文章中的原作者署名与来源单独保留。

## 发布

本项目使用 GitHub Pages，仓库为 `yuzheng310/yuzheng310.github.io`。推送到 `main` 会触发 `.github/workflows/deploy.yml`：构建 Astro 静态页面，然后部署到 GitHub Pages。网站不依赖 Sites 托管。

仓库 Settings → Pages 的发布来源为 GitHub Actions。项目使用账号根域名，不需要配置额外的 `base` 前缀。

## 翻译与阅读笔记

所有文章都使用 `src/content/blog/*.md` 和同一个文章模板。新文章不要嵌入完整 HTML 文档，也不携带独立的导航、页面 CSS 或脚本。

- `tags` 用于主题归档，`date` 表示本站发布日期。
- `sourceURL`、`sourceAuthor`、`translationScope` 统一显示原文来源、作者和整理范围。
- 使用 Markdown 标题、列表、表格、代码块；图表保存在 `public/translations/`。必要时可用语义化 `figure` / `figcaption`，不添加内联样式。
- 数学公式通过统一的 remark-math / rehype-katex 在构建时渲染，使用 `$…$` 或独立的 `$$` 公式块。公式不依赖文章自己的浏览器脚本。
- 8 篇 HTML 发布副本留在 `references/translations/` 作为改写参考，不参与网站渲染。站内正文以 Markdown 为准，后续修改直接编辑正文；旧的全页 HTML 导入器已移除，避免覆盖编辑结果。
- 图片来源记录于 `scripts/translation-assets.json`；第三方资源许可证保留。

Mooncake、Continuum 为论文精读改写，py-kvcache 为选章译文整理。Avi Chawla 的大模型推理图解系列（推理机制与并行策略）已补齐原文直链与完整架构图解。

发布前运行 `npm run build`，检查站内链接、公式、图片、目录和移动端布局。

## 设计与阅读功能

- 浅色纸面与墨绿强调色为默认；深色模式由读者主动切换并保存在本机。
- Inter 用于界面，Lora 与思源宋体用于标题、中文长文；字体本地托管、按 Unicode 子集加载，许可证在 `public/licenses/`。
- 首页依次展示个人介绍、三个项目、原创工程笔记、最新翻译与精读。共享项目摘要维护在 `src/data/projects.ts`，详细内容在 content collection。
- 样式分为 `portfolio.css`（全站与页面）、`reading.css`（阅读与公式）、`archive.css`（时间线与标签）。
- 正文默认桌面 18px / 手机 17px，代码 14px / 13px，两组大小分别保存。目录桌面固定在侧边，手机折叠。
- 代码块支持复制，公式区分行内与行间；阅读页提供进度与返回顶部。
- 背景与按钮只使用少量 CSS 动画；可手动暂停，系统减少动态效果时关闭。文章不加载首页背景动效。
- 全文搜索使用 Pagefind，构建后自动索引 13 篇笔记与 3 个项目（未来新增非草稿内容会自动加入）；不会重复索引标签列表。仅搜索页加载搜索资源，不发送查询到第三方。

### 本地检查

`npm run build` 包括 Astro 类型检查、静态构建和 Pagefind 索引。搜索必须在构建后用 `npm run preview` 检查；开发模式没有生成索引时会给出明确提示。

`python3 scripts/check-site.py` 在构建后检查页面、站内链接、静态资源、锚点、KaTeX 错误和搜索资源。还需在浏览器检查首页、长文、搜索和手机布局；字号与主题是浏览器本地偏好。
