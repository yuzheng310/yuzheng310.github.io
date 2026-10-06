# yuzheng310 个人主页

网站：**[yuzheng310.github.io](https://yuzheng310.github.io/)**

基于 [Astro Nano](https://github.com/markhorn-dev/astro-nano)（MIT）改写，使用简洁的文字排版展示项目与笔记。

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

Mooncake、Continuum 为论文精读改写，py-kvcache 为选章译文整理。`llm-inference-explained` 的来源文件仅保留 Avi Chawla 的作者主页，文章直链待补。

发布前运行 `npm run build`，检查站内链接、公式、图片、目录和移动端布局。
