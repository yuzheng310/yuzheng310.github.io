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

## HTML 翻译文章

- 笔记元数据存放于 `src/content/blog/*.md`：`tags` 用于主题归档，`date` 表示本站发布日期。
- `htmlFile` 指向 `src/documents/translations/` 内的完整 HTML。文章沿用原有阅读排版，并自动补上本站导航、作者、来源、翻译范围和标签。
- 译文图片、公式库和代码高亮资源保存在 `public/translations/` 与 `public/vendor/`。第三方库许可证随资源保留，外链图片来源记录在 `scripts/translation-assets.json`。
- 2026-10-06 收录 8 篇 HTML；已收录的 vLLM 译文不重复导入。Mooncake、Continuum 标注为论文精读，py-kvcache 标注为选章翻译。
- 如需重新从原工作区导入，可运行 `uv run --with beautifulsoup4 python scripts/import-translations.py /path/to/workspace`。该命令会重新生成这 8 篇发布副本，不改动原始 HTML；重新导入前应保留发布副本上的手工修改。
- CUDA 入门译文的静态样式通过 `npx tailwindcss -c scripts/translations-tailwind.cjs -o public/translations/cuda-from-zero-to-hero/tailwind.css --minify` 生成，不依赖运行时 Tailwind CDN。

发布前运行 `npm run build`，并检查译文图片、数学公式、目录链接与移动端阅读布局。`llm-inference-explained` 的来源文件仅保留 Avi Chawla 的作者主页，文章直链仍待补；其中演示视频保留外部链接。
