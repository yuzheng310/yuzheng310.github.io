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

项目介绍根据现有材料整理。技术笔记是本次建站整理的新稿。以公开源码中的实验条件与限制为准。网站使用 GitHub 昵称，不展示真实姓名或私人联系方式。

## 发布

本项目使用 GitHub Pages，仓库为 `yuzheng310/yuzheng310.github.io`。推送到 `main` 会触发 `.github/workflows/deploy.yml`：构建 Astro 静态页面，然后部署到 GitHub Pages。网站不依赖 Sites 托管。

仓库 Settings → Pages 的发布来源为 GitHub Actions。项目使用账号根域名，不需要配置额外的 `base` 前缀。
