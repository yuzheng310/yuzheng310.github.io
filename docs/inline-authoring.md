# 原位编辑实现

编辑核心复用 Milkdown / Crepe 7.22.2（MIT），底层为 ProseMirror。本项目定制文章排版、公开批注、中文译图与 GitHub 发布适配；不把上游编辑内核作为自研成果。

## 模块边界

- `src/editor/milkdown/engine.mjs`：文档编辑、撤销重做和 Markdown 读写。
- `src/editor/milkdown/annotations.mjs`：公开高亮及批注，跨段落选区映射与增改删。
- `src/lib/author-document.mjs`：客户端/Worker 共用的文档兼容性与保存校验。
- `src/editor/milkdown/surface.ts`：本站图片、HTML 保留节点、视觉适配。
- `src/editor/inline-client.ts`：作者会话、导语、草稿、预览和发布流程。
- `server/editor/worker.mjs`：固定仓库的身份核验及带版本检查的写入。

## 数据与发布

文章仍是 Markdown。高亮保存为受限制的 mark 标签；批注保存为脚注定义，通过 author-N 标识将多个选中片段关联到一个公开批注。不在页面 HTML 上做字符串替换，也不依赖 DOM 文本片段定位持久化修改。

作者登录 → 读取当前源文件 → 校验当前页面版本 → 编辑整篇文档 → 阅读预览 → Worker 校验和提交 → GitHub Actions 构建 → Pages 上线。正式网站只有静态阅读页；编辑服务继续部署在现有 Cloudflare Worker。

## 后续复用

先验证本站的长期使用。未来可抽取编辑内核、公开批注插件、Astro 接入和 GitHub 存储适配器，并为第二个站点验证配置接口后再发布包。目前没有新增公共仓库或声称已发布独立框架。

## 已知边界

- 不支持上传图片或多人协作；草稿暂存于内存，可手动下载。
- 原始 HTML 标题/锚点保留，复杂 HTML 不作为富文本自由编辑。
- 正文修改可能规范化 Markdown 排版，语义校验覆盖现有文章。
- 编辑过程中目录保留原始标题；重新发布后由 Astro 重建。
- 阅读预览没有代码语法着色；发布后的阅读页沿用现有高亮。

上游：[Milkdown](https://github.com/Milkdown/milkdown)。部署和权限说明见 `server/editor/README.md`。
