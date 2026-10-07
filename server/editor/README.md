# 文章页内编辑

配置记录（2026-10-07）：GitHub App 已创建并仅安装到本站仓库；Worker 已部署到 `https://yuzheng-article-editor.2565554517.workers.dev`，登录与会话密钥已保存在 Cloudflare Secrets。应用和服务的非敏感标识见 `wrangler.toml`。未设置 `PUBLIC_AUTHOR_API_URL` 时，正式网站不显示编辑入口。编辑内核采用 Milkdown / Crepe，结合本站排版、公开批注及现有 GitHub 发布服务。

## 上线验收记录

2026-10-07：正式文章页已通过 GitHub 作者登录；在网页中添加临时导语并确认发布，生成提交 `9bf7522`，自动部署后在正式页面确认可见，随后恢复文章原内容。匿名读取/发布返回 401，其他 Origin 返回 403；本地 10 项自动测试与正式构建、链接检查通过。其他真实账号的登录拒绝尚未人工验收，服务端身份拒绝已由自动测试覆盖。

本站部署由 GitHub Actions 自动完成；Worker 后续更新在本目录使用 `npx wrangler@4.148.0 deploy`。官方工具凭据以加密文件保存，密钥在 macOS 钥匙串中。若重新登录，所需权限为 `account:read user:read workers:write workers_scripts:write`，无需申请默认的其他产品权限。

## 使用方式

文章标题下点击「作者编辑」，使用作者的 GitHub 账号登录，随后直接在正文输入。可以跨段落、粗体、链接和多个列表项选择文字，通过浮动工具栏高亮或添加公开批注；底部工具栏提供导语、批注管理、正文撤销/重做、阅读预览、草稿下载和发布。批注支持 Markdown 和公式，可以修改、删除，并在相关原文附近显示可展开卡片。

公式、代码、图片保留为文档节点；译图在编辑和预览时也可切换中文/原图。正文未修改时保存原始 Markdown；修改后由 Milkdown 重新序列化，列表符号、空行等格式可能规范化。导入时会对原始文档与序列化结果做语义比对，无法无损支持的文章会拒绝进入编辑，而不是静默丢内容。少量旧文章中的原始 HTML 标题/锚点以受保护节点保留，不能像普通 Markdown 标题那样直接改字。

本地 `npm run dev` 不配置服务即可体验；本地模式不能发布。草稿保存在当前页面内存，离开前有提醒，也可下载 Markdown；尚无跨设备草稿同步或图片上传。阅读预览的代码保留文本格式，正式发布仍使用网站的语法高亮。编辑器仅在点击入口后加载，普通读者无需下载。

## 一次性接入

需要网站所有者完成 GitHub App 的创建/安装授权，以及 Cloudflare 账号登录。密钥只在本机安全输入或 Cloudflare Secrets 中配置，不要发到聊天、放进仓库或 `PUBLIC_*` 变量。

1. 在 Cloudflare Workers 预留服务域名（例如 `https://yuzheng-article-editor.<账户子域>.workers.dev`），保留 `wrangler.toml` 中的实际站点来源。
2. 在 GitHub 个人设置的 Developer settings → GitHub Apps 创建仅自己账户可安装的 App。Homepage URL 为 `https://yuzheng310.github.io`，Callback URL 为实际服务域名加 `/callback`；启用 expiring user authorization tokens，取消 webhook。Repository permissions 仅授予 **Contents: Read and write**，Metadata 保持默认。安装时仅选择 `yuzheng310.github.io` 仓库；不要授予账户权限或其他仓库权限。
3. 将 App 的 Client ID 填入 `wrangler.toml` 的 `GITHUB_CLIENT_ID`，将 `https://api.github.com/users/yuzheng310` 返回的数字 `id` 填入 `OWNER_ID`（不是用户名）。App private key 和 App ID 不用于这套 OAuth 流程。
4. 使用官方 Wrangler 登录 Cloudflare。在本目录执行 `npx wrangler secret put GITHUB_CLIENT_SECRET`，安全输入 App 的 Client secret。再执行 `npx wrangler secret put SESSION_SECRET`，安全输入密码管理器生成的至少 43 字符高熵随机值。不要提交真实 secret。最后执行 `npx wrangler deploy`。
5. 在网站仓库 Settings → Secrets and variables → Actions → Variables 设置 `PUBLIC_AUTHOR_API_URL` 为实际 Worker HTTPS 根地址，无结尾斜杠。提交代码后运行现有部署流程。
6. 用本人账号检查登录、读取文章、改一处文字并发布。确认 GitHub 新提交、Actions 构建和 Pages 新内容都成功，再开放日常使用。另用无权限账号验证拒绝访问；本地测试已模拟，但不能替代真实授权验证。

应用权限与官方部署工具权限已由网站所有者在本次上线过程中明确授权。Cloudflare 套餐、限制以账户实际配置为准。

## 权限与发布机制

- Worker 校验 GitHub 返回的数字用户 ID，每次文章读写都再次核验作者身份；前端隐藏按钮不是权限边界。
- OAuth 使用 state、PKCE S256 和 HttpOnly/Secure 的流程 Cookie。会话经过 AES-GCM 加密，最长一小时；浏览器仅在内存保存不透明会话，不持久化 GitHub token。过期后可通过工具栏重新登录，当前草稿保留。
- 写入目标固定为本站仓库 main 分支的 `src/content/blog/<slug>.md`。新版客户端提交 `format: "markdown-v1"`、`sha`、Markdown 正文与可选导语。服务端保留原始 frontmatter，只开放导语字段；验证文档大小、新增 HTML、链接协议与批注引用，并检查 GitHub 文件 SHA，有冲突就拒绝覆盖。旧版范围操作 API 暂时保留，支持前后端错峰部署。
- 保存产生 GitHub 提交，然后由现有 Actions 部署。返回“已提交”不等于“已上线”；前端提供发布状态链接。构建失败时文章线上仍为旧版，应查看 Actions 并修复或回退提交。
- 服务不负责 GitHub App 安装授权。卸载 App、撤销其用户授权或轮换 `SESSION_SECRET` 可以收回访问/使已有编辑会话失效。退出页面仅清除本页内存会话。
- 需要关闭功能时清空 Actions 的 `PUBLIC_AUTHOR_API_URL` 并重新部署网站；停止 Worker 可立即阻止新的在线写入。

## 检查

在仓库根目录执行：

```sh
npm run test:authoring
npm run build
python3 scripts/check-site.py
```

测试覆盖身份拒绝、OAuth 状态与 PKCE、会话篡改/过期、版本冲突、固定仓库路径、完整 Markdown 发布验证、14 篇现有文章的导入导出、跨列表高亮与批注的增改删/撤销，以及公式、代码和译图保持不变。单元测试中的 GitHub 和 Cloudflare 请求使用模拟；正式环境应执行上述接入验收。

参考：[GitHub App 用户授权](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-user-access-token-for-a-github-app)、[GitHub Contents API](https://docs.github.com/en/rest/repos/contents#create-or-update-file-contents)、[Wrangler 配置](https://developers.cloudflare.com/workers/wrangler/configuration/)。
