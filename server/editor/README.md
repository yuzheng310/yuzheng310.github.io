# 文章页内编辑

配置记录（2026-10-07）：GitHub App 已创建并仅安装到本站仓库；Worker 已部署到 `https://yuzheng-article-editor.2565554517.workers.dev`，登录与会话密钥已保存在 Cloudflare Secrets。应用和服务的非敏感标识见 `wrangler.toml`。未设置 `PUBLIC_AUTHOR_API_URL` 时，正式网站不显示编辑入口。本实现是轻量页内编辑层，使用 GitHub App 登录和 GitHub 历史，不是 TinaCMS 集成。

## 上线验收记录

2026-10-07：正式文章页已通过 GitHub 作者登录；在网页中添加临时导语并确认发布，生成提交 `9bf7522`，自动部署后在正式页面确认可见，随后恢复文章原内容。匿名读取/发布返回 401，其他 Origin 返回 403；本地 10 项自动测试与正式构建、链接检查通过。其他真实账号的登录拒绝尚未人工验收，服务端身份拒绝已由自动测试覆盖。

本站部署由 GitHub Actions 自动完成；Worker 后续更新在本目录使用 `npx wrangler@4.148.0 deploy`。官方工具凭据以加密文件保存，密钥在 macOS 钥匙串中。若重新登录，所需权限为 `account:read user:read workers:write workers_scripts:write`，无需申请默认的其他产品权限。

## 使用方式

文章标题下点击「作者编辑」，用网站作者的 GitHub 账号登录。在同一段普通文字内选择内容，修改、高亮或添加公开批注；底部工具栏可编辑导语、撤销和预览后发布。导语显示在正文前；批注保存为公开脚注，可从正文跳转。原有公式、代码、图片和目录继续使用现有渲染器。

本地 `npm run dev` 不配置服务即可体验；本地模式不能发布。草稿只在当前页面内存中，离开页面前有提醒，请勿把浏览器当作长期草稿存储。第一次实现只支持单个连续文本片段，不支持跨粗体、链接、公式、代码、段落的选择；同一个片段已有草稿时需先撤销再改。公开脚注的后续改写、删除高亮和复杂结构编辑仍需改仓库源文件。

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
- 写入目标固定为本站仓库 main 分支的 `src/content/blog/<slug>.md`。客户端只能提交带原文的范围操作；服务检查范围、原文和 GitHub 文件 SHA，有冲突就拒绝覆盖。
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

测试覆盖身份拒绝、OAuth 状态与 PKCE、会话篡改/过期、版本冲突、固定仓库路径、文本转义、批注脚注渲染，以及公式与代码保持不变。单元测试中的 GitHub 和 Cloudflare 请求使用模拟；正式环境应执行上述接入验收。

参考：[GitHub App 用户授权](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-user-access-token-for-a-github-app)、[GitHub Contents API](https://docs.github.com/en/rest/repos/contents#create-or-update-file-contents)、[Wrangler 配置](https://developers.cloudflare.com/workers/wrangler/configuration/)。
