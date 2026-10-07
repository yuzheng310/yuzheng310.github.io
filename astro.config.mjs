import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import tailwind from "@astrojs/tailwind";

import remarkMath from "remark-math";
import remarkAuthorTargets from "./src/plugins/remark-author-targets.mjs";
import rehypeAuthorNotes from "./src/plugins/rehype-author-notes.mjs";
import rehypeKatex from "rehype-katex";
import rehypeImageDimensions from "./src/plugins/rehype-image-dimensions.mjs";

export default defineConfig({
  site: "https://yuzheng310.github.io",
  devToolbar: { enabled: false },
  markdown: {
    remarkPlugins: [remarkMath, remarkAuthorTargets],
    rehypePlugins: [rehypeKatex, rehypeImageDimensions, rehypeAuthorNotes],
    shikiConfig: { themes: { light: "github-light-high-contrast", dark: "github-dark" } },
  },
  integrations: [mdx(), sitemap(), tailwind()],
});
