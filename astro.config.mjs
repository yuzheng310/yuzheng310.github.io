import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import tailwind from "@astrojs/tailwind";

import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeImageDimensions from "./src/plugins/rehype-image-dimensions.mjs";

export default defineConfig({
  site: "https://yuzheng310.github.io",
  devToolbar: { enabled: false },
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex, rehypeImageDimensions],
    shikiConfig: { themes: { light: "github-light-high-contrast", dark: "github-dark" } },
  },
  integrations: [mdx(), sitemap(), tailwind()],
});
