import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import tailwind from "@astrojs/tailwind";

export default defineConfig({
  site: "https://yuzheng310.gentle-puppy-3957.chatgpt.site",
  devToolbar: { enabled: false },
  integrations: [mdx(), sitemap(), tailwind()],
});
