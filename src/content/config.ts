import { defineCollection, z } from 'astro:content';
const schema = z.object({
  title: z.string(),
  description: z.string(),
  intro: z.string().optional(),
  date: z.coerce.date(),
  draft: z.boolean().optional(),
  repoURL: z.string().optional(),
  tags: z.array(z.string()).default([]),
  sourceURL: z.string().url().optional(),
  sourceAuthor: z.string().optional(),
  translationScope: z.string().optional(),
});
export const collections = {
  projects: defineCollection({ type: 'content', schema }),
  blog: defineCollection({ type: 'content', schema }),
};
