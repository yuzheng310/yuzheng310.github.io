import { defineCollection, z } from 'astro:content';
const schema = z.object({title:z.string(),description:z.string(),date:z.coerce.date(),draft:z.boolean().optional(),repoURL:z.string().optional()});
export const collections = {projects:defineCollection({type:'content',schema}),blog:defineCollection({type:'content',schema})};
