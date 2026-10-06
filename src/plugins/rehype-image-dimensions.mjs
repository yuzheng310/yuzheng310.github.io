import { resolve, sep } from 'node:path';
import sharp from 'sharp';

const publicDir = resolve('public');
const metadataCache = new Map();
async function dimensions(src) {
  if (typeof src !== 'string' || !src.startsWith('/') || src.startsWith('//')) return;
  const file = resolve(publicDir, decodeURIComponent(src.split(/[?#]/)[0]).slice(1));
  if (!file.startsWith(publicDir + sep)) return;
  if (!metadataCache.has(file)) {
    metadataCache.set(file, sharp(file).metadata().then(({ width, height }) => ({ width, height })));
  }
  return metadataCache.get(file);
}

// Reserve local figure space before images load, including HTML figures in translations.
export default function rehypeImageDimensions() {
  return async tree => {
    async function visit(node) {
      if (node.type === 'element' && node.tagName === 'img') {
        const size = await dimensions(node.properties.src);
        if (size?.width && size?.height && !node.properties.width && !node.properties.height) {
          Object.assign(node.properties, size, { decoding: 'async' });
        }
      }
      if (node.type === 'raw' && node.value.includes('<img')) {
        const images = [...node.value.matchAll(/<img\b[^>]*>/gi)];
        for (const [tag] of images) {
          const src = tag.match(/\bsrc=["']([^"']+)["']/i)?.[1];
          if (/\b(?:width|height)=/i.test(tag)) continue;
          const size = await dimensions(src);
          if (size?.width && size?.height) {
            node.value = node.value.replace(tag, tag.replace(/\s*\/?\s*>$/, ` width="${size.width}" height="${size.height}" decoding="async">`));
          }
        }
      }
      if (node.children) await Promise.all(node.children.map(visit));
    }
    await visit(tree);
  };
}
