#!/usr/bin/env node
/**
 * Build content.js for a markdown-based blog post.
 *
 *   node blog/tools/build-content.mjs blog/<slug>
 *
 * Reads <slug>/en.md and <slug>/zh.md and writes <slug>/content.js as
 *   window.POST_CONTENT = { en: String.raw`...`, zh: String.raw`...` };
 *
 * Markdown must not contain backticks or the sequence ${ — both break the
 * String.raw template. Use <code>...</code> for inline code instead.
 */
import { readFile, writeFile, access } from 'node:fs/promises';
import path from 'node:path';

const dir = path.resolve(process.argv[2] || '.');
const files = { en: path.join(dir, 'en.md'), zh: path.join(dir, 'zh.md') };
const out = path.join(dir, 'content.js');

const sources = {};
for (const [lang, file] of Object.entries(files)) {
  try {
    await access(file);
  } catch {
    console.error(`missing ${path.relative(process.cwd(), file)}`);
    process.exit(1);
  }
  const text = await readFile(file, 'utf8');
  for (const bad of ['`', '${']) {
    if (text.includes(bad)) {
      console.error(`${path.basename(file)} contains "${bad}", which breaks String.raw. Replace it before building.`);
      process.exit(1);
    }
  }
  sources[lang] = text.trimEnd();
}

const body = Object.entries(sources)
  .map(([lang, text]) => `  ${lang}: String.raw\`${text}\``)
  .join(',\n');

await writeFile(out, `window.POST_CONTENT = {\n${body}\n};\n`, 'utf8');

const sizes = Object.entries(sources).map(([l, t]) => `${l} ${t.length} chars`).join(', ');
console.log(`wrote ${path.relative(process.cwd(), out)} (${sizes})`);
