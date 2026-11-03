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
import { createHash } from 'node:crypto';
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

const output = `window.POST_CONTENT = {\n${body}\n};\n`;
await writeFile(out, output, 'utf8');

// Stamp content.js with a short hash in index.html so a rebuild is never served
// from the 10-minute GitHub Pages cache. Only the script tag is touched.
const htmlFile = path.join(dir, 'index.html');
try {
  const html = await readFile(htmlFile, 'utf8');
  const hash = createHash('sha1').update(output, 'utf8').digest('hex').slice(0, 8);
  const stamped = html.replace(
    /(<script src="content\.js)(\?v=[0-9a-f]+)?(")/,
    `$1?v=${hash}$3`
  );
  if (stamped !== html) {
    await writeFile(htmlFile, stamped, 'utf8');
    console.log(`stamped content.js?v=${hash} in ${path.relative(process.cwd(), htmlFile)}`);
  }
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
}

const sizes = Object.entries(sources).map(([l, t]) => `${l} ${t.length} chars`).join(', ');
console.log(`wrote ${path.relative(process.cwd(), out)} (${sizes})`);
