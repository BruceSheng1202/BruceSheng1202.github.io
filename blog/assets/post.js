/* Shared renderer for markdown-based blog posts.
 *
 * A post page only needs:
 *   <script src="content.js"></script>            // window.POST_CONTENT = { en, zh }
 *   <script>window.POST_CONFIG = { term: 'slug', bibtex: '...' };</script>
 *   <script src="../assets/post.js"></script>
 *
 * Optional config keys: term, content (global name), bibtex, repo, repoId,
 * category, categoryId.
 */
(() => {
  const cfg = Object.assign({
    repo: 'BruceSheng1202/BruceSheng1202.github.io',
    repoId: 'R_kgDOMH0GSg',
    category: 'Blog Comments',
    categoryId: 'DIC_kwDOMH0GSs4DBPJB',
    term: '',
    content: 'POST_CONTENT',
    bibtex: ''
  }, window.POST_CONFIG || {});

  const content = () => window[cfg.content] || {};

  let request = 0;
  let tocHeadings = [];
  let tocFrame = 0;
  let citationFrame = 0;

  function slug(text, i) {
    return text.toLowerCase()
      .replace(/<[^>]+>/g, '')
      .replace(/[^a-z0-9\u4e00-\u9fff]+/g, '-')
      .replace(/^-|-$/g, '') || `section-${i}`;
  }

  function articleLang(requested) {
    const data = content();
    if (data[requested]) return requested;
    return data.zh ? 'zh' : 'en';
  }

  function findReferences(article) {
    return [...article.querySelectorAll('h1')].find(h => {
      const text = h.textContent.trim().toLowerCase();
      return text === 'references' || text === '参考文献';
    });
  }

  function citationKey(href) {
    try {
      const url = new URL(href, location.href);
      const host = url.hostname.replace(/^www\./, '').toLowerCase();
      if (host === 'arxiv.org') {
        const match = url.pathname.match(/\/(?:abs|pdf)\/([^/]+?)(?:\.pdf)?$/);
        if (match) return `arxiv:${match[1].replace(/v\d+$/i, '')}`;
      }
      if (host === 'openreview.net') {
        return `openreview:${url.searchParams.get('id') || url.pathname.replace(/\/$/, '')}`;
      }
      if (host === 'doi.org') return `doi:${url.pathname.replace(/^\/|\/$/g, '').toLowerCase()}`;
      return `${host}${url.pathname.replace(/\/$/, '')}${url.search}`;
    } catch (_) {
      return href;
    }
  }

  function updateToc() {
    if (!tocHeadings.length) return;
    let current = tocHeadings[0];
    for (const heading of tocHeadings) {
      if (heading.getBoundingClientRect().top <= 118) current = heading;
      else break;
    }
    document.querySelectorAll('#toc-links a').forEach(link => {
      const active = link.getAttribute('href') === `#${current.id}`;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }

  function scheduleTocUpdate() {
    cancelAnimationFrame(tocFrame);
    tocFrame = requestAnimationFrame(updateToc);
  }

  function buildToc(headings) {
    tocHeadings = headings;
    const toc = document.getElementById('toc-links');
    toc.innerHTML = '';
    headings.forEach(h => {
      const a = document.createElement('a');
      a.href = `#${h.id}`;
      a.textContent = h.textContent;
      if (h.tagName === 'H2') a.className = 'toc-subsection';
      toc.appendChild(a);
    });
    scheduleTocUpdate();
  }

  function layoutCitationNotes() {
    const rail = document.getElementById('citation-rail');
    const main = document.querySelector('.article-main');
    const notes = [...document.querySelectorAll('.citation-note')];
    if (!rail || !main || !notes.length) return;
    rail.style.minHeight = `${main.offsetHeight}px`;
    if (getComputedStyle(rail).display === 'none') return;

    const railTop = rail.getBoundingClientRect().top + window.scrollY;
    let nextTop = 0;
    notes.forEach(note => {
      const source = document.getElementById(note.dataset.sourceId);
      if (!source) return;
      const desiredTop = source.getBoundingClientRect().top + window.scrollY - railTop - 2;
      const top = Math.max(desiredTop, nextTop);
      note.style.top = `${Math.round(top)}px`;
      nextTop = top + note.offsetHeight + 14;
    });
  }

  function scheduleCitationLayout() {
    cancelAnimationFrame(citationFrame);
    citationFrame = requestAnimationFrame(() => requestAnimationFrame(layoutCitationNotes));
  }

  function buildCitationRail(article, lang) {
    const notes = document.getElementById('citation-notes');
    notes.innerHTML = '';
    const references = findReferences(article);
    if (!references) return;

    const referencesByUrl = new Map();
    for (let node = references.nextElementSibling; node; node = node.nextElementSibling) {
      if (node.matches('h1, h2, section')) break;
      if (node.tagName !== 'P') continue;
      node.querySelectorAll('a[href]').forEach(link => {
        referencesByUrl.set(citationKey(link.href), node);
      });
    }

    const citedLinks = [];
    for (let node = article.firstElementChild; node && node !== references; node = node.nextElementSibling) {
      if (node.classList.contains('comments-section')) continue;
      citedLinks.push(...node.querySelectorAll('a[href]'));
    }

    let citationIndex = 0;
    const cited = new Set();
    citedLinks.forEach(link => {
      const key = citationKey(link.href);
      const reference = referencesByUrl.get(key);
      if (!reference || cited.has(key)) return;
      cited.add(key);
      citationIndex += 1;
      const sourceId = `citation-source-${citationIndex}`;
      link.id = sourceId;
      link.dataset.citation = String(citationIndex);

      const note = document.createElement('div');
      note.className = 'citation-note';
      note.dataset.sourceId = sourceId;
      note.setAttribute('role', 'note');
      note.setAttribute('aria-label', `${lang === 'zh' ? '引用' : 'Reference'} ${citationIndex}`);
      note.innerHTML = `<span class="citation-number" aria-hidden="true">${citationIndex}</span><div class="citation-copy">${reference.innerHTML}</div>`;
      const copy = note.querySelector('.citation-copy');
      copy.innerHTML = copy.innerHTML.replace(/^\s*\d+\.\s*/, '');
      note.querySelectorAll('a[href]').forEach(source => {
        source.textContent = lang === 'zh' ? '查看来源 ↗' : 'Open source ↗';
        source.target = '_blank';
        source.rel = 'noreferrer';
      });

      const setActive = active => {
        link.classList.toggle('citation-active', active);
        note.classList.toggle('active', active);
      };
      link.addEventListener('mouseenter', () => setActive(true));
      link.addEventListener('mouseleave', () => setActive(false));
      link.addEventListener('focus', () => setActive(true));
      link.addEventListener('blur', () => setActive(false));
      note.addEventListener('mouseenter', () => setActive(true));
      note.addEventListener('mouseleave', () => setActive(false));
      notes.appendChild(note);
    });

    article.querySelectorAll('img').forEach(img => {
      if (!img.complete) img.addEventListener('load', scheduleCitationLayout, { once: true });
    });
    scheduleCitationLayout();
  }

  function insertBibtexCitation(article, lang) {
    const references = findReferences(article);
    if (!references || !cfg.bibtex) return;

    const section = document.createElement('section');
    section.className = 'bibtex-section';
    section.setAttribute('aria-labelledby', 'bibtex-heading');
    section.innerHTML = `
      <div class="bibtex-heading">
        <div>
          <h1 id="bibtex-heading">${lang === 'zh' ? '引用本文' : 'Cite this article'}</h1>
          <p>${lang === 'zh' ? '复制以下 BibTeX 条目用于论文或研究笔记。' : 'Copy this BibTeX entry for papers or research notes.'}</p>
        </div>
        <button class="bibtex-copy" type="button">${lang === 'zh' ? '复制 BibTeX' : 'Copy BibTeX'}</button>
      </div>
      <pre class="bibtex-code"><code></code></pre>
      <p class="bibtex-status" aria-live="polite"></p>`;
    references.before(section);

    section.querySelector('code').textContent = cfg.bibtex;
    const button = section.querySelector('.bibtex-copy');
    const status = section.querySelector('.bibtex-status');
    const defaultLabel = button.textContent;
    button.addEventListener('click', async () => {
      let copied = false;
      try {
        if (!navigator.clipboard?.writeText) throw new Error('Clipboard API unavailable');
        await navigator.clipboard.writeText(cfg.bibtex);
        copied = true;
      } catch (_) {}
      if (!copied) {
        const textarea = document.createElement('textarea');
        textarea.value = cfg.bibtex;
        textarea.setAttribute('readonly', '');
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
          copied = document.execCommand('copy');
        } finally {
          textarea.remove();
        }
      }
      if (!copied) {
        status.textContent = lang === 'zh'
          ? '自动复制失败，请手动选择上方条目。'
          : 'Automatic copy failed. Please select the entry above.';
        return;
      }
      button.textContent = lang === 'zh' ? '已复制' : 'Copied';
      status.textContent = lang === 'zh' ? 'BibTeX 已复制到剪贴板。' : 'BibTeX copied to the clipboard.';
      window.setTimeout(() => {
        button.textContent = defaultLabel;
        status.textContent = '';
      }, 2400);
    });
  }

  function insertComments(article, lang) {
    const references = findReferences(article);
    if (!references) return;

    const section = document.createElement('section');
    section.className = 'comments-section';
    section.setAttribute('aria-labelledby', 'comments-heading');
    section.innerHTML = `
      <h1 id="comments-heading">${lang === 'zh' ? '讨论' : 'Discussion'}</h1>
      <div class="comments-thread" id="comments-thread"></div>`;
    references.before(section);

    const thread = section.querySelector('.comments-thread');
    if (location.protocol === 'file:') return;
    if (!cfg.repoId || !cfg.categoryId || !cfg.term) {
      thread.innerHTML = `<p class="comments-setup">${lang === 'zh'
        ? '评论区正在等待连接 GitHub Discussions。'
        : 'The comment thread is waiting for its GitHub Discussions connection.'}</p>`;
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://giscus.app/client.js';
    script.dataset.repo = cfg.repo;
    script.dataset.repoId = cfg.repoId;
    script.dataset.category = cfg.category;
    script.dataset.categoryId = cfg.categoryId;
    script.dataset.mapping = 'specific';
    script.dataset.term = cfg.term;
    script.dataset.strict = '1';
    script.dataset.reactionsEnabled = '1';
    script.dataset.emitMetadata = '0';
    script.dataset.inputPosition = 'top';
    script.dataset.theme = 'light';
    script.dataset.lang = lang === 'zh' ? 'zh-CN' : 'en';
    script.dataset.loading = 'lazy';
    script.crossOrigin = 'anonymous';
    script.async = true;
    thread.appendChild(script);
  }

  function renderTables(article, lang) {
    article.querySelectorAll('table').forEach(table => {
      const wrap = document.createElement('div');
      wrap.className = 'table-scroll';
      table.parentNode.insertBefore(wrap, table);
      wrap.appendChild(table);
      if (wrap.scrollWidth > wrap.clientWidth + 2) {
        const hint = document.createElement('div');
        hint.className = 'table-hint';
        hint.textContent = lang === 'zh' ? '表格可横向滚动 →' : 'Scroll the table sideways →';
        wrap.after(hint);
      }
    });
  }

  function renderFigures(article) {
    article.querySelectorAll('img').forEach(img => {
      img.loading = 'lazy';
      const paragraph = img.parentElement;
      const figure = document.createElement('figure');
      if (paragraph?.tagName === 'P' && paragraph.children.length === 1) paragraph.replaceWith(figure);
      else img.parentNode.insertBefore(figure, img);
      figure.appendChild(img);

      const note = figure.nextElementSibling;
      const caption = document.createElement('figcaption');
      if (note?.tagName === 'P' && note.children.length === 1 && note.firstElementChild?.tagName === 'EM') {
        caption.innerHTML = note.firstElementChild.innerHTML;
        note.remove();
      } else if (img.alt) {
        caption.textContent = img.alt;
      }
      if (caption.innerHTML.trim()) figure.appendChild(caption);
    });
  }

  async function render(lang) {
    lang = articleLang(lang);
    const current = ++request;
    const article = document.getElementById('article');
    document.getElementById('toc-links').innerHTML = '';
    document.getElementById('citation-notes').innerHTML = '';
    article.innerHTML = `<p>${lang === 'zh' ? '正在载入文章…' : 'Loading article…'}</p>`;
    try {
      const data = content();
      if (!data[lang] || !window.marked) throw new Error('content unavailable');
      if (current !== request) return;
      article.innerHTML = marked.parse(data[lang]);
      renderTables(article, lang);
      renderFigures(article);
      insertBibtexCitation(article, lang);
      insertComments(article, lang);
      const refHeading = findReferences(article);
      if (refHeading) refHeading.id = 'references';
      const headings = [...article.querySelectorAll('h1, h2')];
      headings.forEach((h, i) => { h.id = h.id || slug(h.textContent, i); });
      if (current !== request) return;
      buildToc(headings);
      buildCitationRail(article, lang);
    } catch (_) {
      article.innerHTML = `<p>${lang === 'zh' ? '文章载入失败，请刷新页面重试。' : 'The article could not be loaded. Please refresh the page.'}</p>`;
    }
  }

  function start() {
    window.addEventListener('scroll', scheduleTocUpdate, { passive: true });
    window.addEventListener('resize', () => {
      scheduleTocUpdate();
      scheduleCitationLayout();
    });
    if (document.fonts?.ready) document.fonts.ready.then(scheduleCitationLayout);
    document.addEventListener('languagechange', e => render(e.detail.lang));
    render(window.siteLanguage ? window.siteLanguage.get() : 'en');
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
