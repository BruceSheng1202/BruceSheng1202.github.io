(() => {
  const allowed = ['en', 'zh'];
  const KEY = 'blog-lang';

  const read = () => {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  };
  const save = next => {
    try { localStorage.setItem(KEY, next); } catch (e) {}
  };

  const stored = read();
  const query = new URLSearchParams(location.search).get('lang');
  let lang = allowed.includes(query) ? query : (allowed.includes(stored) ? stored : 'en');

  function syncUrl(next) {
    const url = new URL(location.href);
    if (url.searchParams.get('lang') === next) return;
    url.searchParams.set('lang', next);
    history.replaceState(null, '', url);
  }

  function apply(next) {
    lang = allowed.includes(next) ? next : 'en';
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-en][data-zh]').forEach(el => { el.textContent = el.dataset[lang]; });
    document.querySelectorAll('[data-lang]').forEach(button => {
      const active = button.dataset.lang === lang;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', active);
    });
    document.dispatchEvent(new CustomEvent('languagechange', { detail: { lang } }));
  }

  document.querySelectorAll('[data-lang]').forEach(button => button.addEventListener('click', () => {
    const next = button.dataset.lang;
    apply(next);
    save(next);
    syncUrl(next);
  }));

  window.siteLanguage = { get: () => lang, apply };
  apply(lang);
  if (stored !== lang) save(lang);
  syncUrl(lang);
})();
