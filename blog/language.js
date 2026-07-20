(() => {
  const allowed = ['en', 'zh'];
  const query = new URLSearchParams(location.search).get('lang');
  let lang = allowed.includes(query) ? query : 'en';

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

  document.querySelectorAll('[data-lang]').forEach(button => button.addEventListener('click', () => apply(button.dataset.lang)));
  window.siteLanguage = { get: () => lang, apply };
  apply(lang);
})();
