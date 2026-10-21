/* All content is in HTML: navigation, downloads, and reading also work offline. */
'use strict';
(() => {
  const items = [...document.querySelectorAll('.evidence')];
  const headings = [...document.querySelectorAll('.section-heading')];
  const search = document.getElementById('search');
  const filterButtons = [...document.querySelectorAll('[data-filter]')];
  const count = document.getElementById('result-count');
  const empty = document.getElementById('empty-state');
  let filter = 'all';
  // Index raw text before math rendering so exact numbers and TeX labels remain searchable.
  items.forEach(item => { item.dataset.search = (item.id.replaceAll('-', ' ') + ' ' + item.textContent).toLowerCase(); });
  function applyFilters() {
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    items.forEach(item => {
      const show = (filter === 'all' || item.dataset.kind === filter) && (!query || item.dataset.search.includes(query));
      item.hidden = !show;
      if (show) visible++;
    });
    headings.forEach(heading => { heading.hidden = filter !== 'all' && filter !== 'protocols' || Boolean(query); });
    empty.hidden = visible > 0;
    count.textContent = filter === 'all' && !query ? 'Showing all supplementary material' : `${visible} matching ${visible === 1 ? 'item' : 'items'}${query ? ' · “' + search.value.trim() + '”' : ''}`;
    filterButtons.forEach(button => {
      const selected = button.dataset.filter === filter;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
  }
  filterButtons.forEach(button => button.addEventListener('click', () => { filter = button.dataset.filter; applyFilters(); }));
  search.addEventListener('input', applyFilters);
  document.addEventListener('keydown', event => {
    const isInput = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName);
    if (event.key === '/' && !isInput && !document.querySelector('dialog[open]')) { event.preventDefault(); search.focus(); }
    if (event.key === 'Escape' && document.activeElement === search) { search.value = ''; applyFilters(); search.blur(); }
  });
  function revealTarget(hash, scroll = false) {
    const target = document.getElementById(decodeURIComponent(hash.slice(1)));
    if (!target) return;
    if (target.closest('.evidence, .section-heading')) {
      filter = 'all'; search.value = ''; applyFilters();
      const item = target.closest('.evidence, .section-heading');
      item.classList.remove('highlight-pulse');
      requestAnimationFrame(() => item.classList.add('highlight-pulse'));
    }
    if (scroll) target.scrollIntoView({block: 'start'});
  }
  document.querySelectorAll('a[href^="#"]').forEach(link => link.addEventListener('click', () => revealTarget(link.hash, true)));
  window.addEventListener('hashchange', () => revealTarget(location.hash, true));
  if (location.hash) setTimeout(() => revealTarget(location.hash, true), 200);
  const dialog = document.getElementById('figure-dialog');
  const zoomImage = document.getElementById('zoom-image');
  const caption = document.getElementById('zoom-caption');
  let opener;
  document.querySelectorAll('.figure-zoom').forEach(button => button.addEventListener('click', () => {
    opener = button; zoomImage.src = button.dataset.image; zoomImage.alt = button.dataset.caption;
    caption.textContent = button.dataset.caption;
    if (dialog.showModal) dialog.showModal(); else window.open(button.dataset.image, '_blank', 'noopener');
  }));
  function closeDialog() { dialog.close(); if (opener) opener.focus(); }
  document.getElementById('close-dialog').addEventListener('click', closeDialog);
  dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) closeDialog(); } });
  dialog.addEventListener('close', () => { if (opener) opener.focus(); });
  document.getElementById('print-page').addEventListener('click', () => window.print());
  if (typeof renderMathInElement === 'function') {
    renderMathInElement(document.querySelector('main'), { delimiters: [{left: '\\(', right: '\\)', display: false}], throwOnError: false, strict: 'ignore', trust: false });
  }
  applyFilters();
})();
