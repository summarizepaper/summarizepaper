/* The collection page (templates/summarizer/collection.html): tabs, each paper saved as it is edited, filters
   and sorting, and the work done in the background (table, open questions, checks, suggestions, imports).
   Without this script the page still works: every section shows and each form posts. */
(function () {
  var root = document.getElementById('collection');
  if (!root) return;
  var t = document.getElementById('collection-text').dataset;
  var base = root.dataset.base;
  var pk = root.dataset.pk;
  var here = location.pathname;
  root.classList.add('js');

  // --- tabs ---------------------------------------------------------------------
  var tablist = root.querySelector('[role=tablist]');
  var tabs = Array.prototype.slice.call(root.querySelectorAll('[role=tab]'));
  var panels = tabs.map(function (tab) { return document.getElementById(tab.getAttribute('aria-controls')); });

  function show(id, focus) {
    tabs.forEach(function (tab, i) {
      var on = panels[i].id === id;
      tab.setAttribute('aria-selected', on ? 'true' : 'false');
      tab.tabIndex = on ? 0 : -1;
      panels[i].hidden = !on;
      if (on && focus) tab.focus();
    });
  }

  function fromHash() {
    var hash = location.hash.slice(1);
    if (/^row-\d+$/.test(hash)) return 'table';
    return panels.some(function (p) { return p.id === hash; }) ? hash : 'papers';
  }

  if (tablist) {
    tablist.hidden = false;
    show(fromHash());
    // a link to a tab opens it with the tabs in sight, not under the sticky header
    if (location.hash.slice(1) === fromHash() && tablist.scrollIntoView) tablist.scrollIntoView();
    tabs.forEach(function (tab, i) {
      tab.addEventListener('click', function () {
        show(panels[i].id);
        history.replaceState(null, '', '#' + panels[i].id);
      });
      tab.addEventListener('keydown', function (e) {
        var step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
        if (!step) return;
        e.preventDefault();
        var next = panels[(i + step + tabs.length) % tabs.length].id;
        show(next, true);
        history.replaceState(null, '', '#' + next);
      });
    });
    window.addEventListener('hashchange', function () { show(fromHash()); });
  }

  // --- papers: saved as they change -------------------------------------------------
  var list = document.getElementById('paper-list');
  var count = document.getElementById('selection-count');

  function selectionText(selected, total) {
    if (!count) return;
    var text = count.dataset.text.replace('{n}', selected).replace('{total}', total);
    count.textContent = selected > Number(root.dataset.max) ? text + ' ' + count.dataset.over : text;
  }

  function counted() {
    var cards = list ? Array.prototype.slice.call(list.children) : [];
    selectionText(cards.filter(function (c) { return c.querySelector('input[type=checkbox]').checked; }).length, cards.length);
  }

  function save(form, fields) {
    var status = form.querySelector('.tool-status');
    status.textContent = t.saving;
    fields.action = 'paper';
    fields.arxiv_id = form.dataset.id;
    return SP.post(here, fields).then(function (json) {
      status.textContent = json.status === 'saved' ? t.saved : t.failed;
      return json;
    }).catch(function () { status.textContent = t.failed; });
  }

  function setUp(form) {
    var card = form.closest('.paper-card');
    var reading = form.querySelector('[name=reading]');
    var tags = form.querySelector('[name=tags]');
    var note = form.querySelector('[name=note]');
    var box = form.querySelector('input[type=checkbox]');
    var timer = null;
    function search() {
      card.dataset.search = [card.dataset.title, card.querySelector('.meta').textContent, note.value, tags.value].join(' ').toLowerCase();
    }
    reading.addEventListener('change', function () {
      card.dataset.reading = reading.value;
      save(form, {reading: reading.value});
    });
    tags.addEventListener('change', function () {
      save(form, {tags: tags.value}).then(function (json) {
        if (json && json.tags !== undefined) tags.value = json.tags;
        card.dataset.tags = tags.value.toLowerCase();
        search();
      });
    });
    note.addEventListener('input', function () {
      clearTimeout(timer);
      timer = setTimeout(function () { timer = null; save(form, {note: note.value}); search(); }, 1200);
    });
    note.addEventListener('blur', function () {
      if (timer) { clearTimeout(timer); timer = null; save(form, {note: note.value}); search(); }
    });
    box.addEventListener('change', function () {
      save(form, {selected: box.checked ? '1' : '0'});
      counted();
    });
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var removing = e.submitter && e.submitter.value === 'remove';
      if (!removing) return save(form, {note: note.value, tags: tags.value, reading: reading.value});
      if (!confirm(t.remove)) return;
      SP.post(here, {action: 'remove', arxiv_id: form.dataset.id}).then(function (json) {
        if (json.status !== 'removed') return;
        card.remove();
        counted();
        filter();
      });
    });
  }
  if (list) Array.prototype.slice.call(list.querySelectorAll('.paper-form')).forEach(setUp);

  // --- filters and sorting ---------------------------------------------------------
  var query = document.getElementById('paper-filter');
  var tagPick = document.getElementById('paper-tag');
  var readingPick = document.getElementById('paper-reading');
  var sortPick = document.getElementById('paper-sort');
  var empty = document.getElementById('paper-empty');

  function number(value, missing) {
    return value === '' || value === undefined ? missing : Number(value);
  }

  var ORDERS = {
    order: function (a, b) { return number(a.dataset.order, 0) - number(b.dataset.order, 0); },
    newest: function (a, b) { return number(b.dataset.year, -1) - number(a.dataset.year, -1); },
    oldest: function (a, b) { return number(a.dataset.year, 1e9) - number(b.dataset.year, 1e9); },
    cited: function (a, b) { return number(b.dataset.cited, -1) - number(a.dataset.cited, -1); },
    title: function (a, b) { return a.dataset.title.localeCompare(b.dataset.title); }
  };

  function filter() {
    if (!list) return;
    var q = query.value.trim().toLowerCase();
    var tag = tagPick.value;
    var reading = readingPick.value;
    var cards = Array.prototype.slice.call(list.children);
    var shown = 0;
    cards.forEach(function (card) {
      var tags = card.dataset.tags.split(',').map(function (s) { return s.trim(); });
      var ok = (!q || card.dataset.search.indexOf(q) >= 0) && (!tag || tags.indexOf(tag) >= 0) &&
        (!reading || (reading === 'none' ? !card.dataset.reading : card.dataset.reading === reading));
      card.hidden = !ok;
      if (ok) shown++;
    });
    var order = ORDERS[sortPick.value] || ORDERS.order;
    cards.sort(function (a, b) { return order(a, b) || ORDERS.order(a, b); }).forEach(function (c) { list.appendChild(c); });
    empty.hidden = shown > 0 || !cards.length;
    Array.prototype.slice.call(root.querySelectorAll('[data-export]')).forEach(function (link) {
      link.href = link.href.split('?')[0] + (tag ? '?tag=' + encodeURIComponent(tag) : '');
    });
  }
  if (list) {
    query.addEventListener('input', filter);
    [tagPick, readingPick, sortPick].forEach(function (el) { el.addEventListener('change', filter); });
  }

  // --- work done in the background ----------------------------------------------------
  // name: the prefix of its panel (name-working) and status (name-status); buttons: [data-run=name], with their
  // data-mode ('' starts or follows, 'update' reads only what changed, 'again' writes it all again).
  var URLS = {compare: '/compare/', questions: '/questions/', meta: '/refresh/', suggest: '/suggest/'};

  function runner(name) {
    var panel = document.getElementById(name + '-working');
    var status = document.getElementById(name + '-status');
    var buttons = Array.prototype.slice.call(root.querySelectorAll('[data-run="' + name + '"]'));
    if (!panel || !status) return null;
    function failed(message) {
      status.textContent = message || t.failed;
      panel.hidden = true;
      buttons.forEach(function (b) { b.disabled = false; b.hidden = false; });
    }
    function start(mode) {
      status.textContent = '';
      panel.hidden = false;
      buttons.forEach(function (b) { b.disabled = true; });
      return SP.post(base + '/collections/' + pk + URLS[name], {mode: mode || ''}).then(function (json) {
        if (json.status === 'premium') return failed(t.premium);
        if (json.status === 'too_few') return failed(t.tooFew);
        if (!json.id) return failed();
        SP.track(panel, json.id, function () { location.reload(); }, failed);
      }).catch(function () { failed(); });
    }
    buttons.forEach(function (b) {
      b.addEventListener('click', function () { panel.dataset.started = ''; start(b.dataset.mode); });
    });
    if (status.dataset.pending === '1') start('');
    return start;
  }
  Object.keys(URLS).forEach(runner);

  // --- imports (a long list, a Zotero collection) ---------------------------------------
  var importPanel = document.getElementById('import-working');
  var importStatus = document.getElementById('import-status');

  function imported(id) {
    importPanel.dataset.started = '';
    SP.track(importPanel, id, function (counts) {
      var params = Object.keys(counts || {}).filter(function (k) { return counts[k]; })
        .map(function (k) { return k + '=' + encodeURIComponent(counts[k]); }).join('&');
      location.href = here + (params ? '?' + params : '') + '#papers';
    }, function () { importPanel.hidden = true; importStatus.textContent = t.failed; });
  }
  if (importStatus && importStatus.dataset.job) imported(importStatus.dataset.job);

  var zLoad = document.getElementById('zotero-import-load');
  var zPick = document.getElementById('zotero-import-pick');
  var zGo = document.getElementById('zotero-import-go');
  var zStatus = document.getElementById('zotero-import-status');
  if (zLoad) zLoad.addEventListener('click', function () {
    zStatus.textContent = t.loading;
    fetch(base + '/collections/zotero/', {credentials: 'same-origin'}).then(function (r) { return r.json(); }).then(function (json) {
      if (json.status === 'unavailable') return zStatus.textContent = t.unavailable;
      if (json.status !== 'done') return zStatus.textContent = t.failed;
      if (!json.collections.length) return zStatus.textContent = t.empty;
      zPick.textContent = '';
      json.collections.forEach(function (c) {
        var option = document.createElement('option');
        option.value = c.key;
        option.textContent = c.name + (c.items ? ' (' + c.items + ')' : '');
        zPick.appendChild(option);
      });
      zStatus.textContent = '';
      zLoad.hidden = true;
      zPick.hidden = false;
      zGo.hidden = false;
    }).catch(function () { zStatus.textContent = t.unavailable; });
  });
  if (zGo) zGo.addEventListener('click', function () {
    zGo.disabled = true;
    SP.post(base + '/collections/' + pk + '/zotero/', {key: zPick.value}).then(function (json) {
      if (!json.id) { zGo.disabled = false; zStatus.textContent = json.status === 'premium' ? t.premium : t.failed; return; }
      imported(json.id);
    });
  });

  // --- Zotero export, shared link ----------------------------------------------------------
  var zotero = document.getElementById('zotero-go');
  if (zotero) zotero.addEventListener('click', function () {
    var zs = document.getElementById('zotero-status');
    zs.textContent = t.sending;
    SP.post(base + '/zotero/send/', {collection: pk}).then(function (json) {
      if (json.status === 'premium') return zs.textContent = t.premium;
      if (json.status === 'setup') { zs.textContent = t.setup + ' '; var a = document.createElement('a'); a.href = json.url; a.textContent = '→'; zs.appendChild(a); return; }
      if (!json.id) return zs.textContent = t.failed;
      SP.wait(json.id, function () { zs.textContent = t.sent; }, function () { zs.textContent = t.failed; });
    });
  });

  var copy = document.getElementById('share-copy');
  if (copy) copy.addEventListener('click', function () {
    var field = document.getElementById('share-url');
    field.select();
    (navigator.clipboard ? navigator.clipboard.writeText(field.value) : Promise.reject()).catch(function () {
      document.execCommand('copy');
    }).then(function () { copy.textContent = copy.dataset.copied; });
  });

  var template = document.getElementById('columns-template');
  if (template) template.addEventListener('change', function () {
    Array.prototype.slice.call(root.querySelectorAll('[data-template]')).forEach(function (span) {
      span.hidden = span.dataset.template !== template.value;
    });
  });
})();
