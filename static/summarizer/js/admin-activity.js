// The staff activity page /admin/activite/ (activity.py): the whole day comes as JSON and is filtered here, by kind,
// members and text, in time order or per person; the other days are fetched and swapped in without reloading.
// The filters live in the address (type, membres, q, vue), so a filtered view can be shared.
const ActivityFeed = (() => {
  const PAGE_KINDS = new Set(['visit']);  // only shown in time order when asked: they would drown the actions

  function member(e) { return (e.p || '').startsWith('u'); }

  function matches(e, state, view) {
    if (state.kind ? e.k !== state.kind : view === 'time' && PAGE_KINDS.has(e.k)) return false;
    if (state.members && !member(e)) return false;
    if (state.q) {
      const q = state.q.toLowerCase();
      if (![e.x, e.d, e.w].some(v => (v || '').toLowerCase().includes(q))) return false;
    }
    return true;
  }

  function filter(events, state, view) { return events.filter(e => matches(e, state, view)); }

  // [{key, who, whoUrl, events (oldest first), counts {kind: n}, first, last, origin}], the latest active first;
  // the actions no visit can be tied to (anonymous questions, searches...) last
  function group(events) {
    const people = new Map();
    for (const e of events) {
      const key = e.p || '';
      if (!people.has(key)) people.set(key, {key, who: '', whoUrl: '', events: [], counts: {}, origin: ''});
      const person = people.get(key);
      person.events.push(e);
      person.counts[e.k] = (person.counts[e.k] || 0) + 1;
      if (e.w && !person.who) { person.who = e.w; person.whoUrl = e.wu; }
    }
    const out = [...people.values()];
    for (const person of out) {
      person.events.reverse();  // the feed is latest first
      person.first = person.events[0].t;
      person.last = person.events[person.events.length - 1].t;
      const arrival = person.events.find(e => e.k === 'visit');
      person.origin = arrival && arrival.d || '';
    }
    return out.sort((a, b) => (a.key === '') - (b.key === '') || (b.last > a.last) - (b.last < a.last));
  }

  function readState(search) {
    const params = new URLSearchParams(search);
    return {kind: params.get('type') || '', members: params.get('membres') === '1', q: params.get('q') || '',
            view: params.get('vue') === 'personnes' ? 'person' : 'time'};
  }

  function writeState(search, state) {
    const params = new URLSearchParams(search);
    const set = (name, value) => value ? params.set(name, value) : params.delete(name);
    set('type', state.kind); set('membres', state.members ? '1' : ''); set('q', state.q);
    set('vue', state.view === 'person' ? 'personnes' : '');
    const text = params.toString();
    return text ? '?' + text : '';
  }

  // --- the page --------------------------------------------------------------

  const LINK = 'text-primary-600 dark:text-primary-500';
  const ON = ['border-primary-600', 'text-primary-600', 'dark:text-primary-500'];
  const OFF = ['border-base-200', 'dark:border-base-700'];

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(attrs || {})) {
      if (name === 'class') node.className = value;
      else if (name === 'style') node.style.cssText = value;
      else node.setAttribute(name, value);
    }
    for (const child of children) if (child !== null && child !== undefined && child !== '') node.append(child);
    return node;
  }

  function safeHref(href) { return typeof href === 'string' && href.startsWith('/') && !href.startsWith('//') ? href : ''; }

  function link(text, href, external) {
    const url = safeHref(href);
    if (!url) return document.createTextNode(text);
    const a = el('a', {class: LINK, href: url}, text);
    if (external) { a.target = '_blank'; a.rel = 'noopener'; }
    return a;
  }

  function cut(text, n) { text = text || ''; return text.length > n ? text.slice(0, n - 1) + '…' : text; }

  function kindCell(e, kinds) {
    const kind = kinds[e.k] || {label: e.k, icon: 'circle'};
    return el('span', {class: 'flex items-center gap-1', style: e.a ? 'color:#dc2626' : ''},
              el('span', {class: 'material-symbols-outlined', style: 'font-size:16px'}, kind.icon), kind.label);
  }

  function textCell(e) {
    return el('td', {class: 'py-2 pr-3', style: 'word-break:break-word'}, link(cut(e.x, 160), e.h, true),
              e.d ? el('span', {class: 'text-subtle', style: e.a ? 'color:#dc2626' : ''}, ' · ' + e.d) : null);
  }

  function whoCell(e) {
    if (e.w) return link(cut(e.w, 32), e.wu);
    return e.n ? el('span', {class: 'text-subtle'}, 'anonyme') : null;
  }

  function timeTable(events, kinds, withWho) {
    const table = el('table', {class: 'w-full text-sm'});
    for (const e of events) {
      table.append(el('tr', {class: 'border-t border-base-200 dark:border-base-800 align-top'},
        el('td', {class: 'py-2 pr-3 text-subtle whitespace-nowrap'}, e.t),
        el('td', {class: 'py-2 pr-3 whitespace-nowrap'}, kindCell(e, kinds)),
        textCell(e),
        withWho ? el('td', {class: 'py-2 text-right whitespace-nowrap'}, whoCell(e)) : null));
    }
    return table;
  }

  function personTitle(person) {
    if (person.key === '') return el('span', {class: 'font-semibold'}, 'Actions anonymes sans visite rattachée');
    if (person.who) return link(person.who, person.whoUrl);
    return el('span', {class: 'font-semibold'}, (person.key.startsWith('u') ? 'Membre' : 'Visiteur anonyme') + ' · ' + person.key.slice(1, 7));
  }

  function personBlock(person, kinds) {
    const counts = Object.entries(person.counts).sort((a, b) => b[1] - a[1])
      .map(([k, n]) => ((kinds[k] || {}).label || k) + (n > 1 ? ' ×' + n : '')).join(' · ');
    const summary = el('summary', {style: 'cursor:pointer;list-style-position:outside', class: 'py-2'},
      personTitle(person), el('span', {class: 'text-subtle'}, ' · ' + person.first + (person.last !== person.first ? ' → ' + person.last : '') + ' · ' + counts),
      person.origin ? el('span', {class: 'text-subtle'}, ' · ' + person.origin) : null);
    return el('details', {class: 'border-t border-base-200 dark:border-base-800'}, summary, timeTable(person.events, kinds, false));
  }

  function setActive(button, on) {
    for (const c of ON) button.classList.toggle(c, on);
    for (const c of OFF) button.classList.toggle(c, !on);
  }

  function render(root, events, kinds, state) {
    const shown = filter(events, state, state.view);
    const target = root.querySelector('[data-feed]');
    const summary = root.querySelector('[data-summary]');
    target.replaceChildren();
    root.querySelectorAll('[data-view]').forEach(b => setActive(b, b.dataset.view === state.view));
    root.querySelectorAll('[data-kind]').forEach(b => setActive(b, b.dataset.kind === state.kind));
    if (!shown.length) {
      summary.textContent = '';
      target.append(el('p', {class: 'text-subtle'}, events.length ? 'Rien ne correspond à ces filtres.' : 'Rien ce jour-là.'));
      return;
    }
    if (state.view === 'person') {
      const people = group(shown);
      const known = people.filter(p => p.key !== '');
      const members = known.filter(p => p.key.startsWith('u')).length;
      summary.textContent = known.length + ' personne' + (known.length > 1 ? 's' : '') + ' (' + members + ' membre' + (members > 1 ? 's' : '') +
        '), ' + shown.length + ' actions et pages. Les visites anonymes se suivent sur une seule journée.';
      for (const person of people) target.append(personBlock(person, kinds));
    } else {
      summary.textContent = shown.length + ' ligne' + (shown.length > 1 ? 's' : '') + (state.kind ? '' : ' (les pages vues sont dans leur filtre et dans la vue par personne)');
      target.append(timeTable(shown, kinds, true));
    }
  }

  function init(root) {
    const events = JSON.parse(root.querySelector('#activity-feed').textContent);
    const kinds = JSON.parse(root.querySelector('#activity-kinds').textContent);
    const state = readState(location.search);
    const search = root.querySelector('[data-search]');
    const members = root.querySelector('[data-members]');
    search.value = state.q;
    members.checked = state.members;
    const update = () => {
      history.replaceState(history.state, '', location.pathname + writeState(location.search, state));
      render(root, events, kinds, state);
    };
    root.querySelectorAll('[data-kind]').forEach(b => b.addEventListener('click', () => { state.kind = b.dataset.kind; update(); }));
    root.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => { state.view = b.dataset.view; update(); }));
    members.addEventListener('change', () => { state.members = members.checked; update(); });
    let typing;
    search.addEventListener('input', () => { clearTimeout(typing); typing = setTimeout(() => { state.q = search.value.trim(); update(); }, 150); });
    render(root, events, kinds, state);

    root.querySelectorAll('[data-period-button]').forEach(b => b.addEventListener('click', () => {
      root.querySelectorAll('[data-period-button]').forEach(o => setActive(o, o === b));
      root.querySelectorAll('[data-period-table]').forEach(t => { t.style.display = t.dataset.periodTable === b.dataset.periodButton ? '' : 'none'; });
    }));

    // another day: fetched and swapped in, keeping the filters
    root.querySelectorAll('[data-nav]').forEach(a => a.addEventListener('click', event => {
      if (event.metaKey || event.ctrlKey || event.shiftKey) return;
      event.preventDefault();
      go(a.getAttribute('href'), state);
    }));
    const form = root.querySelector('[data-day-form]');
    form.addEventListener('submit', event => event.preventDefault());
    form.querySelector('input[type=date]').addEventListener('change', event => {
      const params = new URLSearchParams(new FormData(form));
      if (event.target.value) go('?' + params.toString(), state);
    });

    const lazy = root.querySelector('[data-lazy]');
    if (lazy) {
      fetch(lazy.dataset.lazy, {credentials: 'same-origin'})
        .then(r => r.ok ? r.text() : Promise.reject(r.status))
        .then(html => { lazy.innerHTML = html; })
        .catch(() => { lazy.replaceChildren(el('p', {class: 'text-subtle'}, 'La fiabilité de l’assistant n’a pas pu être chargée.')); });
    }
  }

  function go(href, state, push = true) {
    const url = new URL(href, location.href);
    if (state) {
      for (const [name, value] of new URLSearchParams(writeState('', state))) url.searchParams.set(name, value);
    }
    const root = document.getElementById('activity-root');
    root.style.opacity = '.5';
    fetch(url, {credentials: 'same-origin'})
      .then(r => r.ok ? r.text() : Promise.reject(r.status))
      .then(html => {
        const next = new DOMParser().parseFromString(html, 'text/html');
        const fresh = next.getElementById('activity-root');
        if (!fresh) throw new Error('no activity');
        if (push) history.pushState({activity: true}, '', url.pathname + url.search);
        // the title of the admin's header, outside the swapped part
        const before = root.dataset.title, after = fresh.dataset.title;
        document.querySelectorAll('h1, h2, span, a, li').forEach(n => {
          if (!root.contains(n) && !n.children.length && n.textContent.trim() === before) n.textContent = after;
        });
        document.title = next.title;
        root.replaceWith(fresh);
        init(fresh);
      })
      .catch(() => { location.href = url.href; });
  }

  if (typeof document !== 'undefined' && typeof module === 'undefined') {
    const start = () => {
      const root = document.getElementById('activity-root');
      if (root) init(root);
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
    window.addEventListener('popstate', () => go(location.href, null, false));
  }

  return {filter, group, readState, writeState, safeHref};
})();

if (typeof module !== 'undefined') module.exports = ActivityFeed;
