const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function harness({search = '', stored = null, share = true} = {}) {
  const events = {}, clicks = [], copied = [], tracked = [], requests = [], replaced = [], storage = {};
  if (stored) storage['presskit-role'] = stored;
  const tab = role => ({dataset: {roleTab: role}, attrs: {}, setAttribute(k, v) { this.attrs[k] = v; },
    addEventListener(type, fn) { this.click = fn; }});
  const tabs = ['author', 'commentator', 'writer'].map(tab);
  const panels = ['author', 'commentator', 'writer'].map(role => ({dataset: {rolePanel: role}, hidden: role !== 'commentator'}));
  const links = [{href: 'https://example.org/en/press-kit/2101.00001v1/pdf/?role=commentator'}];
  const field = {value: '', select() {}};
  const outdated = {hidden: false};
  const box = {hidden: true, querySelector: () => outdated};
  const buttons = {};
  ['share', 'unshare', 'update'].forEach(name => { buttons[name] = {addEventListener(type, fn) { this.click = fn; }}; });
  const root = {dataset: Object.assign({role: 'commentator', citation: 'Ann Lee et al., arXiv: 2101.00001v1', copied: 'Copied'},
                                       share ? {share: '/share/'} : {}),
    querySelectorAll(selector) {
      return {'[data-role-tab]': tabs, '[data-role-panel]': panels, '[data-role-link]': links}[selector] || [];
    },
    querySelector(selector) {
      if (selector.startsWith('[data-role-panel="')) return panels.find(p => selector.includes('"' + p.dataset.rolePanel + '"'));
      return {'[data-kit-share-box]': box, '[data-kit-share]': buttons.share, '[data-kit-unshare]': buttons.unshare,
              '[data-kit-share-update]': buttons.update}[selector] || null;
    },
    addEventListener(type, fn) { clicks.push(fn); }};
  const context = {
    URL, URLSearchParams, setTimeout: fn => fn(), Promise,
    location: {href: 'https://example.org/en/press-kit/2101.00001v1/' + search, search},
    history: {replaceState: (a, b, url) => replaced.push(url)},
    localStorage: {getItem: k => storage[k] || null, setItem: (k, v) => { storage[k] = v; }},
    navigator: {clipboard: {writeText: text => { copied.push(text); return Promise.resolve(); }}},
    window: {umami: {track: (name, data) => tracked.push(JSON.stringify([name, data]))}},
    SP: {post: (url, data) => new Promise(resolve => requests.push({url, data, resolve}))},
    document: {addEventListener: (type, fn) => { events[type] = fn; },
               getElementById: id => ({presskit: root, 'kit-share-url': field, 'kit-report-dialog': null})[id] || null},
  };
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/presskit.js', 'utf8'), context);
  events.DOMContentLoaded();
  const click = target => clicks.forEach(fn => fn({target: {closest: selector => target(selector)}}));
  return {root, tabs, panels, links, copied, tracked, requests, replaced, storage, click, box, field, outdated, buttons};
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test('a role tab shows its part, sets the exports to it, and is remembered and counted', () => {
  const h = harness();
  h.tabs[0].click();
  assert.equal(h.root.dataset.role, 'author');
  assert.deepEqual(h.panels.map(p => p.hidden), [false, true, true]);
  assert.equal(h.tabs[0].attrs['aria-selected'], 'true');
  assert.equal(h.links[0].href, '/en/press-kit/2101.00001v1/pdf/?role=author');
  assert.equal(h.storage['presskit-role'], 'author');
  assert.equal(h.replaced[0], '/en/press-kit/2101.00001v1/?role=author');
  assert.equal(h.tracked[0], '["presskit_role",{"role":"author"}]');
});

test('the remembered role applies unless the link asks for one, without counting it', () => {
  assert.equal(harness({stored: 'writer'}).root.dataset.role, 'writer');
  const asked = harness({stored: 'writer', search: '?role=author'});
  assert.equal(asked.root.dataset.role, 'commentator');
  assert.equal(asked.tracked.length, 0);
});

test('a quote is copied with the reference of the paper and its place', async () => {
  const h = harness();
  const button = {dataset: {copyQuote: 'We observe a change.', copyWhere: 'Results'}, textContent: 'Copy'};
  h.click(selector => selector === '[data-copy-quote]' ? button : null);
  await settle();
  assert.equal(h.copied[0], '“We observe a change.” (Ann Lee et al., arXiv: 2101.00001v1, Results)');
  assert.equal(h.tracked[0], '["presskit_copy",{"what":"quote"}]');
});

test('sharing the current kit again hides the outdated notice', async () => {
  const h = harness();
  h.buttons.update.click();
  assert.equal(h.requests[0].data.action, 'update');
  h.requests[0].resolve({status: 'ok', url: 'https://example.org/en/press-kit/shared/abc/'});
  await settle();
  assert.equal(h.field.value, 'https://example.org/en/press-kit/shared/abc/');
  assert.equal(h.box.hidden, false);
  assert.equal(h.outdated.hidden, true);
  assert.equal(h.tracked[0], '["presskit_share",{"action":"update"}]');
});
