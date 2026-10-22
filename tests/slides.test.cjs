const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/summarizer/js/slides.js', 'utf8');

function setup() {
  function element(extra = {}) {
    const classes = new Set();
    return Object.assign({handlers: {}, dataset: {}, hidden: false, disabled: false, style: {setProperty() {}},
      clientWidth: 1000, scrollWidth: 1000, clientHeight: 400, scrollHeight: 400, offsetHeight: 50,
      addEventListener(type, fn) {this.handlers[type] = fn;}, setAttribute() {},
      classList: {toggle(name, on) {if (on === undefined) on = !classes.has(name); on ? classes.add(name) : classes.delete(name); return on;},
        contains: name => classes.has(name), add: name => classes.add(name), remove: name => classes.delete(name)},
      replaceChildren() {this.children = [];}, appendChild(child) {(this.children ||= []).push(child);},
    }, extra);
  }
  const inputs = Object.fromEntries(Object.entries({title: 'A title', bullets: 'Original result', secondary: '', notes: 'Only for the speaker',
    layout: 'right', figure_size: '50', text_size: '30'}).map(([key, value]) => [key, element({value})]));
  const heading = element(), bullets = element(), notes = element();
  const slide = element({dataset: {kind: 'content'}});
  const body = element({querySelectorAll: () => []});
  const frame = element();
  const selectors = {'.slide': slide, '.slide-body': body, '.slide-frame': frame,
    '[data-preview-title]': heading, '[data-preview-bullets]': bullets, '[data-preview-notes]': notes};
  const item = element({querySelectorAll: () => Object.values(inputs), querySelector(selector) {
    const match = selector.match(/^\[data-edit="(\w+)"\]$/);
    return match ? inputs[match[1]] : selectors[selector];
  }});
  const deck = element({dataset: {revision: 'r1', saveUrl: '/save/', saved: 'Saved', saving: 'Saving', unsaved: 'Unsaved',
    error: 'Error', conflict: 'Conflict', imageError: 'Missing image'}, querySelectorAll: selector => selector === '.slide-item' ? [item] : []});
  const ids = Object.fromEntries(['deck-status', 'deck-save', 'deck-pdf', 'deck-exit', 'deck-edit', 'deck-notes', 'deck-overflow', 'deck-present']
    .map(id => [id, element()])); ids.deck = deck;
  const pageEvents = {}, requests = [];
  let response = () => Promise.resolve({ok: true, status: 200, json: () => Promise.resolve({revision: 'r2'})});
  let prints = 0;
  vm.runInNewContext(source, {document: {getElementById: id => ids[id], querySelector: () => ({value: 'csrf-test'}),
    createElement: () => element(), addEventListener() {}},
    window: {addEventListener: (name, fn) => {pageEvents[name] = fn;}, print: () => {prints++;}},
    fetch: (url, options) => {requests.push({url, options}); return response();}, setTimeout, Promise});
  return {ids, inputs, item, body, slide, heading, bullets, deck, notes, requests, pageEvents,
    response(fn) {response = fn;}, prints: () => prints,
    edit(name, value) {inputs[name].value = value; inputs[name].handlers.input();}};
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test('text remains literal, figure controls update the preview, and edits are persisted with CSRF', async () => {
  const s = setup(); s.edit('title', '<script>literal</script>'); s.edit('layout', 'left');
  assert.equal(s.heading.textContent, '<script>literal</script>');
  assert.equal(s.slide.dataset.layout, 'left');
  let warned = false; s.pageEvents.beforeunload({preventDefault() {warned = true;}}); assert.equal(warned, true);
  s.ids['deck-save'].handlers.click(); await tick();
  const request = s.requests[0], data = JSON.parse(request.options.body);
  assert.equal(request.options.headers['X-CSRFToken'], 'csrf-test');
  assert.equal(data.pages[0].layout, 'left'); assert.equal(data.pages[0].notes, 'Only for the speaker');
  assert.equal(data.revision, 'r1'); assert.equal(s.deck.dataset.revision, 'r2');
  assert.equal(s.ids['deck-status'].textContent, 'Saved');
  s.pageEvents.beforeunload({preventDefault() {assert.fail('saved edits must not trigger navigation warning');}});
});

test('changes made during a save remain dirty and can be saved again', async () => {
  const s = setup(); let finish;
  s.response(() => new Promise(resolve => {finish = resolve;}));
  s.edit('title', 'First'); s.ids['deck-save'].handlers.click(); s.edit('title', 'Second');
  finish({ok: true, json: () => Promise.resolve({revision: 'r2'})}); await tick();
  assert.equal(s.ids['deck-status'].textContent, 'Unsaved'); assert.equal(s.ids['deck-save'].disabled, false);
});

test('conflicts and network errors preserve the editor and allow retry', async () => {
  const s = setup(); s.edit('title', 'My changes');
  s.response(() => Promise.resolve({status: 409})); s.ids['deck-save'].handlers.click(); await tick();
  assert.equal(s.ids['deck-status'].textContent, 'Conflict'); assert.equal(s.inputs.title.value, 'My changes');
  s.response(() => Promise.reject(new Error('network'))); s.ids['deck-save'].handlers.click(); await tick();
  assert.equal(s.ids['deck-status'].textContent, 'Error'); assert.equal(s.ids['deck-save'].disabled, false);
});

test('PDF export waits for readiness and blocks text that would be clipped', async () => {
  const s = setup(); s.ids['deck-pdf'].handlers.click(); await tick(); assert.equal(s.prints(), 1);
  s.body.scrollHeight = 1000; s.edit('bullets', 'Much too much text');
  assert.equal(s.ids['deck-overflow'].hidden, false);
  s.ids['deck-pdf'].handlers.click(); await tick(); assert.equal(s.prints(), 1);
});
