const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/summarizer/js/slides-language.js', 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));

function setup(pending = false) {
  const form = {dataset: {startUrl: '/en/paper/start/', error: 'Retry', returnUrl: '/en/paper/?deck_lang=fr&deck_version=2'},
    setAttribute() {}, addEventListener(name, fn) {this[name] = fn;}};
  if (pending) form.dataset.pendingId = '7';
  const select = {value: 'fr', disabled: false}, button = {disabled: false};
  const error = {hidden: true}, panel = {dataset: {}}, start = {hidden: false};
  const elements = {'deck-language-form': form, 'deck-language': select, 'deck-go': button,
    'deck-language-error': error, 'deck-working': panel, 'deck-start': start};
  const posts = [], tracks = [], visits = [];
  let ready, reply = () => Promise.resolve({status: 'pending', id: 7, url: form.dataset.returnUrl});
  vm.runInNewContext(source, {document: {addEventListener(name, fn) {ready = fn;}, getElementById: id => elements[id]},
    window: {location: {assign: url => visits.push(url)}},
    SP: {post(url, data) {posts.push({url, data}); return reply();},
      track(panel, id, done, failed) {tracks.push({panel, id, done, failed});}}});
  ready();
  return {form, select, button, error, panel, start, posts, tracks, visits,
    reply(fn) {reply = fn;}, submit() {form.submit({preventDefault() {}});}};
}

test('generation sends the chosen content language and follows its target URL', async () => {
  const s = setup(); s.submit(); s.submit(); await tick();
  assert.equal(s.posts.length, 1); assert.equal(s.posts[0].data.deck_lang, 'fr');
  assert.equal(s.select.disabled, true); assert.equal(s.button.disabled, true);
  assert.equal(s.tracks[0].id, 7); s.tracks[0].done();
  assert.equal(s.visits[0], '/en/paper/?deck_lang=fr&deck_version=2');
  assert.equal(s.select.disabled, false); // Cancelling an unsaved-changes navigation must not lock the form.
});

test('an existing ready language version opens without polling or regenerating', async () => {
  const s = setup(); s.reply(() => Promise.resolve({status: 'done', url: '/en/paper/?deck_lang=en&deck_version=2'}));
  s.select.value = 'en'; s.submit(); await tick();
  assert.equal(s.posts[0].data.deck_lang, 'en'); assert.equal(s.tracks.length, 0);
  assert.equal(s.visits[0], '/en/paper/?deck_lang=en&deck_version=2');
});

test('request failures allow retry without discarding the language choice', async () => {
  const s = setup(); s.reply(() => Promise.resolve({status: 'invalid_language'})); s.submit(); await tick();
  assert.equal(s.error.hidden, false); assert.equal(s.error.textContent, 'Retry');
  assert.equal(s.button.disabled, false); assert.equal(s.select.value, 'fr');
  s.reply(() => Promise.reject(new Error('network'))); s.submit(); await tick();
  assert.equal(s.select.disabled, false); assert.equal(s.posts.length, 2);
});

test('failed background generation unlocks the controls for retry', async () => {
  const s = setup(); s.submit(); await tick(); s.tracks[0].failed();
  assert.equal(s.button.disabled, false); assert.equal(s.select.disabled, false);
  assert.equal(s.error.hidden, false); assert.equal(s.visits.length, 0);
});

test('reopening a pending version follows the same language without another model request', () => {
  const s = setup(true);
  assert.equal(s.posts.length, 0); assert.equal(s.tracks[0].id, 7);
  assert.equal(s.select.disabled, true); s.tracks[0].done();
  assert.equal(s.visits[0], s.form.dataset.returnUrl);
});
