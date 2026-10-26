const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync('static/summarizer/js/collection.js', 'utf8');

function node(props) {
  return Object.assign({hidden: false, disabled: false, dataset: {}, textContent: '', value: '', handlers: {},
    classList: {add() {}, remove() {}, toggle() {}},
    addEventListener(event, fn) { this.handlers[event] = fn; },
    click() { this.handlers.click && this.handlers.click(); },
    querySelector() { return null; }, querySelectorAll() { return []; }}, props);
}

// The page with the open questions section only, or with the paper list (cards: their datasets).
function setup({post, track, cards}) {
  const go = node({id: 'questions-go', dataset: {}});
  const again = node({id: 'questions-again', dataset: {mode: 'again'}});
  const nodes = {
    'collection': node({dataset: {base: '/en', pk: '5', max: '25'},
      querySelectorAll: s => s === '[data-run="questions"]' ? [go, again] : []}),
    'collection-text': node({dataset: {failed: 'Failed', premium: 'Premium', tooFew: 'Too few'}}),
    'questions-working': node({hidden: true}), 'questions-status': node(), 'questions-go': go, 'questions-again': again,
  };
  if (cards) {
    const list = node({children: cards.map(dataset => node({dataset}))});
    list.appendChild = c => { list.children.splice(list.children.indexOf(c), 1); list.children.push(c); };
    Object.assign(nodes, {'paper-list': list, 'paper-filter': node(), 'paper-tag': node(), 'paper-reading': node(),
      'paper-sort': node({value: 'order'}), 'paper-empty': node({hidden: true})});
  }
  const posted = [];
  vm.runInNewContext(source, {
    document: {getElementById: id => nodes[id] || null},
    SP: {post: (url, data) => { posted.push([url, data]); return post(url, data); }, track},
    location: {pathname: '/en/collections/5/', hash: '', reload() {}}, history: {replaceState() {}},
    window: {addEventListener() {}}, Promise, setTimeout, clearTimeout, encodeURIComponent, Number,
  });
  return {nodes, posted};
}
const tick = () => new Promise(resolve => setImmediate(resolve));
const titles = nodes => [...nodes['paper-list'].children.filter(c => !c.hidden).map(c => c.dataset.title)];

test('a generation failure restores the retry button and stops the spinner', async () => {
  const {nodes, posted} = setup({post: async () => ({id: 1}), track: (panel, id, done, failed) => failed()});
  nodes['questions-go'].click();
  await tick();
  assert.equal(posted[0][0], '/en/collections/5/questions/');
  assert.equal(posted[0][1].mode, '');
  assert.equal(nodes['questions-go'].disabled, false);
  assert.equal(nodes['questions-go'].hidden, false);
  assert.equal(nodes['questions-working'].hidden, true);
  assert.equal(nodes['questions-status'].textContent, 'Failed');
});

test('a failed start request also restores regeneration and hides progress', async () => {
  const {nodes, posted} = setup({post: async () => { throw new Error('offline'); }, track: () => assert.fail('must not poll')});
  nodes['questions-again'].click();
  await tick();
  assert.equal(posted[0][1].mode, 'again');
  assert.equal(nodes['questions-again'].disabled, false);
  assert.equal(nodes['questions-working'].hidden, true);
  assert.equal(nodes['questions-status'].textContent, 'Failed');
});

test('too few selected papers are said, not shown as a failure', async () => {
  const {nodes} = setup({post: async () => ({status: 'too_few'}), track: () => assert.fail('must not poll')});
  nodes['questions-go'].click();
  await tick();
  assert.equal(nodes['questions-status'].textContent, 'Too few');
});

test('papers are filtered by text, tag and reading status, and sorted with unknown values last', () => {
  const cards = [
    {order: '0', year: '2019', cited: '', title: 'b clouds', tags: 'core, method', reading: 'read', search: 'b clouds lee'},
    {order: '1', year: '', cited: '40', title: 'a winds', tags: '', reading: '', search: 'a winds ray'},
    {order: '2', year: '2023', cited: '3', title: 'c hazes', tags: 'core', reading: 'key', search: 'c hazes lee'},
  ];
  const {nodes} = setup({post: async () => ({}), track() {}, cards});
  const sort = nodes['paper-sort'];
  sort.value = 'newest'; sort.handlers.change();
  assert.deepEqual(titles(nodes), ['c hazes', 'b clouds', 'a winds']);
  sort.value = 'oldest'; sort.handlers.change();
  assert.deepEqual(titles(nodes), ['b clouds', 'c hazes', 'a winds']);
  sort.value = 'cited'; sort.handlers.change();
  assert.deepEqual(titles(nodes), ['a winds', 'c hazes', 'b clouds']);
  nodes['paper-tag'].value = 'core'; nodes['paper-tag'].handlers.change();
  assert.deepEqual(titles(nodes), ['c hazes', 'b clouds']);
  nodes['paper-reading'].value = 'key'; nodes['paper-reading'].handlers.change();
  assert.deepEqual(titles(nodes), ['c hazes']);
  nodes['paper-reading'].value = 'none'; nodes['paper-reading'].handlers.change();
  assert.deepEqual(titles(nodes), []);
  assert.equal(nodes['paper-empty'].hidden, false);
  nodes['paper-tag'].value = ''; nodes['paper-reading'].value = '';
  nodes['paper-filter'].value = ' LEE '; nodes['paper-filter'].handlers.input();
  assert.deepEqual(titles(nodes), ['c hazes', 'b clouds']);
});
