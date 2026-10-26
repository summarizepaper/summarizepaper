const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag = '') {
    this.tagName = tag; this.children = []; this.events = {}; this.dataset = {};
    this.textContent = ''; this._value = ''; this.hidden = false; this.isConnected = true;
    const classes = new Set();
    this.classList = {add: x => classes.add(x), remove: x => classes.delete(x), contains: x => classes.has(x)};
  }
  addEventListener(type, fn) {this.events[type] = fn;}
  fire(type) {return this.events[type]?.();}
  appendChild(e) {this.children.push(e); return e;}
  replaceChildren(...children) {this.children = children; this._value = '';}
  setAttribute(key, value) {this[key] = key === 'open' ? true : value;}
  removeAttribute(key) {delete this[key];}
  focus() {this.focused = true;}
  get value() {return this._value || this.children[0]?.value || '';}
  set value(value) {this._value = value;}
  dispatchEvent() {this.inputEvents = (this.inputEvents || 0) + 1;}
  showModal() {this.open = true;}
  close() {this.open = false; this.fire('close');}
}
function harness({multi = false} = {}) {
  const box = new Element(), input = new Element(), requests = [], typeset = [], cleared = [];
  const elements = {};
  for (const key of ['select', '.chat-figures-picker', '.chat-figures-status', '.chat-figures-grid', '.chat-figures-retry',
    '.chat-figures-source', 'dialog', '.chat-figure-enlarged', '.chat-figure-close']) elements[key] = new Element();
  box.querySelector = key => elements[key];
  box.parentElement = {querySelector: key => elements[key]};
  elements.dialog.querySelector = key => elements[key];
  box.closest = () => ({querySelector: () => input});
  Object.assign(box.dataset, {paper: multi ? '' : '2401.04088', url: '/figures/PAPER_ID/',
    question: 'About {figure} in {paper} ({id}): ', none: 'Choose a paper', loading: 'Loading',
    error: 'Retry', pending: 'Preparing', empty: 'No figures', ask: 'Ask', enlarge: 'Zoom'});
  box.dataset.visionUsed = 'Figures supplied'; box.dataset.visionUnavailable = 'Text only';
  let papers = [], notify;
  const selected = {querySelectorAll: () => papers.map(([id, title]) => ({querySelector: key =>
    key === 'strong' ? {textContent: title} : {id}}))};
  const document = {createElement: tag => new Element(tag), querySelectorAll: () => [box],
    querySelector: () => ({textContent: 'Paper A'}), getElementById: () => multi ? selected : null};
  const window = {SPChat: {typeset: e => typeset.push(e)}, MathJax: {typesetClear: e => cleared.push(...e)}};
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/chat-figures.js', 'utf8'), {
    document, URL, AbortController, Event, Map,
    window,
    MutationObserver: class {constructor(fn) {notify = fn;} observe() {}},
    fetch: (url, options) => new Promise(resolve => requests.push({url, options,
      reply: data => resolve({json: () => Promise.resolve(data)})})),
  });
  return {box, input, requests, typeset, cleared, window, ...elements,
    open() {box.open = true; box.fire('toggle');},
    close() {box.open = false; box.fire('toggle');},
    papers(value) {papers = value; notify();}};
}
const settle = () => new Promise(resolve => setImmediate(resolve));
const ready = (title = 'Paper A') => ({status: 'ready', title, paper_id: '2401.04088v1',
  source_url: 'https://arxiv.org/abs/2401.04088v1', figures: [{label: 'Figure 3', caption: '<b>Untrusted caption</b>',
    images: ['https://evil.test/image.png', 'https://arxiv.org/html/2401.04088v1/x3.png',
      'https://arxiv.org/html/2401.04088v1/x4.png']}]});

test('lazy loading, trusted images, literal captions, multi-panel zoom and question draft', async () => {
  const h = harness();
  assert.equal(h.requests.length, 0);
  h.open();
  assert.equal(h.requests.length, 1);
  assert.match(h.requests[0].url, /2401\.04088v1/);
  h.requests[0].reply(ready()); await settle();
  const card = h['.chat-figures-grid'].children[0];
  assert.equal(card.children[0].children.length, 1);
  assert.equal(card.children[0].children[0].src, 'https://arxiv.org/html/2401.04088v1/x3.png');
  assert.equal(card.children[1].textContent, '<b>Untrusted caption</b>');
  card.children[0].fire('click');
  assert.equal(h.dialog.open, true);
  assert.equal(h['.chat-figure-enlarged'].children.filter(e => e.tagName === 'img').length, 2);
  h['.chat-figure-close'].fire('click');
  assert.equal(card.children[0].focused, true);
  h.input.value = 'What does it show?';
  card.children[2].fire('click');
  assert.match(h.input.value, /^About Figure 3 in Paper A \(2401\.04088v1\): What does it show\?/);
  card.children[2].fire('click');
  assert.equal(h.input.value.match(/About Figure 3/g).length, 1);
  assert.equal(h.input.inputEvents, 2);
  assert.equal(h.requests.length, 1); // drafting never sends a chat request
  h.close(); h.open(); assert.equal(h.requests.length, 1); // cached gallery
});

test('aborted load restarts on reopening and stale results never replace another paper', async () => {
  const h = harness({multi: true});
  h.papers([['2401.04088', 'A'], ['2401.04089v1', 'B']]);
  h.open(); h.close();
  assert.equal(h.requests[0].options.signal.aborted, true);
  h.open(); assert.equal(h.requests.length, 2);
  h.select.value = '2401.04089v1'; h.select.fire('change');
  h.requests[2].reply(ready('Paper B')); await settle();
  h.requests[1].reply(ready('Paper A')); await settle();
  assert.equal(h['.chat-figures-status'].textContent, 'Paper B');
  h.papers([]);
  assert.equal(h['.chat-figures-grid'].children.length, 0);
  assert.equal(h['.chat-figures-status'].textContent, 'Choose a paper');
  assert.equal(h['.chat-figures-status'].classList.contains('is-loading'), false);
});

test('preparing paper retries after preparation and errors offer explicit retry', async () => {
  const h = harness({multi: true});
  h.papers([['2401.04088v1', 'A']]); h.open();
  h.requests[0].reply({status: 'not_ready'}); await settle();
  assert.equal(h['.chat-figures-retry'].hidden, false);
  h.papers([['2401.04088v1', 'A']]);
  assert.equal(h.requests.length, 2);
  h.requests[1].reply({status: 'unavailable'}); await settle();
  h['.chat-figures-retry'].fire('click');
  h.requests[2].reply(ready()); await settle();
  assert.equal(h['.chat-figures-retry'].hidden, true);
});

test('unsafe source links and images are never rendered', async () => {
  const h = harness(); h.open();
  h.requests[0].reply({status: 'ready', source_url: 'javascript:alert(1)', figures: [
    {images: ['data:image/svg+xml,evil', 'http://arxiv.org/x.png', 'https://u:p@arxiv.org/x.png']}]});
  await settle();
  assert.equal(h['.chat-figures-grid'].children.length, 0);
  assert.equal(h['.chat-figures-source'].hidden, true);
});

test('single-paper galleries have no picker; multiple papers retain selection', () => {
  const single = harness();
  assert.equal(single['.chat-figures-picker'].hidden, true);
  assert.equal(single.select.disabled, true);
  const multi = harness({multi: true});
  multi.papers([['2401.04088v1', 'A']]);
  assert.equal(multi['.chat-figures-picker'].hidden, true);
  multi.papers([['2401.04088v1', 'A'], ['2401.04089v1', 'B']]);
  assert.equal(multi['.chat-figures-picker'].hidden, false);
  assert.equal(multi.select.disabled, false);
});

test('caption TeX stays literal until typeset in both thumbnails and visible dialog', async () => {
  const h = harness(); h.open();
  const data = ready();
  data.figures[0].caption = 'Noise \\(\\sigma\\) = 20 \\(\\mu\\)Jy beam\\({}^{-1}\\)';
  h.requests[0].reply(data); await settle();
  const card = h['.chat-figures-grid'].children[0];
  assert.equal(card.children[1].textContent, data.figures[0].caption);
  assert.ok(h.typeset.includes(h['.chat-figures-grid']));
  card.children[0].fire('click');
  assert.equal(h.dialog.open, true);
  assert.ok(h.typeset.includes(h['.chat-figure-enlarged']));
  assert.equal(h['.chat-figure-enlarged'].children.find(e => e.tagName === 'p').textContent, data.figures[0].caption);
  card.children[0].fire('click');
  assert.ok(h.cleared.includes(h['.chat-figure-enlarged']));
});

test('actual visual inputs appear inside the answer and enlarge with the gallery closed', () => {
  const h = harness(), answer = new Element();
  const data = ready();
  h.window.SPChatFigures.render(answer, [{paper_id: data.paper_id, title: data.title, figure: data.figures[0]}], 'ready');
  assert.equal(h.requests.length, 0);
  assert.equal(answer.children[0].children[0].textContent, 'Figures supplied');
  const card = answer.children[0].children[1].children[0];
  assert.equal(card.children[0].children.length, 1);
  card.children[0].fire('click');
  assert.equal(h.dialog.open, true);
  assert.notEqual(h.box.open, true);
  assert.match(card.children[3].textContent, /2401\.04088v1/);
  const fallback = new Element();
  h.window.SPChatFigures.render(fallback, [], 'unavailable');
  assert.equal(fallback.children[0].children[0].textContent, 'Text only');
  const plain = new Element();
  h.window.SPChatFigures.render(plain, [], 'not_requested');
  assert.equal(plain.children.length, 0);
});

test('both websocket handlers preserve figure metadata through the final answer', () => {
  for (const path of ['static/summarizer/js/chat-page.js', 'static/summarizer/js/paper-ready.js']) {
    const template = fs.readFileSync(path, 'utf8');
    assert.match(template, /data\.type === 'chat_figures'/);
    assert.match(template, /SPChatFigures\.render\(currentResponse, pendingFigures\.figures, pendingFigures\.status\)/);
    assert.match(template, /pendingFigures = null/);
  }
});
