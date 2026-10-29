const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(dataset = {}) {this.dataset = dataset; this.hidden = false; this.events = {}; this.attrs = {};}
  addEventListener(name, fn) {this.events[name] = fn;}
  fire(name, event = {}) {this.events[name]?.(event);}
  setAttribute(name, value) {this.attrs[name] = value;}
  removeAttribute(name) {delete this.attrs[name];}
  focus(options) {this.focused = options || true;}
  scrollIntoView(options) {this.scrolled = options;}
}
function harness(hash = '', domain = 'biology') {
  const names = ['summary', 'figures', 'question'];
  const panels = names.map(panel => new Element({panel}));
  const buttons = names.map(step => new Element({step}));
  const headings = names.map(() => new Element());
  const next = [new Element({next: 'figures'}), new Element({next: 'question'})];
  const nav = new Element(), suggestions = new Element(), gallery = new Element();
  const evidence = new Element();
  const input = new Element(); input.value = 'Original draft';
  const choices = [new Element({question: 'Draft about Figure 2'})];
  const formats = ['summary', 'plain', 'points', 'blog', 'abstract'];
  const readingTabs = formats.map(reading => new Element({reading}));
  const readingPanels = formats.map(readingPanel => new Element({readingPanel}));
  const readingNav = new Element();
  const widget = {querySelector: () => readingNav, querySelectorAll: selector => selector === '[data-reading]' ? readingTabs : readingPanels};
  const root = {dataset: {domain}, querySelector(selector) {
    if (selector.startsWith('#step-')) return headings[names.indexOf(selector.split(' ')[0].slice(6))];
    return {'.getting-started-steps':nav, '.getting-started-suggestions':suggestions,
      '.chat-figures':gallery, 'textarea':input}[selector];
  }, querySelectorAll(selector) {
    return {'[data-panel]':panels, '[data-step]':buttons, '[data-next]':next,
      '.getting-started-suggestions [data-question]':choices, '.getting-started-evidence':[evidence], '[data-reading-widget]':[widget]}[selector];
  }};
  const events = {}, tracks = [];
  const location = {get hash() {return hash;}, set hash(value) {hash = value.startsWith('#') ? value : '#' + value;}};
  const window = {addEventListener: (name, fn) => events[name] = fn,
    umami: {track: (name, data) => tracks.push({name, data: {...data}})}};
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/getting-started.js', 'utf8'), {
    document: {querySelector: () => root}, location, window, Promise,
  });
  return {panels, buttons, headings, next, nav, gallery, evidence, input, choices, tracks, readingTabs, readingPanels, readingNav,
    change() {events.hashchange();}};
}

test('step navigation shows the requested content, focuses its heading and scrolls navigation below the header', () => {
  const h = harness();
  assert.deepEqual(h.panels.map(p => p.hidden), [false, true, true]);
  h.buttons[1].fire('click'); h.change();
  assert.deepEqual(h.panels.map(p => p.hidden), [true, false, true]);
  assert.equal(h.buttons[1].attrs['aria-current'], 'step');
  assert.equal(h.headings[1].focused.preventScroll, true);
  assert.equal(h.nav.scrolled.block, 'start');
  assert.equal(h.gallery.open, true);
  h.next[1].fire('click'); h.change();
  assert.deepEqual(h.panels.map(p => p.hidden), [true, true, false]);
  assert.equal(h.input.value, 'Original draft');
  h.choices[0].fire('click');
  assert.equal(h.input.value, 'Draft about Figure 2');
});

test('a return link opens and scrolls to the right step; unknown hashes fall back safely', () => {
  const h = harness('#question');
  assert.equal(h.panels[2].hidden, false);
  assert.equal(h.headings[2].focused.preventScroll, true);
  assert.equal(h.nav.scrolled.block, 'start');
  const invalid = harness('#arbitrary');
  assert.equal(invalid.panels[0].hidden, false);
});

test('source disclosures stay in place and track one actual opening without sending the draft', () => {
  const h = harness();
  h.evidence.open = false; h.evidence.fire('toggle');
  assert.equal(h.tracks.filter(t => t.name === 'getting_started_source_open').length, 0);
  h.evidence.open = true; h.evidence.fire('toggle'); h.evidence.fire('toggle');
  const opened = h.tracks.filter(t => t.name === 'getting_started_source_open');
  assert.equal(opened.length, 1);
  assert.deepEqual(opened[0].data, {step:'summary', domain:'biology'});
  assert.equal(h.evidence.scrolled, undefined);
  assert.equal(h.input.value, 'Original draft');
  assert.doesNotMatch(JSON.stringify(h.tracks), /Original draft|Draft about/);
});

test('reading formats switch real panels, keep a single active tab and support keyboard navigation', () => {
  const h = harness();
  assert.deepEqual(h.readingPanels.map(p => p.hidden), [false, true, true, true, true]);
  h.readingTabs[3].fire('click');
  assert.deepEqual(h.readingPanels.map(p => p.hidden), [true, true, true, false, true]);
  assert.equal(h.readingTabs[3].attrs['aria-selected'], 'true');
  assert.equal(h.readingTabs[0].attrs.tabindex, '-1');
  let prevented = false;
  h.readingTabs[3].fire('keydown', {key:'ArrowRight', preventDefault() {prevented = true;}});
  assert.equal(prevented, true);
  assert.equal(h.readingPanels[4].hidden, false);
  assert.equal(h.readingTabs[4].focused, true);
  h.readingTabs[4].fire('keydown', {key:'Home', preventDefault() {}});
  assert.equal(h.readingPanels[0].hidden, false);
  assert.equal(h.input.value, 'Original draft');
  assert.deepEqual(h.tracks.filter(t => t.name === 'getting_started_reading_format').map(t => t.data.format), ['blog', 'abstract', 'summary']);
});
