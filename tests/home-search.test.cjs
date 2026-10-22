const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/summarizer/js/home.js', 'utf8');
function setup(options = {}) {
  const handlers = {}, windowHandlers = {}, attrs = {};
  const input = {value: '', focus() { this.focused = true; }};
  const button = {textContent: 'Search', disabled: false, dataset: {loading: 'Searching…'}};
  const status = {hidden: true};
  const radio = {checked: false};
  const example = {dataset: {query: 'Yoshua Bengio', field: 'author'}, addEventListener(type, fn) {this.click = fn;}};
  const examples = {hidden: true, querySelectorAll: () => [example]};
  const form = {
    elements: options.results ? {q: input} : {arxiv_id: input}, addEventListener: (type, fn) => {handlers[type] = fn;},
    setAttribute: (key, value) => {attrs[key] = value;},
    getAttribute: key => attrs[key], removeAttribute: key => {delete attrs[key];},
    querySelectorAll: selector => selector === 'button[type="submit"]' ? [button] :
      (options.orcid && selector.includes('input[name=orcid]') ? [{value: options.orcid}] : []),
    querySelector: selector => selector === 'button[type="submit"]' ? button :
      selector === '[data-search-status]' ? status : selector === '.search-examples' ? examples : radio,
  };
  vm.runInNewContext(source, {document: {querySelectorAll: () => [form]},
    window: {addEventListener: (type, fn) => {windowHandlers[type] = fn;}}});
  return {handlers, windowHandlers, attrs, input, button, status, example, examples, radio};
}
test('blank input focuses the field without entering a loading state', () => {
  const state = setup(); let prevented = false;
  state.input.value = '   ';
  state.handlers.submit({preventDefault() {prevented = true;}});
  assert.equal(prevented, true);
  assert.equal(state.input.focused, true);
  assert.equal(state.button.disabled, false);
});
test('loading is visible, prevents duplicate submits, and resets on back navigation', () => {
  const s = setup(); s.input.value = 'attention';
  s.handlers.submit({preventDefault() {assert.fail('valid submit');}});
  assert.equal(s.button.textContent, 'Searching…');
  assert.equal(s.status.hidden, false);
  assert.equal(s.attrs['aria-busy'], 'true');
  let prevented = false;
  s.handlers.submit({preventDefault() {prevented = true;}});
  assert.equal(prevented, true);
  s.windowHandlers.pageshow();
  assert.equal(s.button.textContent, 'Search');
  assert.equal(s.button.disabled, false);
  assert.equal(s.status.hidden, true);
  assert.equal(s.attrs['aria-busy'], undefined);
});
test('example fills the input and selects its scope without submitting', () => {
  const s = setup();
  assert.equal(s.examples.hidden, false);
  s.example.click();
  assert.equal(s.input.value, 'Yoshua Bengio');
  assert.equal(s.radio.checked, true);
  assert.equal(s.input.focused, true);
  assert.equal(s.attrs['aria-busy'], undefined);
});

test('results form allows an ORCID-only search with an empty query', () => {
  const s = setup({results: true, orcid: '0000-0001-6527-4684'});
  s.handlers.submit({preventDefault() {assert.fail('ORCID-only search must submit');}});
  assert.equal(s.attrs['aria-busy'], 'true');
  assert.equal(s.button.disabled, true);
});
