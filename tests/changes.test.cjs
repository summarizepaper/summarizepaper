const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
function setup(old, next) {
  const from = {value: String(old)}, to = {value: String(next)};
  let onChange, submitted = 0;
  const form = {querySelector: name => name.includes('from') ? from : to,
    addEventListener: (_, fn) => {onChange = fn;}, requestSubmit: () => {submitted++;}};
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/changes.js', 'utf8'), {
    document: {readyState: 'complete', getElementById: () => form}
  });
  return {from, to, change: target => onChange({target}), submissions: () => submitted};
}
test('changing the old version selects a later destination and submits the actual pair', () => {
  const page = setup(1, 2);
  page.from.value = '2'; page.change(page.from);
  assert.equal(page.to.value, '3'); assert.equal(page.submissions(), 1);
});
test('changing the destination keeps an earlier origin and reloads its saved comparison', () => {
  const page = setup(2, 3);
  page.to.value = '2'; page.change(page.to);
  assert.equal(page.from.value, '1'); assert.equal(page.submissions(), 1);
});
