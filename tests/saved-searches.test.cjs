const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('activation shows progress, blocks duplicates and resets on back navigation', () => {
  const events = {}, pageEvents = {}, attrs = {};
  const button = {disabled: false, dataset: {loading: 'Enabling…'}};
  const label = {textContent: 'Enable email watch'};
  const spinner = {hidden: true}, status = {hidden: true};
  // The action is a hidden input, so disabling the button cannot drop it from POST.
  const elements = {'button[type="submit"]': button, '[data-watch-label]': label,
    '.watch-spinner': spinner, '[data-watch-status]': status};
  const form = {
    querySelector: key => elements[key],
    addEventListener: (key, fn) => {events[key] = fn;},
    getAttribute: key => attrs[key],
    setAttribute: (key, value) => {attrs[key] = value;},
    removeAttribute: key => {delete attrs[key];},
  };
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/saved-searches.js', 'utf8'), {
    document: {querySelectorAll: () => [form]},
    window: {addEventListener: (key, fn) => {pageEvents[key] = fn;}},
  });
  events.submit({preventDefault: () => assert.fail('first submission must continue')});
  assert.equal(attrs['aria-busy'], 'true');
  assert.equal(button.disabled, true);
  assert.equal(label.textContent, 'Enabling…');
  assert.equal(spinner.hidden, false);
  assert.equal(status.hidden, false);
  let prevented = false;
  events.submit({preventDefault: () => {prevented = true;}});
  assert.equal(prevented, true);
  pageEvents.pageshow();
  assert.equal(button.disabled, false);
  assert.equal(label.textContent, 'Enable email watch');
  assert.equal(spinner.hidden, true);
  assert.equal(status.hidden, true);
  assert.equal(attrs['aria-busy'], undefined);
});
