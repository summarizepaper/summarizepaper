const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function harness({upgrade = true, buttonPresent = true} = {}) {
  const source = fs.readFileSync('templates/summarizer/presskit.html', 'utf8');
  const script = source.match(/<script>([\s\S]*?)<\/script>/)[1]
    .replace(/{% if row and row.status == 'pending' %}[\s\S]*?{% endif %}/g, '')
    .replace('{{ LANGUAGE_CODE }}', 'en').replace('{{ arxiv_id|escapejs }}', '2101.00001v1');
  const events = {}, clicks = {}, requests = [], tracking = [];
  const button = {dataset: upgrade ? {again: '1'} : {}, disabled: false,
    addEventListener: (type, fn) => {clicks[type] = fn;}};
  const start = {hidden: false}, panel = {dataset: {}}, error = {hidden: true};
  const nodes = {'kit-start': start, 'kit-working': panel, 'kit-request-error': error,
    'kit-go': buttonPresent ? button : null};
  const SP = {post: (url, data) => new Promise((resolve, reject) => requests.push({url, data, resolve, reject})),
    track: (...args) => tracking.push(args)};
  let reloads = 0;
  vm.runInNewContext(script, {window: {addEventListener: (type, fn) => {events[type] = fn;}},
    document: {getElementById: id => nodes[id]}, SP, location: {reload: () => {reloads++;}}});
  events.DOMContentLoaded();
  return {button, start, panel, error, requests, tracking, clicks, reloads: () => reloads};
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test('enrichment requires a click, prevents duplicates and restores retry after failed work', async () => {
  const h = harness();
  assert.equal(h.requests.length, 0);
  h.clicks.click(); h.clicks.click();
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].data.again, '1');
  assert.equal(h.button.disabled, true);
  h.requests[0].resolve({id: 42}); await settle();
  assert.equal(h.start.hidden, true);
  h.tracking[0][3]();
  assert.equal(h.start.hidden, false);
  assert.equal(h.error.hidden, false);
  assert.equal(h.button.disabled, false);
  h.clicks.click();
  h.requests[1].reject(new Error('offline')); await settle();
  assert.equal(h.button.disabled, false);
  assert.equal(h.error.hidden, false);
});

test('new kits use the normal action; ready kits never regenerate on page load', async () => {
  const h = harness({upgrade: false});
  h.clicks.click();
  assert.equal(Object.keys(h.requests[0].data).length, 0);
  h.requests[0].resolve({id: 1}); await settle();
  h.tracking[0][2]();
  assert.equal(h.reloads(), 1);
  const ready = harness({buttonPresent: false});
  assert.equal(ready.requests.length, 0);
  assert.equal(ready.tracking.length, 0);
});
