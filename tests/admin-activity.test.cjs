const {test} = require('node:test');
const assert = require('node:assert/strict');
const feed = require('../static/summarizer/js/admin-activity.js');

const events = [  // latest first, as the page sends them
  {t: '10:40', k: 'tool', x: 'Journal club · 2410.1', p: 'u7', w: 'ada@example.org'},
  {t: '10:30', k: 'visit', x: 'Journal club', p: 'u7', d: ''},
  {t: '10:05', k: 'visit', x: 'Accueil', p: 'u7', d: 'depuis arxiv.org · première page'},
  {t: '09:00', k: 'question', x: 'What is the limit?', p: '', n: true},
  {t: '08:00', k: 'visit', x: 'Recherche · « dark matter »', p: 'vabc', d: 'depuis google.com'},
];

test('time order leaves the page views to their own filter', () => {
  assert.deepEqual(feed.filter(events, {kind: ''}, 'time').map(e => e.k), ['tool', 'question']);
  assert.equal(feed.filter(events, {kind: 'visit'}, 'time').length, 3);
  assert.equal(feed.filter(events, {kind: ''}, 'person').length, 5);
});

test('members only keeps the pages a member opened before signing in, the text search reads every field', () => {
  assert.deepEqual(feed.filter(events, {kind: '', members: true}, 'person').map(e => e.t), ['10:40', '10:30', '10:05']);
  assert.deepEqual(feed.filter(events, {kind: '', q: 'GOOGLE'}, 'person').map(e => e.p), ['vabc']);
});

test('per person: each path in time order, where it came from, the unattached actions last', () => {
  const people = feed.group(events);
  assert.deepEqual(people.map(p => p.key), ['u7', 'vabc', '']);
  assert.equal(people[0].who, 'ada@example.org');
  assert.deepEqual(people[0].events.map(e => e.t), ['10:05', '10:30', '10:40']);
  assert.equal(people[0].origin, 'depuis arxiv.org · première page');
  assert.deepEqual(people[0].counts, {tool: 1, visit: 2});
});

test('the filters live in the address, and only our own paths become links', () => {
  const search = feed.writeState('?jour=2026-11-04', {kind: 'tool', members: true, q: 'x', view: 'person'});
  assert.equal(search, '?jour=2026-11-04&type=tool&membres=1&q=x&vue=personnes');
  assert.deepEqual(feed.readState(search), {kind: 'tool', members: true, q: 'x', view: 'person'});
  assert.equal(feed.writeState('?jour=2026-11-04&type=tool', {kind: '', members: false, q: '', view: 'time'}), '?jour=2026-11-04');
  assert.equal(feed.safeHref('/fr/paper/1/'), '/fr/paper/1/');
  assert.equal(feed.safeHref('//evil.example/'), '');
  assert.equal(feed.safeHref('javascript:alert(1)'), '');
});
