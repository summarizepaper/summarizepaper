const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/summarizer/js/share-post.js', 'utf8');

const URL_ = 'https://www.summarizepaper.com/en/paper/2401.04088v1/';
const TEXT = 'Title\n\nWhat it shows.\n\nRead it: ' + URL_ + '\n\n#arXiv';

function setup({edited = null, share = null} = {}) {
  let clickHandler, copied = null;
  const status = {textContent: ''};
  const area = {defaultValue: TEXT, value: edited === null ? TEXT : edited};
  const fetched = [];
  const box = {
    id: 'share-topic-2', open: false, dataset: {url: URL_, short: 'What it shows.\n\n#arXiv', subject: 'Title', copied: 'Copied', linkedinCopied: 'Copied for LinkedIn', image: 'https://example.org/og.png'},
    getAttribute: name => URL_ + '?utm_source=' + name.replace('data-link-', ''),
    hasAttribute: name => name === 'data-share',
    querySelector: sel => (sel === '[data-share-text]' ? area : status),
    scrollIntoView() {},
  };
  const document = {readyState: 'complete', addEventListener: (type, handler) => { if (type === 'click') clickHandler = handler; },
                    querySelectorAll: () => [], querySelector: () => null, getElementById: id => (id === box.id ? box : null)};
  const navigator = {clipboard: {writeText: text => { copied = text; return Promise.resolve(); }}, share};
  const window = {isSecureContext: true, addEventListener() {}};
  const context = {document, navigator, window, location: {hash: ''}, Promise, fetch: url => { fetched.push(url); return Promise.resolve(); }};
  vm.runInNewContext(source, context);
  vm.runInNewContext(source, context);  // a second box of the page loads the file again: one handler only
  function click(network) {
    const button = {dataset: {shareNetwork: network}, closest: sel => (sel === '[data-share]' ? box : sel.indexOf('data-share-open') !== -1 ? null : button)};
    clickHandler({target: {closest: sel => button.closest(sel)}});
    return button;
  }
  function openFrom(id) {
    const opener = {dataset: {shareOpen: id}};
    let prevented = false;
    clickHandler({preventDefault: () => { prevented = true; }, target: {closest: sel => (sel.indexOf('data-share-open') !== -1 ? opener : null)}});
    return prevented;
  }
  return {click, openFrom, box, fetched, copied: () => copied, status};
}

test('X gets the short post and the link tagged for X, apart', () => {
  const {click} = setup();
  const href = new URL(click('x').href);
  assert.equal(href.searchParams.get('text'), 'What it shows.\n\n#arXiv');
  assert.equal(href.searchParams.get('url'), URL_ + '?utm_source=x');
});

test('LinkedIn gets the whole post with its tagged link, also copied', async () => {
  const {click, copied} = setup();
  const text = new URL(click('linkedin').href).searchParams.get('text');
  assert.equal(text, TEXT.replace(URL_, URL_ + '?utm_source=linkedin'));
  await Promise.resolve();
  assert.equal(copied(), text);
});

test('an edited post goes to X and Bluesky too, the link added when removed', () => {
  const {click} = setup({edited: 'My own words about it.'});
  assert.equal(new URL(click('x').href).searchParams.get('text'), 'My own words about it.');
  assert.equal(new URL(click('bluesky').href).searchParams.get('text'), 'My own words about it.\n\n' + URL_ + '?utm_source=bluesky');
});

test('copy puts the post with the copy link in the clipboard and says so', async () => {
  const {click, copied, status} = setup();
  click('copy');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(copied(), TEXT.replace(URL_, URL_ + '?utm_source=copy'));
  assert.equal(status.textContent, 'Copied');
});

test('the phone share sheet gets the text and the link apart', () => {
  let shared = null;
  const {click} = setup({share: data => { shared = data; return Promise.resolve(); }});
  click('native');
  assert.equal(shared.url, URL_ + '?utm_source=native');
  assert.equal(shared.text, 'Title\n\nWhat it shows.\n\nRead it:\n\n#arXiv');
});

test('an email gets the subject and the whole post with its tagged link', () => {
  const {click} = setup();
  const href = click('email').href;
  assert.ok(href.startsWith('mailto:?subject=Title&body='));
  assert.equal(decodeURIComponent(href.split('body=')[1]), TEXT.replace(URL_, URL_ + '?utm_source=email'));
});

test("a card's Share button opens its box and asks for its picture once", () => {
  const {openFrom, box, fetched} = setup();
  assert.equal(openFrom('share-topic-2'), true);
  assert.equal(box.open, true);
  openFrom('share-topic-2');
  assert.deepEqual(fetched, ['https://example.org/og.png']);
});
