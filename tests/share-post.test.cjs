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
  const box = {
    dataset: {url: URL_, short: 'What it shows.\n\n#arXiv', copied: 'Copied', linkedinCopied: 'Copied for LinkedIn'},
    getAttribute: name => URL_ + '?utm_source=' + name.replace('data-link-', ''),
    querySelector: sel => (sel === '[data-share-text]' ? area : status),
  };
  const document = {addEventListener: (type, handler) => { if (type === 'click') clickHandler = handler; }, querySelectorAll: () => [], getElementById: () => null};
  const navigator = {clipboard: {writeText: text => { copied = text; return Promise.resolve(); }}, share};
  const window = {isSecureContext: true, addEventListener() {}};
  vm.runInNewContext(source, {document, navigator, window, location: {hash: ''}, Promise});
  function click(network) {
    const button = {dataset: {shareNetwork: network}, closest: sel => (sel === '[data-share]' ? box : button)};
    clickHandler({target: {closest: () => button}});
    return button;
  }
  return {click, copied: () => copied, status};
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
