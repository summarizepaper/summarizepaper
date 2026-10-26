const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

// A small DOM: elements and text nodes, enough for chat-common.js
class Text {
  constructor(value) {this.nodeType = 3; this.nodeValue = value; this.parentNode = null;}
  get textContent() {return this.nodeValue;}
}
class Element {
  constructor(tag) {
    this.nodeType = 1; this.tagName = tag.toUpperCase(); this.className = ''; this.childNodes = []; this.parentNode = null;
    this.dataset = {}; this.listeners = {}; this.attributes = {}; this.hidden = false;
  }
  set textContent(value) {this.childNodes = []; if (value) this.appendChild(new Text(value));}
  get textContent() {return this.childNodes.map(n => n.textContent).join('');}
  appendChild(child) {return this.insertBefore(child, null);}
  insertBefore(child, before) {
    if (child.parentNode) child.parentNode.removeChild(child);
    const index = before ? this.childNodes.indexOf(before) : -1;
    if (index < 0) this.childNodes.push(child); else this.childNodes.splice(index, 0, child);
    child.parentNode = this;
    return child;
  }
  removeChild(child) {this.childNodes = this.childNodes.filter(n => n !== child); child.parentNode = null; return child;}
  all() {return this.childNodes.filter(n => n.nodeType === 1).flatMap(n => [n, ...n.all()]);}
  querySelector(selector) {return this.all().find(n => n.className.split(' ').includes(selector.slice(1))) || null;}
  addEventListener(type, callback) {this.listeners[type] = callback;}
  click() {this.listeners.click && this.listeners.click({});}
}
function load(extra = {}) {
  const document = {createElement: tag => new Element(tag), createTextNode: value => new Text(value)};
  const context = {window: {}, document, setTimeout, console, ...extra};
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/chat-common.js', 'utf8'), context);
  return {tools: context.window.SPChatTools, document};
}
function answer(html) {  // <div class="chatbot-response"><div class="chat-answer">...</div></div>
  const response = new Element('div'), body = new Element('div');
  body.className = 'chat-answer';
  html.forEach(part => body.appendChild(part));
  response.appendChild(body);
  return {response, body};
}
const papers = [{tag: '[Jiang et al. 2023]', url: '/en/paper/2310.06825v1/', title: 'Mistral 7B'},
                {tag: '[Vaswani et al. 2017]', url: '/en/paper/1706.03762v7/', title: 'Attention'}];

test('paper tags in the prose of an answer become links to the papers, once', () => {
  const {tools} = load();
  const code = new Element('code');
  code.textContent = '[Jiang et al. 2023]';
  const {response, body} = answer([new Text('As [Jiang et al. 2023] and [Vaswani et al. 2017] show. '), code]);
  tools.linkTags(response, papers);
  tools.linkTags(response, papers);  // again, as when the sources of the selection arrive later
  const links = body.childNodes.filter(n => n.tagName === 'A');
  assert.deepEqual(links.map(a => [a.textContent, a.href, a.title, a.className]),
    [['[Jiang et al. 2023]', '/en/paper/2310.06825v1/', 'Mistral 7B', 'chat-tag'], ['[Vaswani et al. 2017]', '/en/paper/1706.03762v7/', 'Attention', 'chat-tag']]);
  assert.equal(body.textContent, 'As [Jiang et al. 2023] and [Vaswani et al. 2017] show. [Jiang et al. 2023]');
  assert.equal(code.childNodes[0].nodeType, 3, 'code is left alone');
});

test('an unknown tag or a tag-like text stays text', () => {
  const {tools} = load();
  const {response, body} = answer([new Text('As [Smith 2020] says, see [1].')]);
  tools.linkTags(response, papers);
  assert.equal(body.childNodes.length, 1);
  assert.equal(body.childNodes[0].nodeType, 3);
});

test('Copy gives the answer and its sources, Ask again asks the same question', async () => {
  let copied = '';
  const asked = [];
  const SPChat = {raw: () => 'The answer.', claims: () => [
    {claim: 'A claim.', status: 'verified', paper: 'arXiv: 1', section: '3 Results', quote: 'We find it.'},
    {claim: 'Known.', status: 'general'}]};
  const {tools} = load({SPChat, navigator: {clipboard: {writeText: text => {copied = text; return Promise.resolve();}}}});
  const {response} = answer([new Text('The answer.')]);
  const evidence = new Element('details');
  evidence.className = 'chat-evidence';
  response.appendChild(evidence);
  const labels = {copy: 'Copy', copied: 'Copied', askAgain: 'Ask again', sources: 'Sources', exportGeneral: 'general knowledge'};
  tools.actions(response, {question: 'What is new?', ask: q => asked.push(q), labels});
  tools.actions(response, {question: 'What is new?', ask: q => asked.push(q), labels});
  const bars = response.childNodes.filter(n => n.className === 'chat-answer-actions');
  assert.equal(bars.length, 1);
  assert.equal(response.childNodes.indexOf(bars[0]), response.childNodes.indexOf(evidence) - 1, 'before the sources');
  const [copy, again] = bars[0].childNodes;
  copy.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(copied, 'The answer.\n\n**Sources**\n\n1. A claim. — arXiv: 1, 3 Results\n   > We find it.\n2. Known. (general knowledge)');
  assert.equal(copy.textContent, 'Copied');
  again.click();
  assert.deepEqual(asked, ['What is new?']);
});

test('the questions left today are shown, and hidden when there is no limit', async () => {
  const answers = [{left: 1}, {left: null}];
  const {tools} = load({fetch: () => Promise.resolve({json: () => Promise.resolve(answers.shift())})});
  const element = new Element('span');
  Object.assign(element.dataset, {left: '30', one: '1 question left today', many: '{n} questions left today'});
  const refresh = tools.quota(element, '/en/chat/quota/');
  assert.equal(element.textContent, '30 questions left today');
  assert.equal(element.hidden, false);
  refresh();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(element.textContent, '1 question left today');
  refresh();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(element.hidden, true);
});
