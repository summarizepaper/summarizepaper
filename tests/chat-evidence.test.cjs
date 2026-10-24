const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function element(tag) {
  return {tagName: tag, className: '', textContent: '', dataset: {}, children: [], parent: null,
    appendChild(node) {node.parent = this; this.children.push(node); return node;},
    append(...nodes) {nodes.forEach(node => this.appendChild(node));},
    replaceChildren(...nodes) {this.children = []; this.append(...nodes);},
    remove() {this.parent.children = this.parent.children.filter(node => node !== this);},
    querySelector(selector) {return all(this).find(node => node.className.split(' ').includes(selector.slice(1)));}};
}
function all(node) {return node.children.flatMap(child => [child, ...all(child)]);}
function harness(labels, createElement = element) {
  const document = {createElement, addEventListener() {}, querySelectorAll() {return [];},
    getElementById(id) {return id === 'chat-evidence-labels' && labels ? {dataset: labels} : null;}};
  const context = {window: {markdownit: null}, document, URL, console};
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/chat-rendering.js', 'utf8'), context);
  return context.window.SPChat;
}
const verified = {claim: 'Rubin finds lenses.', status: 'verified', paper: 'arXiv: 2406.08919v3', section: '3.1 Counts',
  paragraph: 2, figure: 'Figure 1', quote: 'We find 170000 lenses.', numbers_absent: [], gapped: false,
  link: 'https://arxiv.org/html/2406.08919v3#:~:text=We%20find'};

test('each claim shows its status, place in the paper, passage and link', () => {
  const SPChat = harness(), answer = element('div');
  SPChat.evidence(answer, {status: 'done', claims: [verified, {claim: 'Invented.', status: 'not_found', paper: 'arXiv: 2406.08919v3'}]});
  const box = answer.querySelector('.chat-evidence');
  assert.equal(box.tagName, 'details');
  assert.match(box.children[0].textContent, /1 of 2 claims/);
  const [first, second] = box.children[1].children;
  assert.equal(first.className, 'chat-claim chat-claim-verified');
  assert.equal(first.querySelector('.chat-claim-where').textContent,
    'Passage found in the paper · Section “3.1 Counts” · paragraph 2 · Figure 1');
  assert.equal(first.children.find(n => n.tagName === 'blockquote').textContent, 'We find 170000 lenses.');
  assert.equal(first.children.find(n => n.tagName === 'a').href, verified.link);
  assert.equal(second.querySelector('.chat-claim-where').textContent, 'Passage not found in the paper: check this claim');
  assert.ok(!second.children.some(n => n.tagName === 'blockquote' || n.tagName === 'a'));
});

test('claims stay text and only arXiv links are kept', () => {
  const SPChat = harness(), answer = element('div');
  SPChat.evidence(answer, {status: 'done', claims: [{...verified, claim: '<img src=x onerror=alert(1)>',
    link: 'javascript:alert(1)', status: 'bogus'}]});
  const item = answer.querySelector('.chat-claim');
  assert.equal(item.className, 'chat-claim chat-claim-unsupported');
  assert.equal(item.querySelector('.chat-claim-text').textContent, '<img src=x onerror=alert(1)>');
  assert.ok(!item.children.some(n => n.tagName === 'a'));
});

test('checking is replaced by the result, and translated labels are used', () => {
  const SPChat = harness({checking: 'Vérification…', verified: 'Passage retrouvé', section: 'Section « {name} »'});
  const answer = element('div');
  SPChat.evidence(answer, {status: 'checking'});
  assert.equal(answer.querySelector('.chat-evidence').textContent, 'Vérification…');
  SPChat.evidence(answer, {status: 'done', claims: [verified]});
  assert.equal(answer.children.length, 1);
  assert.match(answer.querySelector('.chat-claim-where').textContent, /^Passage retrouvé · Section « 3.1 Counts »/);
});

test('an answer without factual claims gets no sources block', () => {
  const SPChat = harness(), answer = element('div');
  SPChat.evidence(answer, {status: 'checking'});
  assert.equal(SPChat.evidence(answer, {status: 'done', claims: []}), null);
  assert.equal(answer.children.length, 0);
});

test('the paper is named only when the answer draws on several papers', () => {
  const SPChat = harness(), answer = element('div');
  SPChat.evidence(answer, {status: 'done', claims: [verified, {...verified, paper: 'arXiv: 2101.00001v1'}]});
  assert.match(answer.querySelector('.chat-claim-where').textContent, /· arXiv: 2406\.08919v3$/);
});

// A DOM with text nodes, for the [n] placed in the answer.
function text(value) {
  return {nodeType: 3, nodeValue: value, parentNode: null,
    splitText(offset) {
      const rest = text(this.nodeValue.slice(offset));
      this.nodeValue = this.nodeValue.slice(0, offset);
      this.parentNode.insertBefore(rest, this.nextSibling());
      return rest;
    },
    nextSibling() {const nodes = this.parentNode.childNodes; return nodes[nodes.indexOf(this) + 1] || null;}};
}
function tree(tag, className, ...nodes) {
  const node = element(tag);
  node.nodeType = 1; node.tagName = tag.toUpperCase(); node.className = className || ''; node.childNodes = [];
  node.addEventListener = (type, callback) => {node.onclick = callback;};
  node.insertBefore = (child, before) => {
    child.parentNode = node;
    const at = before ? node.childNodes.indexOf(before) : -1;
    node.childNodes.splice(at < 0 ? node.childNodes.length : at, 0, child);
  };
  nodes.forEach(child => node.insertBefore(child, null));
  return node;
}
function textOf(node) {
  return node.childNodes.map(child => child.nodeType === 3 ? child.nodeValue
    : child.tagName === 'SUP' ? '<' + child.children[0].textContent + '>' : textOf(child)).join('');
}

test('each claim is numbered in the answer where its sentence ends', () => {
  const SPChat = harness(null, tag => tree(tag, ''));
  const paragraph = tree('p', '', text('Rubin finds '), tree('strong', '', text('170 000')), text(' lenses in ten years. It\nwill transform lensing, see '),
    tree('code', '', text('will transform lensing')), text('.'));
  const body = tree('div', 'chat-answer', paragraph);
  const answer = element('div');
  answer.appendChild(body);
  SPChat.evidence(answer, {status: 'done', claims: [
    {...verified, anchor: 'lenses in ten years'}, {...verified, anchor: 'will transform lensing'},
    {...verified, anchor: 'Rubin finds 170 000'}, {...verified, anchor: ''}]});
  assert.equal(textOf(paragraph), 'Rubin finds 170 000 lenses in ten years<[1]>. It\nwill transform lensing<[2]>, see will transform lensing.');
  const box = answer.querySelector('.chat-evidence');
  const ref = paragraph.childNodes.find(node => node.tagName === 'SUP').children[0];
  assert.equal(ref.href, '#' + box.children[1].children[0].id);
  box.open = false;
  ref.onclick();
  assert.equal(box.open, true);
});

test('the abstract is named in the reader\'s language', () => {
  const SPChat = harness({abstract: 'Résumé scientifique'}), answer = element('div');
  SPChat.evidence(answer, {status: 'done', claims: [{...verified, section: 'Abstract', paragraph: null, figure: ''}]});
  assert.equal(answer.querySelector('.chat-claim-where').textContent, 'Passage found in the paper · Résumé scientifique');
});

test('numbers the paper never states and interrupted passages are said under the claim', () => {
  const SPChat = harness({numbers: 'Nombres absents : {numbers}'}), answer = element('div');
  SPChat.evidence(answer, {status: 'done', claims: [{...verified, numbers_absent: ['250000', '0.5'], gapped: true}]});
  const item = answer.querySelector('.chat-claim');
  assert.equal(item.querySelector('.chat-claim-warning').textContent, 'Nombres absents : 250000, 0.5');
  assert.match(item.querySelector('.chat-claim-note').textContent, /interrupts this passage/);
});
