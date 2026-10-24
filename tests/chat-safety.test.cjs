const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {createHarness, element} = require('./chat-rendering-helper.cjs');

for (const template of ['chat', 'arxividpage']) {
  test(`${template}: streamed and final model output only use the safe Markdown renderer`, () => {
    const source = fs.readFileSync(template === 'chat' ? 'templates/summarizer/chat.html' : 'static/summarizer/js/paper-ready.js', 'utf8');
    const handler = template === 'arxividpage' ? source.match(/function receive\(event\) \{([\s\S]*?)\n    \}/) : source.match(/socketl\.addEventListener\("message", function\(event\) \{([\s\S]*?)\n\s*\}\);/);
    assert.ok(handler, 'Exercise the actual chat message handler');
    const elements = [];
    const {SPChat} = createHarness();
    const visualCalls = [];
    const SPChatFigures = {render: (...args) => visualCalls.push(args)};
    const context = {
      currentResponse: null,
      pendingFigures: null,
      checking: {},
      nudge: null,
      finished: 0,
      window: {SPChatFigures}, SPChatFigures,
      SPChat,
      chatbotMessages: {appendChild(node) {elements.push(node);}},
      messages: {appendChild(node) {elements.push(node);}},
      root: {dataset: {ai: 'AI -'}}, scroll() {}, finish() {context.finished++;},
      document: {createElement: element, getElementById: element, querySelector: () => null},
      MathJax: {typeset() {}}, scrollToBottom() {},
    };
    vm.createContext(context);
    vm.runInContext(`function handle(event) {${handler[1]}\n}`, context);
    const attack = '<img src=x onerror="alert(document.cookie)"><script>alert(1)</script>';
    context.handle({data: JSON.stringify({type: 'chat_figures', status: 'ready', figures: [{paper_id: 'paper'}]})});
    assert.equal(elements.length, 0);
    context.handle({data: JSON.stringify({type: 'chat_delta', delta: attack})});
    assert.equal(elements.length, 1);
    assert.ok(elements[0].children[1].innerHTML.includes('&lt;script&gt;'));
    assert.ok(!elements[0].children[1].innerHTML.includes('<img'));
    context.handle({data: JSON.stringify({message: attack})});
    assert.equal(elements.length, 1);
    assert.ok(elements[0].children[1].innerHTML.includes('&lt;script&gt;'));
    assert.ok(!elements[0].children[1].innerHTML.includes('<script'));
    assert.equal(context.currentResponse, null);
    assert.equal(context.pendingFigures, null);
    assert.equal(visualCalls.length, 1);
    assert.equal(visualCalls[0][0], elements[0]);
    assert.equal(visualCalls[0][2], 'ready');
    context.handle({data: JSON.stringify({message: 'A text-only answer'})});
    assert.equal(elements.length, 2);
    assert.equal(visualCalls.length, 1);

    // an answer whose claims are checked next: a notice, then the claims, both as text
    const finished = context.finished;
    context.handle({data: JSON.stringify({message: 'Checked answer', evidence: 'checking', answer_id: 7})});
    const answer = elements[2];
    assert.equal(answer.querySelector('.chat-evidence').className, 'chat-evidence chat-evidence-checking');
    if (template === 'arxividpage') assert.equal(context.finished, finished + 1, 'the next question does not wait for the check');
    context.handle({data: JSON.stringify({message: 'Next answer', evidence: 'checking', answer_id: 8})});
    context.handle({data: JSON.stringify({type: 'chat_evidence', status: 'done', answer_id: 7,
      claims: [{claim: attack, status: 'verified', quote: 'We find lenses.', link: 'javascript:alert(1)'}]})});
    const box = answer.querySelector('.chat-evidence');
    assert.equal(box.tagName, 'details');
    const claim = box.children[1].children[0];
    assert.equal(claim.children[0].textContent, attack);
    assert.ok(!claim.children.some(node => node.tagName === 'a'));
    assert.equal(elements[3].querySelector('.chat-evidence').className, 'chat-evidence chat-evidence-checking');
    context.handle({data: JSON.stringify({type: 'chat_evidence', status: 'unavailable', answer_id: 8})});
    assert.equal(elements[3].querySelector('.chat-evidence').className, 'chat-evidence chat-evidence-unavailable');
    // a late or repeated result has no answer to attach to
    context.handle({data: JSON.stringify({type: 'chat_evidence', status: 'unavailable', answer_id: 7})});
    assert.equal(answer.querySelector('.chat-evidence').tagName, 'details');
    assert.equal(elements.length, 4);
  });
}

test('math rendering loads the safe extension and forbids executable and local URLs', () => {
  const source = fs.readFileSync('templates/summarizer/mathjax.html', 'utf8');
  const context = {};
  vm.runInNewContext(source.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  assert.ok(context.MathJax.loader.load.includes('ui/safe'));
  assert.equal(context.MathJax.options.safeOptions.allow.URLs, 'safe');
  for (const protocol of ['javascript', 'data', 'file']) {
    assert.equal(context.MathJax.options.safeOptions.safeProtocols[protocol], false);
  }
  assert.equal(context.MathJax.options.safeOptions.safeProtocols.https, true);
  assert.equal(context.MathJax.tex.inlineMath[1][0], '\\(');
  for (const template of ['chat', 'arxividpage']) {
    assert.ok(fs.readFileSync(`templates/summarizer/${template}.html`, 'utf8').includes("include 'summarizer/mathjax.html'"));
  }
});
