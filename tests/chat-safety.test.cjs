const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

for (const template of ['chat', 'arxividpage']) {
  test(`${template}: streamed and final model output never become HTML`, () => {
    const source = fs.readFileSync(`templates/summarizer/${template}.html`, 'utf8');
    const handler = source.match(/socketl\.addEventListener\("message", function\(event\) \{([\s\S]*?)\n\s*\}\);/);
    assert.ok(handler, 'Exercise the actual chat message handler');
    const elements = [];
    function element() {
      return {innerText: '', style: {}, classList: {add() {}},
        set innerHTML(_) {throw new Error('Untrusted HTML insertion');}};
    }
    const context = {
      currentResponse: null,
      chatbotMessages: {appendChild(node) {elements.push(node);}},
      document: {createElement: element, getElementById: element, querySelector: () => null},
      MathJax: {typeset() {}}, scrollToBottom() {},
    };
    vm.createContext(context);
    vm.runInContext(`function handle(event) {${handler[1]}\n}`, context);
    const attack = '<img src=x onerror="alert(document.cookie)"><script>alert(1)</script>';
    context.handle({data: JSON.stringify({type: 'chat_delta', delta: attack})});
    assert.equal(elements.length, 1);
    assert.ok(elements[0].innerText.includes(attack));
    context.handle({data: JSON.stringify({message: attack})});
    assert.equal(elements.length, 1);
    assert.ok(elements[0].innerText.endsWith(attack));
    assert.equal(context.currentResponse, null);
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
