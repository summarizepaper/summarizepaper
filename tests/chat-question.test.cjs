const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/summarizer/js/chat-page.js', 'utf8');
const sendQuestion = source.match(/  function ask\(question\) \{[\s\S]*?\n  \}/)[0];
const prefill = source.slice(source.indexOf('  // a question proposed by another tool')).replace(/\}\)\(\);\s*$/, '');
for (const search of ['', '?ask=Explain%20the%20figure']) {
  test(`chat can send after startup ${search ? 'with a proposed question' : 'without a proposed question'}`, () => {
    const sent = [], displayed = [];
    const context = {URLSearchParams, location: {search}, input: {value: ''}, busy: false,
      sendButton: {disabled: false}, readyIds: () => ['2311.08680v2'],
      asked: q => displayed.push(q), grow() {}, start() {}, send: p => sent.push(p)};
    vm.createContext(context);
    vm.runInContext(sendQuestion + '\n' + prefill, context);
    assert.equal(sent.length, 0, 'a proposed question must not be sent automatically');
    if (search) assert.equal(context.input.value, 'Explain the figure');
    else context.input.value = 'What does the figure show?';
    const question = context.input.value;
    vm.runInContext('ask(input.value)', context);
    assert.equal(sent.length, 1);
    assert.equal(sent[0].message, question);
    assert.deepEqual(sent[0].selectedpapers, ['2311.08680v2']);
    assert.equal(displayed[0], question);
    assert.equal(context.input.value, '');
  });
}
