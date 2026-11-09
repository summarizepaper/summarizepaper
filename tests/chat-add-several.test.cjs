const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/summarizer/js/chat-page.js', 'utf8');
const references = source.match(/  function references\(value\) \{[\s\S]*?\n  \}/)[0];

test('a pasted list of references gives each identifier or link once', () => {
  const context = {};
  vm.createContext(context);
  vm.runInContext(references, context);
  const split = value => Array.from(vm.runInContext('references(' + JSON.stringify(value) + ')', context));
  assert.deepEqual(split('2310.06825'), ['2310.06825']);
  assert.deepEqual(split(' https://arxiv.org/abs/1706.03762, 10.1038/s41586-021-03819-2 ;arXiv:2401.12345 '),
                   ['https://arxiv.org/abs/1706.03762', '10.1038/s41586-021-03819-2', 'arXiv:2401.12345']);
  // the line breaks a text field drops would glue the links together
  assert.deepEqual(split('https://arxiv.org/abs/1706.03762https://alphaxiv.org/abs/2401.12345'),
                   ['https://arxiv.org/abs/1706.03762', 'https://alphaxiv.org/abs/2401.12345']);
  assert.deepEqual(split('   '), []);
});
