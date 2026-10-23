const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
function setup(post, track) {
  const nodes = {};
  for (const id of ['questions-status','questions-working','questions-go','questions-again']) {
    nodes[id] = {hidden: false, disabled: false, dataset: {}, textContent: '',
      addEventListener(event, fn) {this[event] = fn;}};
  }
  let source = fs.readFileSync('templates/summarizer/collection.html','utf8')
    .split("  var qStatus =")[1].split("  var zotero =")[0];
  source = ('var qStatus ='+source).replace(/\{% if questions and questions.status == 'pending' %\}[\s\S]*?\{% endif %\}/,'');
  vm.runInNewContext(source, {document: {getElementById: id => nodes[id]},
    SP: {post, track}, base: '/en', t: {failed: 'Failed', premium: 'Premium'}, location: {reload() {}}});
  return nodes;
}
const tick = () => new Promise(resolve => setImmediate(resolve));
test('a generation failure restores the retry button and stops the spinner', async () => {
  const nodes = setup(async () => ({id: 1}), (panel, id, done, failed) => failed());
  nodes['questions-go'].click();
  await tick();
  assert.equal(nodes['questions-go'].disabled, false);
  assert.equal(nodes['questions-go'].hidden, false);
  assert.equal(nodes['questions-working'].hidden, true);
  assert.equal(nodes['questions-status'].textContent, 'Failed');
});
test('a failed start request also restores regeneration and hides progress', async () => {
  const nodes = setup(async () => {throw new Error('offline');}, () => assert.fail('must not poll'));
  nodes['questions-again'].click();
  await tick();
  assert.equal(nodes['questions-again'].disabled, false);
  assert.equal(nodes['questions-working'].hidden, true);
  assert.equal(nodes['questions-status'].textContent, 'Failed');
});
