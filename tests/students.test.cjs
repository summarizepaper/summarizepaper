const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
function setup() {
  const handlers = {}, windowHandlers = {};
  const status = {textContent: 'Saved', classList: {toggle(name, value) {this[name] = value;}}};
  const button = {disabled: false};
  const form = {dataset: {unsaved:'Unsaved',saving:'Saving'},
    elements: {status:{value:'explore'},notes:{value:'Initial notes'}},
    querySelector(selector) {return selector === '[data-student-save-status]' ? status : button;},
    addEventListener(name, fn) {handlers[name] = fn;}};
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/students.js','utf8'), {
    document:{readyState:'complete',querySelectorAll(){return [form];}},
    window:{addEventListener(name, fn){windowHandlers[name]=fn;}}});
  return {form,status,button,handlers,windowHandlers};
}
test('changing only the application status is explicitly unsaved; reverting clears it', () => {
  const {form,status,handlers} = setup();
  form.elements.status.value = 'contacted'; handlers.change();
  assert.equal(status.textContent,'Unsaved'); assert.equal(status.classList.notice,true);
  form.elements.status.value='explore'; handlers.change();
  assert.equal(status.textContent,'Saved'); assert.equal(status.classList.notice,false);
});
test('notes and status share the save action; navigating back restores the button and dirty state', () => {
  const {form,status,button,handlers,windowHandlers} = setup();
  form.elements.notes.value='Unsaved note'; handlers.input();
  assert.equal(status.textContent,'Unsaved');
  handlers.submit({submitter:{value:''}});
  assert.equal(button.disabled,true); assert.equal(status.textContent,'Saving');
  windowHandlers.pageshow();
  assert.equal(button.disabled,false); assert.equal(status.textContent,'Unsaved');
});
test('removing a lead does not claim that progress was saved', () => {
  const {status,button,handlers} = setup();
  handlers.submit({submitter:{value:'remove'}});
  assert.equal(button.disabled,false); assert.equal(status.textContent,'Saved');
});
