const fs = require('node:fs');
const vm = require('node:vm');
const markdownit = require('../static/summarizer/vendor/markdown-it/markdown-it.min.js');

function element() {
  return {textContent: '', innerText: '', innerHTML: '', children: [], dataset: {}, style: {},
    classList: {add() {}}, appendChild(node) {this.children.push(node);},
    replaceChildren(...nodes) {this.children = nodes;},
    querySelector() {return this.children.find(node => node.className === 'chat-answer');}};
}
function createHarness({library = true, history = []} = {}) {
  const handlers = {};
  const document = {createElement: element,
    addEventListener(type, callback) {handlers[type] = callback;},
    querySelectorAll() {return history;}};
  const context = {window: {markdownit: library ? markdownit : null}, document, console};
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/chat-rendering.js', 'utf8'), context);
  return {SPChat: context.window.SPChat, window: context.window, document, handlers};
}
module.exports = {createHarness, element};
