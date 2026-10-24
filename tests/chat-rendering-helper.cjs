const fs = require('node:fs');
const vm = require('node:vm');
const markdownit = require('../static/summarizer/vendor/markdown-it/markdown-it.min.js');

function element(tag) {
  return {tagName: tag, className: '', textContent: '', innerText: '', innerHTML: '', children: [], dataset: {}, style: {},
    classList: {add() {}}, appendChild(node) {node.parent = this; this.children.push(node);},
    append(...nodes) {nodes.forEach(node => this.appendChild(node));},
    replaceChildren(...nodes) {this.children = []; this.append(...nodes);},
    remove() {this.parent.children = this.parent.children.filter(node => node !== this);},
    querySelector(selector) {return this.children.find(node => (node.className || '').split(' ').includes(selector.slice(1)));}};
}
function createHarness({library = true, history = []} = {}) {
  const handlers = {};
  const document = {createElement: element,
    addEventListener(type, callback) {handlers[type] = callback;},
    querySelectorAll() {return history;}, getElementById() {return null;}};
  const context = {window: {markdownit: library ? markdownit : null}, document, console};
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/chat-rendering.js', 'utf8'), context);
  return {SPChat: context.window.SPChat, window: context.window, document, handlers};
}
module.exports = {createHarness, element};
