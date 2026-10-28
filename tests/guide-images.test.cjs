const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
function harness(supported = true) {
  const events = {}, listeners = {};
  const image = {}, caption = {}, thumbnail = {src: '/capture.jpg', alt: 'Example'};
  const link = {isConnected: true, querySelector: () => thumbnail,
    closest: () => ({querySelector: () => ({textContent: 'A real result'})}), focus() {this.focused = true;}};
  const close = {addEventListener: (type, fn) => {listeners.button = fn;}};
  const dialog = {open: false, showModal: supported ? function () {this.open = true;} : undefined,
    close() {this.open = false; listeners.close();},
    querySelector: s => ({img: image, figcaption: caption, '.guide-image-close': close}[s]),
    getBoundingClientRect: () => ({left: 10, right: 100, top: 10, bottom: 100}),
    addEventListener: (type, fn) => {listeners[type] = fn;}};
  vm.runInNewContext(fs.readFileSync('static/summarizer/js/guide-images.js', 'utf8'), {
    document: {getElementById: () => dialog, addEventListener: (type, fn) => {events[type] = fn;}}
  });
  return {dialog, image, caption, link, listeners,
    open(extra = {}) {const e = {button: 0, target: {closest: () => link}, preventDefault() {this.prevented = true;}, ...extra}; events.click?.(e); return e;}};
}
test('guide screenshots open in place with caption and return focus when dismissed', () => {
  const h = harness(); assert.equal(h.open().prevented, true);
  assert.equal(h.dialog.open, true); assert.equal(h.image.src, '/capture.jpg');
  assert.equal(h.caption.textContent, 'A real result');
  h.listeners.button(); assert.equal(h.dialog.open, false); assert.equal(h.link.focused, true);
});
test('Escape, native cancellation and outside clicks close; content and padding clicks stay open', () => {
  const h = harness(), e = {key: 'Escape', preventDefault() {}, stopPropagation() {}};
  h.open(); h.listeners.keydown(e); assert.equal(h.dialog.open, false);
  h.open(); h.listeners.cancel(e); assert.equal(h.dialog.open, false);
  h.open(); h.listeners.click({target: {}, clientX: 0, clientY: 0}); assert.equal(h.dialog.open, true);
  h.listeners.click({target: h.dialog, clientX: 50, clientY: 50}); assert.equal(h.dialog.open, true);
  h.listeners.click({target: h.dialog, clientX: 0, clientY: 50}); assert.equal(h.dialog.open, false);
});
test('modified clicks and unsupported browsers retain normal image navigation', () => {
  assert.equal(harness().open({ctrlKey: true}).prevented, undefined);
  assert.equal(harness(false).open().prevented, undefined);
});
