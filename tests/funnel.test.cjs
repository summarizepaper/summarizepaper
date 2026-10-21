const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/summarizer/js/funnel.js', 'utf8');
function run(pathname, search = '', umami) {
  const events = [], handlers = {};
  const document = {
    addEventListener: (type, handler) => {handlers[type] = handler;},
    querySelectorAll: () => [],
  };
  vm.runInNewContext(source, {document, URLSearchParams, window: {
    location: {pathname, search}, umami: umami || {track: (name, data) => events.push({name, data})},
  }});
  return {events, handlers};
}
test('return from Stripe never counts as a confirmed activation', () => {
  const {events} = run('/en/premium/', '?success=1');
  assert.deepEqual(events.map(e => e.name), ['premium_view', 'checkout_return']);
});
test('checkout sends only the selected period', () => {
  const {events, handlers} = run('/fr/premium/');
  handlers.submit({target: {dataset: {funnel: 'checkout_start'}, email: 'private@example.org'}, submitter: {value: 'yearly'}});
  assert.equal(JSON.stringify(events.at(-1)), '{"name":"checkout_start","data":{"period":"yearly"}}');
});
test('tracking failures never break forms or links', () => {
  const {handlers} = run('/en/', '', {track() {throw new Error('blocked');}});
  assert.doesNotThrow(() => handlers.submit({target: {dataset: {funnel: 'signup_submit'}}}));
});
test('Umami payload strips email, query strings and signed tokens', () => {
  const base = fs.readFileSync('templates/base.html', 'utf8');
  const script = base.match(/function umamiBeforeSend[\s\S]*?\n  <\/script>/)[0].replace(/\n  <\/script>$/, '');
  const context = {URLSearchParams, navigator: {userAgent: 'Firefox', webdriver: false}};
  vm.createContext(context); vm.runInContext(script, context);
  const payload = context.umamiBeforeSend('event', {url: '/en/login/link/secret/?next=/en/premium/', referrer: 'https://example.org/?email=private'});
  assert.equal(payload.url, '/en/login/link/redacted/');
  assert.equal(payload.referrer, 'https://example.org/');
  assert.equal(context.umamiBeforeSend('event', {url: '/en/?utm_source=letter&q=private'}).url, '/en/?utm_source=letter');
  assert.equal(context.umamiBeforeSend('event', {url: '/en/reset/user/token/'}).url, '/en/reset/redacted/');
  context.navigator.webdriver = true;
  assert.equal(context.umamiBeforeSend('event', {}), false);
});
