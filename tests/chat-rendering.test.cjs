const {test} = require('node:test');
const assert = require('node:assert/strict');
const {createHarness, element} = require('./chat-rendering-helper.cjs');

function html(text) {
  const {SPChat} = createHarness();
  const node = element();
  SPChat.render(node, text, 'AI -');
  return node.children[1].innerHTML;
}

test('assistant Markdown supports emphasis, headings, nested lists, quotes, code and tables', () => {
  const result = html('## Findings\n\n**Water clouds**, *probably*.\n\n1. Result\n   - Caveat\n\n> Preliminary\n\n`x_i`\n\n```python\nx = 1\n```\n\n| A | B |\n| --- | --- |\n| 1 | 2 |');
  for (const tag of ['h2', 'strong', 'em', 'ol', 'ul', 'blockquote', 'code', 'pre', 'table']) {
    assert.ok(result.includes(`<${tag}`), tag);
  }
  assert.ok(!result.includes('**Water clouds**'));
});

test('raw HTML and attributes from model output remain escaped', () => {
  for (const attack of ['<script>alert(1)</script>', '<img src=x onerror=alert(1)>',
    '<svg onload=alert(1)>', '<iframe srcdoc="<script>alert(1)</script>">',
    '<a href="javascript:alert(1)">Click</a>', '<math href="javascript:alert(1)">x</math>']) {
    const result = html(attack);
    assert.ok(!/<(?:script|img|svg|iframe|a|math)\b/i.test(result), result);
    assert.ok(result.includes('&lt;'), result);
  }
});

test('unsafe Markdown links and tracking images never become executable links or images', () => {
  for (const url of ['javascript:alert%281%29', 'jAvAsCrIpT:alert%281%29',
    'jav&#x61;script:alert%281%29', 'data:text/html;base64,PHNjcmlwdD4=',
    'file:///etc/passwd', 'vbscript:msgbox%281%29', '//tracker.example/track']) {
    const result = html(`[Click](${url})`);
    assert.ok(!result.includes('<a '), result);
  }
  assert.ok(!html('![Tracking pixel](https://tracker.example/pixel)').includes('<img'));
  const link = html('[Paper](https://arxiv.org/abs/2101.00001)');
  assert.ok(link.includes('href="https://arxiv.org/abs/2101.00001"'));
  assert.ok(link.includes('rel="nofollow noopener noreferrer"'));
  assert.ok(html('[Source](/en/arxiv-id/2101.00001v1/)').includes('<a '));
});

test('LaTeX survives Markdown unchanged, including underscores, stars, tags and multiline display', () => {
  for (const math of ['$x_i * y_j$', '\\(x_i + \\alpha\\)', '\\[x_i^2 + \\beta\\]',
    '$$\nx_i = \\frac{a_b}{c_d}\n\ny_j = 2\n$$']) {
    const result = html('Equation:\n\n' + math);
    assert.ok(result.includes(math), result);
    assert.ok(!result.includes('<em>'), result);
  }
  const result = html('$<img src=x onerror=alert(1)>$');
  assert.ok(result.includes('&lt;img'));
  assert.ok(!result.includes('<img'));
  assert.ok(html('`$x_i$`').includes('<code>$x_i$</code>'));
});

test('streaming buffers raw Markdown and final output replaces the buffer without duplicate labels', () => {
  const {SPChat} = createHarness();
  const node = element();
  SPChat.append(node, '**Clou', 'AI -');
  SPChat.append(node, 'ds**\n\n- Result', 'AI -');
  assert.ok(node.children[1].innerHTML.includes('<strong>Clouds</strong>'));
  assert.equal(node.children.length, 2);
  SPChat.render(node, '**Final answer**', 'AI -');
  assert.ok(node.children[1].innerHTML.includes('<strong>Final answer</strong>'));
  assert.ok(!node.children[1].innerHTML.includes('Clouds'));
});

test('history is rendered with the same safe parser; missing library falls back to text', () => {
  const node = element();
  node.dataset.chatLabel = 'AI -';
  const body = element(); body.className = 'chat-answer'; body.textContent = '**Previous answer**';
  node.children = [body];
  const {handlers} = createHarness({history: [node]});
  handlers.DOMContentLoaded();
  assert.ok(node.children[1].innerHTML.includes('<strong>Previous answer</strong>'));
  const {SPChat} = createHarness({library: false});
  SPChat.render(node, '<script>bad()</script>', 'AI -');
  assert.equal(node.children[1].textContent, '<script>bad()</script>');
  assert.equal(node.children[1].innerHTML, '');
});
