/* Model output is untrusted. HTML is disabled; only our escaped math rule is added. */
(function () {
  'use strict';
  var md = window.markdownit ? window.markdownit({html: false, breaks: true, linkify: false}) : null;
  var raw = new WeakMap();
  var mathQueue = Promise.resolve();

  function mathAt(source, start) {
    var open = source.slice(start, start + 2);
    var close;
    if (open === '\\(') close = '\\)';
    else if (open === '\\[') close = '\\]';
    else if (open === '$$') close = '$$';
    else if (source[start] === '$' && !/\s/.test(source[start + 1] || ' ')) {open = '$'; close = '$';}
    else return null;
    var end = start + open.length;
    while ((end = source.indexOf(close, end)) !== -1) {
      var slashes = 0;
      for (var i = end - 1; i >= 0 && source[i] === '\\'; i--) slashes++;
      if (slashes % 2 === 0 && (close !== '$' || !/\s/.test(source[end - 1]))) {
        return {text: source.slice(start, end + close.length), end: end + close.length};
      }
      end += close.length;
    }
    return null;
  }

  if (md) {
    // No remote image requests, including images hidden in model-generated Markdown.
    md.disable('image');
    var validateLink = md.validateLink.bind(md);
    md.validateLink = function (url) {
      return validateLink(url) && !/[\u0000-\u0020\u007f]/.test(url) &&
        (/^https?:\/\//i.test(url) || !/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(url));
    };
    md.renderer.rules.link_open = function (tokens, index, options, env, renderer) {
      tokens[index].attrSet('rel', 'nofollow noopener noreferrer');
      tokens[index].attrSet('target', '_blank');
      return renderer.renderToken(tokens, index, options);
    };
    // Preserve _, *, backslashes and line breaks inside LaTeX before Markdown sees them.
    md.inline.ruler.before('escape', 'chat_math', function (state, silent) {
      var found = mathAt(state.src, state.pos);
      if (!found) return false;
      if (!silent) state.push('chat_math', '', 0).content = found.text;
      state.pos = found.end;
      return true;
    });
    md.renderer.rules.chat_math = function (tokens, index) {
      return '<span class="chat-math">' + md.utils.escapeHtml(tokens[index].content) + '</span>';
    };
    md.block.ruler.before('fence', 'chat_math_block', function (state, start, end, silent) {
      if (state.sCount[start] - state.blkIndent >= 4) return false;
      var begin = state.bMarks[start] + state.tShift[start];
      var opening = state.src.slice(begin, begin + 2);
      if (opening !== '$$' && opening !== '\\[') return false;
      var found = mathAt(state.src.slice(0, state.eMarks[end - 1]), begin);
      if (!found) return false;
      var last = start + (found.text.match(/\n/g) || []).length;
      if (state.src.slice(found.end, state.eMarks[last]).trim()) return false;
      if (silent) return true;
      var token = state.push('chat_math_block', '', 0);
      token.content = found.text;
      token.block = true;
      token.map = [start, last + 1];
      state.line = last + 1;
      return true;
    }, {alt: ['paragraph', 'reference', 'blockquote', 'list']});
    md.renderer.rules.chat_math_block = function (tokens, index) {
      return '<div class="chat-math">' + md.utils.escapeHtml(tokens[index].content) + '</div>\n';
    };
  }

  function render(element, text, label) {
    text = String(text == null ? '' : text);
    raw.set(element, text);
    var name = document.createElement('span');
    name.className = 'chat-label';
    name.textContent = label;
    var body = document.createElement('div');
    body.className = 'chat-answer';
    if (md) body.innerHTML = md.render(text);
    else body.textContent = text; // A missing library must never break chat or insert raw HTML.
    element.replaceChildren(name, body);
  }

  function typeset(element) {
    var engine = window.MathJax;
    if (engine && typeof engine.typesetPromise === 'function') {
      mathQueue = mathQueue.then(function () {return engine.typesetPromise([element]);}).catch(function () {});
    }
  }

  window.SPChat = {
    render: render,
    append: function (element, delta, label) {render(element, (raw.get(element) || '') + delta, label);},
    typeset: typeset
  };
  function init(root) {
    (root || document).querySelectorAll('[data-chat-markdown]').forEach(function (element) {
      // History is ignored by MathJax until its original text has been parsed as Markdown.
      if (element.dataset.markdownMounted) return;
      element.dataset.markdownMounted = '1';
      var body = element.querySelector('.chat-answer');
      render(element, body.textContent, element.dataset.chatLabel);
      typeset(element);
    });
  }
  window.SPChat.init = init;
  document.addEventListener('DOMContentLoaded', function () {init(document);});
})();
