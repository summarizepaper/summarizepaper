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

  // Wording comes from the page (#chat-evidence-labels, translated); these are the fallbacks.
  var EVIDENCE_LABELS = {
    title: 'Where each claim comes from', count: '{found} of {total} claims found word for word in the paper',
    checking: 'Checking each claim against the paper…', unavailable: 'This answer could not be checked against the paper.',
    section: 'Section “{name}”', abstract: 'Abstract', paragraph: 'paragraph {n}', open: 'Open this passage on arXiv',
    ref: 'Claim {n}: where it comes from',
    verified: 'Passage found in the paper', not_found: 'Passage not found in the paper: check this claim',
    visual: 'Read on the figure image', general: 'General knowledge, not from the paper',
    unsupported: 'Not supported by the paper', numbers: 'Numbers the paper never states: {numbers}',
    gapped: 'The extracted text interrupts this passage (page break, caption or link)'
  };

  function evidenceLabels() {
    var element = document.getElementById('chat-evidence-labels'), labels = {};
    Object.keys(EVIDENCE_LABELS).forEach(function (key) {
      labels[key] = (element && element.dataset[key]) || EVIDENCE_LABELS[key];
    });
    return labels;
  }

  function evidenceLink(value) {
    try {var u = new URL(value); return u.protocol === 'https:' && u.hostname === 'arxiv.org';} catch (_) {return false;}
  }

  var evidenceCount = 0;

  // Text nodes of the answer a reader sees as prose: not code, not equations (raw or typeset).
  function proseNodes(node, list) {
    Array.prototype.forEach.call(node.childNodes || [], function (child) {
      if (child.nodeType === 3) list.push(child);
      else if (child.nodeType === 1 && !/^(CODE|PRE|MJX-)/i.test(child.tagName) &&
               !/(^| )(chat-math|chat-claim-ref)( |$)/.test(child.className || '')) proseNodes(child, list);
    });
    return list;
  }

  /* [n] after the words that end the sentence making claim n; nothing when they are not found in one text node. */
  function markClaim(body, anchor, n, target, box, label) {
    var words = anchor.split(/\s+/).filter(Boolean).map(function (w) {return w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');});
    if (words.length < 2) return;
    var pattern = new RegExp(words.join('\\s+'));
    var nodes = proseNodes(body, []);
    for (var i = 0; i < nodes.length; i++) {
      var match = pattern.exec(nodes[i].nodeValue);
      if (!match) continue;
      var after = nodes[i].splitText(match.index + match[0].length);
      var ref = document.createElement('sup'), link = document.createElement('a');
      ref.className = 'chat-claim-ref';
      link.href = '#' + target; link.textContent = '[' + n + ']'; link.title = label.replace('{n}', n);
      link.addEventListener('click', function () {box.open = true;});
      ref.appendChild(link);
      after.parentNode.insertBefore(ref, after);
      return;
    }
  }

  /* The claims of an answer, each with the passage of the paper behind it (see summarizer/evidence.py).
     data: {status: 'checking'|'done'|'unavailable', claims: [...]}. Claims are model output: text only. */
  function evidence(element, data) {
    var old = element.querySelector('.chat-evidence');
    if (old) old.remove();
    var body = element.querySelector('.chat-answer');
    if (body && body.querySelectorAll) Array.prototype.forEach.call(body.querySelectorAll('.chat-claim-ref'), function (ref) {ref.remove();});
    data = data || {};
    var labels = evidenceLabels(), box;
    function node(tag, className, text) {
      var e = document.createElement(tag);
      if (className) e.className = className;
      if (text) e.textContent = text;
      return e;
    }
    if (data.status === 'checking' || data.status === 'unavailable') {
      box = node('p', 'chat-evidence chat-evidence-' + data.status, labels[data.status]);
      element.appendChild(box);
      return box;
    }
    var claims = Array.isArray(data.claims) ? data.claims.filter(function (c) {return c && typeof c.claim === 'string';}) : [];
    if (!claims.length) return null;
    var found = claims.filter(function (c) {return c.status === 'verified';}).length;
    var papers = {};
    claims.forEach(function (c) {if (c.paper) papers[c.paper] = true;});
    var several = Object.keys(papers).length > 1;
    box = node('details', 'chat-evidence');
    box.open = true;
    var summary = node('summary', '', labels.title + ' · ' + labels.count.replace('{found}', found).replace('{total}', claims.length));
    var list = node('ol'), id = 'claims-' + (++evidenceCount);
    claims.forEach(function (c, index) {
      var status = labels.hasOwnProperty(c.status) ? c.status : 'unsupported';
      var item = node('li', 'chat-claim chat-claim-' + status);
      item.id = id + '-' + (index + 1);
      item.appendChild(node('p', 'chat-claim-text', c.claim));
      var where = [labels[status]];
      if (c.section === 'Abstract') where.push(labels.abstract);
      else if (c.section) where.push(labels.section.replace('{name}', c.section));
      if (c.paragraph) where.push(labels.paragraph.replace('{n}', c.paragraph));
      if (c.figure) where.push(c.figure);
      if (several && c.paper) where.push(c.paper);
      item.appendChild(node('p', 'chat-claim-where', where.join(' · ')));
      if (c.quote) item.appendChild(node('blockquote', '', c.quote));
      if (c.gapped) item.appendChild(node('p', 'chat-claim-note', labels.gapped));
      if (Array.isArray(c.numbers_absent) && c.numbers_absent.length) {
        item.appendChild(node('p', 'chat-claim-warning', labels.numbers.replace('{numbers}', c.numbers_absent.join(', '))));
      }
      if (c.link && evidenceLink(c.link)) {
        var link = node('a', '', labels.open);
        link.href = c.link; link.target = '_blank'; link.rel = 'nofollow noopener noreferrer';
        item.appendChild(link);
      }
      list.appendChild(item);
    });
    box.append(summary, list);
    element.appendChild(box);
    if (body) claims.forEach(function (c, index) {
      if (typeof c.anchor === 'string' && c.anchor) markClaim(body, c.anchor, index + 1, id + '-' + (index + 1), box, labels.ref);
    });
    typeset(box);
    return box;
  }

  window.SPChat = {
    render: render,
    append: function (element, delta, label) {render(element, (raw.get(element) || '') + delta, label);},
    typeset: typeset,
    evidence: evidence
  };
  function init(root) {
    (root || document).querySelectorAll('[data-chat-markdown]').forEach(function (element) {
      // History is ignored by MathJax until its original text has been parsed as Markdown.
      if (element.dataset.markdownMounted) return;
      element.dataset.markdownMounted = '1';
      var body = element.querySelector('.chat-answer');
      render(element, body.textContent, element.dataset.chatLabel);
      if (element.dataset.evidence) {
        try {evidence(element, JSON.parse(element.dataset.evidence));} catch (_) {}
      }
      typeset(element);
    });
  }
  window.SPChat.init = init;
  document.addEventListener('DOMContentLoaded', function () {init(document);});
})();
