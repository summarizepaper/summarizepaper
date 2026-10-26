// What the two chat pages share (the chat page, chat-page.js, and a paper page, paper-ready.js): the actions
// under an answer, the question box, the paper tags of an answer as links, the questions left today and the
// sharing of a conversation. Wording comes from the page root's data attributes; model output stays text.
(function () {
  'use strict';

  function node(tag, className, text) {
    var element = document.createElement(tag);
    if (className) element.className = className;
    if (text) element.textContent = text;
    return element;
  }

  function post(root, url, data) {
    var csrf = root.querySelector('[name=csrfmiddlewaretoken]');
    return fetch(url, {method: 'POST', body: new URLSearchParams(data || {}), headers: {'X-CSRFToken': csrf ? csrf.value : ''}})
      .then(function (response) {
        if (!response.ok) throw new Error(response.status);
        return response.json();
      });
  }

  function copied(button, text, labels) {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(text).then(function () {
      button.textContent = labels.copied;
      setTimeout(function () {button.textContent = labels.copy;}, 1500);
    });
  }

  // the claims of an answer and their passages, as Markdown after the answer (copies and exports)
  function sourcesMarkdown(claims, labels) {
    if (!claims || !claims.length) return '';
    var notes = {general: labels.exportGeneral, not_found: labels.exportNotFound, unsupported: labels.exportUnsupported};
    return '\n\n**' + labels.sources + '**\n\n' + claims.map(function (c, i) {
      var where = [c.paper, c.section, c.figure].filter(Boolean).join(', ');
      return (i + 1) + '. ' + c.claim + (notes[c.status] ? ' (' + notes[c.status] + ')' : '') +
        (where ? ' — ' + where : '') + (c.quote ? '\n   > ' + c.quote : '');
    }).join('\n');
  }

  // Copy (the answer with its sources, or text()) and Ask again, before the sources of the answer
  function actions(response, opts) {
    if (response.querySelector('.chat-answer-actions')) return;
    var labels = opts.labels;
    if (opts.question) response.dataset.question = opts.question;
    var bar = node('div', 'chat-answer-actions'), copy = node('button', '', labels.copy);
    copy.type = 'button';
    copy.addEventListener('click', function () {
      copied(copy, opts.text ? opts.text() : SPChat.raw(response) + sourcesMarkdown(SPChat.claims(response), labels), labels);
    });
    bar.appendChild(copy);
    if (response.dataset.question && opts.ask) {
      var again = node('button', '', labels.askAgain);
      again.type = 'button';
      again.addEventListener('click', function () {opts.ask(response.dataset.question);});
      bar.appendChild(again);
    }
    response.insertBefore(bar, response.querySelector('.chat-evidence'));
  }

  // a question box that grows with its text; Enter sends, Shift+Enter starts a new line
  function questionBox(textarea, send) {
    function grow() {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 240) + 'px';
    }
    textarea.addEventListener('input', grow);
    textarea.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {event.preventDefault(); send();}
    });
    return grow;
  }

  // [Smith et al. 2021] in an answer becomes a link to the paper's page, its title shown on hover
  function linkTags(element, papers) {
    var body = element.querySelector('.chat-answer');
    var known = (papers || []).filter(function (p) {return p.tag && p.url;});
    if (!body || !known.length) return;
    var pattern = new RegExp(known.map(function (p) {return p.tag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');}).join('|'), 'g');
    var walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT, {acceptNode: function (n) {
      return n.parentElement.closest('code, pre, .chat-math, mjx-container, a') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
    }});
    var texts = [];
    while (walker.nextNode()) texts.push(walker.currentNode);
    texts.forEach(function (text) {
      var value = text.nodeValue, last = 0, match, parts = [];
      pattern.lastIndex = 0;
      while ((match = pattern.exec(value))) {
        var paper = known.filter(function (p) {return p.tag === match[0];})[0];
        parts.push(document.createTextNode(value.slice(last, match.index)));
        var link = node('a', 'chat-tag', match[0]);
        link.href = paper.url;
        link.target = '_blank';
        link.title = paper.title;
        parts.push(link);
        last = match.index + match[0].length;
      }
      if (!parts.length) return;
      parts.push(document.createTextNode(value.slice(last)));
      parts.forEach(function (part) {text.parentNode.insertBefore(part, text);});
      text.remove();
    });
  }

  // "7 questions left today", asked again after each answer
  function quota(element, url) {
    if (!element) return function () {};
    function show(left) {
      element.hidden = left === null || left === undefined;
      if (!element.hidden) element.textContent = (left === 1 ? element.dataset.one : element.dataset.many).replace('{n}', left);
    }
    show(element.dataset.left === '' ? null : Number(element.dataset.left));
    return function () {
      fetch(url).then(function (r) {return r.json();}).then(function (data) {show(data.left);}).catch(function () {});
    };
  }

  // the Share button: a public, read-only page of the conversation as it is now (chat_page.share)
  function share(root, room) {
    var button = root.querySelector('#share-conversation'), box = root.querySelector('#share-box');
    if (!button || !box) return;
    var url = box.querySelector('#share-url'), stop = box.querySelector('#share-stop'), labels = root.dataset;
    button.addEventListener('click', function () {
      button.disabled = true;
      post(root, labels.shareUrl, {room: room()}).then(function (data) {
        url.value = data.url;
        stop.dataset.token = data.url.split('/shared/')[1].replace(/\/$/, '');
        box.hidden = false;
        url.select();
      }).catch(function () {alert(labels.shareFailed);}).then(function () {button.disabled = false;});
    });
    box.querySelector('#share-copy').addEventListener('click', function () {copied(this, url.value, labels);});
    stop.addEventListener('click', function () {
      post(root, labels.unshareUrl.replace('TOKEN', encodeURIComponent(stop.dataset.token))).then(function () {
        box.hidden = true;
        url.value = '';
      }).catch(function () {alert(labels.shareFailed);});
    });
  }

  window.SPChatTools = {node: node, post: post, actions: actions, sourcesMarkdown: sourcesMarkdown,
                        questionBox: questionBox, linkTags: linkTags, quota: quota, share: share};
})();
