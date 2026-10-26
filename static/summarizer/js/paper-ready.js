/* Explicit initializers shared by a regular paper page and a completed live synthesis. */
(function () {
  'use strict';
  window.switchTab = function (index) {
    var root = document.getElementById('results');
    if (!root) return;
    root.querySelectorAll('.message-tabs li').forEach(function (tab, i) {tab.classList.toggle('active', i === index);});
    root.querySelectorAll('.message-tab').forEach(function (panel, i) {panel.hidden = i !== index; panel.style.display = i === index ? 'block' : 'none';});
  };
  function init() {
    var root = document.getElementById('paper-detail');
    if (!root || root.dataset.mounted) return;
    root.dataset.mounted = '1';
    if (window.SPPaperTools) SPPaperTools.init();
    if (window.SPChatFigures && SPChatFigures.init) SPChatFigures.init(root);
    if (window.SPChat && SPChat.init) SPChat.init(root);
    var download = root.querySelector('#download-form');
    if (download) download.addEventListener('submit', function () {
      root.querySelector('#localDateInput').value = new Date().toLocaleString();
      var spinner = root.querySelector('#spinner'); spinner.style.display = 'inline-block';
      setTimeout(function () {spinner.style.display = 'none';}, 1500);
    });
    root.querySelectorAll('.vote-actions form').forEach(function (form) {
      form.addEventListener('submit', function (event) {
        event.preventDefault();
        var data = new FormData(form); data.append('direction', (event.submitter || form.querySelector('button')).value);
        var buttons = root.querySelectorAll('.vote-btn'); buttons.forEach(function (b) {b.disabled = true;});
        fetch(form.action, {method: 'POST', body: data, headers: {'X-Requested-With': 'fetch'}})
          .then(function (r) {if (!r.ok) throw r; return r.json();})
          .then(function (res) {root.querySelector('#vote-total').textContent = res.total; var msg = root.querySelector('#vote-message'); msg.textContent = res.message; msg.hidden = false; msg.classList.toggle('ok', !res.already);})
          .catch(function () {var msg = root.querySelector('#vote-message'); msg.textContent = root.dataset.error; msg.hidden = false;})
          .finally(function () {buttons.forEach(function (b) {b.disabled = false;});});
      });
    });
    ['','2'].forEach(function (suffix) {
      var link = root.querySelector('#popover-link' + suffix), popover = root.querySelector('#popover' + suffix);
      if (!link || !popover) return;
      link.addEventListener('mouseover', function (event) {popover.style.display = 'block'; popover.style.left = event.clientX - 40 + 'px'; popover.style.top = event.clientY + 20 + 'px';});
      link.addEventListener('mouseout', function () {popover.style.display = 'none';});
    });
    var toggle = root.querySelector('#chatbot-toggle');
    if (toggle) toggle.addEventListener('click', function (event) {event.preventDefault(); var help = root.querySelector('.chatbot-instructions'); help.style.display = help.style.display === 'none' ? 'block' : 'none';});
    var form = root.querySelector('.chatbot-form');
    if (!form) return;
    var input = form.querySelector('textarea, input[type="text"]'), messages = root.querySelector('.chatbot-messages');
    var button = root.querySelector('#send-btn'), spinner = root.querySelector('#loading-spinner'), stop = root.querySelector('#stop-btn');
    var socket = null, currentResponse = null, pendingFigures = null, checking = {}, busy = false, lastQuestion = '';
    var nudge = root.querySelector('#account-nudge'), tools = window.SPChatTools, labels = root.dataset;
    var share = root.querySelector('#share-conversation'), reading = root.querySelector('#chat-reading');
    var refreshQuota = tools ? tools.quota(root.querySelector('#chat-quota'), labels.quotaUrl) : function () {};
    var grow = tools && input.tagName === 'TEXTAREA' ? tools.questionBox(input, function () {ask(input.value);}) : function () {};
    if (tools) tools.share(root, function () {return labels.paper;});
    root.querySelectorAll('.chat-suggestions button').forEach(function (suggestion) {
      suggestion.addEventListener('click', function () {ask(suggestion.dataset.question);});
    });
    // what the assistant reads of the paper: its full text, part of it, or its abstract
    function showReading(data) {
      var paper = data && data.papers && data.papers[0];
      if (!reading || !paper) return;
      var partial = paper.kind === 'full' && paper.coverage < 100 && (paper.left_out || []).join() !== 'references';
      reading.textContent = paper.kind === 'full' ? (partial ? labels.readsPartial.replace('{n}', paper.coverage) : labels.readsFull) : labels.readsAbstract;
      reading.hidden = false;
    }
    if (reading && labels.sourcesUrl) fetch(labels.sourcesUrl + '?ids=' + encodeURIComponent(labels.paper))
      .then(function (r) {return r.ok ? r.json() : null;}).then(showReading).catch(function () {});
    function scroll() {messages.scrollTop = messages.scrollHeight;}
    function actions(response) {
      if (tools) tools.actions(response, {question: response.dataset.question || lastQuestion, ask: ask, labels: labels});
    }
    function finish(response) {
      busy = false; button.disabled = false; spinner.style.display = 'none';
      if (stop) stop.hidden = true;
      if (response) {actions(response); refreshQuota();}
      if (share) share.disabled = !messages.querySelector('.chatbot-response');
    }
    function stopped() {
      var note = document.createElement('p'); note.className = 'chat-stopped'; note.textContent = labels.stopped;
      (currentResponse || messages).appendChild(note);
      currentResponse = null; pendingFigures = null; finish(null); scroll();
    }
    function error() {if (busy) {var msg = document.createElement('p'); msg.textContent = root.dataset.error; messages.appendChild(msg);} finish(null); socket = null; currentResponse = null; pendingFigures = null; checking = {};}
    function receive(event) {
      var data;
      try {data = JSON.parse(event.data);} catch (_) {return;}
      if (data.type === 'chat_stopped') {stopped(); return;}
      if (data.type === 'chat_sources') {showReading(data); return;}
      if (data.type === 'chat_figures') {pendingFigures = data; return;}
      if (data.type === 'chat_evidence') {
        if (checking[data.answer_id]) SPChat.evidence(checking[data.answer_id], data);
        delete checking[data.answer_id]; return;
      }
      if (data.type && data.type !== 'chat_delta') return;
      if (data.type !== 'chat_delta' && typeof data.message !== 'string') return;
      if (!currentResponse) {currentResponse = document.createElement('div'); currentResponse.className = 'chatbot-response'; messages.appendChild(currentResponse);}
      if (data.type === 'chat_delta') {SPChat.append(currentResponse, data.delta, root.dataset.ai); scroll(); return;}
      SPChat.render(currentResponse, data.message, root.dataset.ai);
      SPChat.typeset(currentResponse);
      if (pendingFigures && window.SPChatFigures) SPChatFigures.render(currentResponse, pendingFigures.figures, pendingFigures.status);
      // the claims of the answer are then checked against the paper, while the reader goes on
      if (data.evidence === 'checking') {checking[data.answer_id] = currentResponse; SPChat.evidence(currentResponse, {status: 'checking'});}
      var done = currentResponse;
      currentResponse = null; pendingFigures = null; finish(done); scroll();
      if (nudge) nudge.hidden = false;  // after a first answer: a free account keeps the conversation
    }
    function ask(question) {
      question = String(question || '').trim(); if (!question || busy) return;
      busy = true; button.disabled = true; spinner.style.display = 'block'; lastQuestion = question;
      if (stop) stop.hidden = false;
      var msg = document.createElement('div'); msg.className = 'chatbot-message'; msg.textContent = root.dataset.user + ' ' + question; messages.appendChild(msg); input.value = ''; grow(); scroll();
      function send() {socket.send(JSON.stringify({message: question, count: root.dataset.public === '1' ? 0 : 1}));}
      if (socket && socket.readyState === WebSocket.OPEN) {send(); return;}
      socket = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/arxivid/' + root.dataset.paper + '/' + root.dataset.lang + '/');
      socket.addEventListener('message', receive); socket.addEventListener('open', send, {once: true});
      socket.addEventListener('close', error); socket.addEventListener('error', error);
    }
    form.addEventListener('submit', function (event) {event.preventDefault(); ask(input.value);});
    if (stop) stop.addEventListener('click', function () {
      if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({stop: true}));
    });
    // the stored conversation gets its actions once rendered (SPChat.init runs on DOMContentLoaded)
    function storedActions() {messages.querySelectorAll('.chatbot-response').forEach(actions);}
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () {setTimeout(storedActions);});
    else storedActions();
    scroll();
  }
  window.SPPaper = {init: init};
  init();
  document.addEventListener('DOMContentLoaded', init);
})();
