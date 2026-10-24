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
    var input = form.querySelector('input[type="text"]'), messages = root.querySelector('.chatbot-messages');
    var button = root.querySelector('#send-btn'), spinner = root.querySelector('#loading-spinner');
    var socket = null, currentResponse = null, pendingFigures = null, checking = {}, busy = false;
    var nudge = root.querySelector('#account-nudge');
    root.querySelectorAll('.chat-suggestions button').forEach(function (suggestion) {
      suggestion.addEventListener('click', function () {
        if (busy) return;
        input.value = suggestion.dataset.question;
        form.requestSubmit ? form.requestSubmit() : form.dispatchEvent(new Event('submit', {cancelable: true}));
      });
    });
    function scroll() {messages.scrollTop = messages.scrollHeight;}
    function finish() {busy = false; button.disabled = false; spinner.style.display = 'none';}
    function error() {if (busy) {var msg = document.createElement('p'); msg.textContent = root.dataset.error; messages.appendChild(msg);} finish(); socket = null; currentResponse = null; pendingFigures = null; checking = {};}
    function receive(event) {
      var data;
      try {data = JSON.parse(event.data);} catch (_) {return;}
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
      currentResponse = null; pendingFigures = null; finish(); scroll();
      if (nudge) nudge.hidden = false;  // after a first answer: a free account keeps the conversation
    }
    form.addEventListener('submit', function (event) {
      event.preventDefault(); var question = input.value.trim(); if (!question || busy) return;
      busy = true; button.disabled = true; spinner.style.display = 'block';
      var msg = document.createElement('div'); msg.className = 'chatbot-message'; msg.textContent = root.dataset.user + ' ' + question; messages.appendChild(msg); input.value = ''; scroll();
      function send() {socket.send(JSON.stringify({message: question, count: root.dataset.public === '1' ? 0 : 1}));}
      if (socket && socket.readyState === WebSocket.OPEN) {send(); return;}
      socket = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/arxivid/' + root.dataset.paper + '/' + root.dataset.lang + '/');
      socket.addEventListener('message', receive); socket.addEventListener('open', send, {once: true});
      socket.addEventListener('close', error); socket.addEventListener('error', error);
    });
    scroll();
  }
  window.SPPaper = {init: init};
  init();
  document.addEventListener('DOMContentLoaded', init);
})();
