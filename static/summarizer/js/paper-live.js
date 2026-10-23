/* Live progress never evaluates scripts or navigates away from a normal synthesis. */
(function () {
  'use strict';
  var box = document.getElementById('synthesis-live');
  if (!box) return;
  var status = box.querySelector('#live-status'), progress = box.querySelector('progress'), retry = box.querySelector('#live-retry');
  var pendingCommand = box.dataset.command;
  var socket, poll, reconnect, complete = false, installing = false, manuallySelected = false, shownFirst = false, awaitingRetry = false;
  var fields = ['summary', 'notes', 'lay_summary', 'blog'], seen = {};
  var fr = box.dataset.lang === 'fr';
  var scopeText = fr ? {
    checking: 'Vérification de la licence et de l’accès au texte intégral…',
    full_text_allowed: 'La licence permet l’analyse du texte intégral. Nous vérifions sa disponibilité ; un éventuel recours à l’abstract sera signalé ici.',
    full_text: 'Synthèse fondée sur le texte intégral de l’article.',
    abstract_license: 'Abstract et métadonnées uniquement : le texte intégral n’est pas analysé en raison de la licence de cet article.',
    abstract_unavailable: 'Abstract et métadonnées uniquement : le texte intégral n’a pas pu être récupéré. Vérifiez les détails dans l’article original.'
  } : {
    checking: 'Checking the license and access to the full text…',
    full_text_allowed: 'The license permits full-text analysis. We are checking availability and will indicate any fallback to the abstract here.',
    full_text: 'This synthesis is based on the paper’s full text.',
    abstract_license: 'Abstract and metadata only: this paper’s license does not permit full-text analysis here.',
    abstract_unavailable: 'Abstract and metadata only: the full text could not be retrieved. Check details in the original paper.'
  };
  // Only server-sanitized prose tags are admitted; no attributes, links, images or scripts.
  function safeProse(html) {
    var parsed = new DOMParser().parseFromString(String(html), 'text/html');
    var allowed = ['H2', 'H3', 'P', 'UL', 'OL', 'LI', 'STRONG', 'EM', 'B', 'I', 'BR'];
    parsed.querySelectorAll('script,style,iframe,object').forEach(function (el) {el.remove();});
    Array.from(parsed.body.querySelectorAll('*')).reverse().forEach(function (el) {
      if (allowed.indexOf(el.tagName) < 0) el.replaceWith.apply(el, Array.from(el.childNodes));
      else Array.from(el.attributes).forEach(function (attr) {el.removeAttribute(attr.name);});
    });
    return document.importNode(parsed.body, true);
  }
  function scope(value) {
    var banner = box.querySelector('.source-scope');
    if (banner && scopeText[value]) {banner.dataset.scope = value; banner.querySelector('[data-scope-text]').textContent = scopeText[value];}
  }
  function select(index) {
    switchTab(index);
    box.querySelectorAll('[data-live-tab]').forEach(function (tab) {tab.setAttribute('aria-selected', String(Number(tab.dataset.liveTab) === index));});
  }
  function section(field, content, html) {
    var index = fields.indexOf(field), panel = box.querySelector('#live-panel-' + index);
    if (!panel || content == null || seen[field] === JSON.stringify([content, html])) return;
    seen[field] = JSON.stringify([content, html]); panel.replaceChildren();
    if (field === 'lay_summary' && html) {var formatted = safeProse(html); panel.style.whiteSpace = 'normal'; panel.classList.add('paper-lay-summary'); panel.append.apply(panel, Array.from(formatted.childNodes));}
    else if (field === 'blog') {var prose = safeProse(content); panel.append.apply(panel, Array.from(prose.childNodes));}
    else if (field === 'notes') {
      var ul = document.createElement('ul'); ul.className = 'paper-key-points';
      (Array.isArray(content) ? content : [content]).forEach(function (point) {var li = document.createElement('li'); var prose = safeProse(point); li.append.apply(li, Array.from(prose.childNodes)); ul.appendChild(li);});
      panel.appendChild(ul);
    } else {panel.textContent = String(content); panel.style.whiteSpace = 'pre-line';}
    box.querySelector('#live-tab-' + index).classList.add('is-ready');
    if (!manuallySelected && !shownFirst) select(index);
    shownFirst = true;
    if (window.SPChat) SPChat.typeset(panel);
  }
  function failed(message) {status.textContent = message || box.dataset.retry; retry.hidden = false;}
  function stop() {clearInterval(poll); clearTimeout(reconnect); if (socket) {socket.onclose = null; socket.close();}}
  async function install() {
    if (installing || complete) return;
    if (box.dataset.tree === '0') {complete = true; stop(); location.assign(box.dataset.treeUrl); return;}
    installing = true;
    try {
      var response = await fetch(box.dataset.fragment, {credentials: 'same-origin', cache: 'no-store'});
      if (!response.ok) throw new Error('fragment');
      var parsed = new DOMParser().parseFromString(await response.text(), 'text/html');
      var detail = parsed.querySelector('#paper-detail'); if (!detail) throw new Error('detail');
      // Initializers are static code loaded with this page, never code taken from an HTML response.
      detail.querySelectorAll('script').forEach(function (script) {script.remove();});
      var active = box.querySelector('.message-tabs li.active');
      var selected = active ? Array.from(active.parentNode.children).indexOf(active) : 0;
      var wrapper = document.createElement('div'); wrapper.className = 'main-content2'; wrapper.appendChild(document.importNode(detail, true));
      var scrollY = window.scrollY;
      document.getElementById('paper-page').replaceChildren(wrapper);
      complete = true; stop();
      history.replaceState(history.state, '', box.dataset.canonical + (box.dataset.tree === '1' ? '#closest' : location.hash));
      SPPaper.init(); switchTab(selected);
      // Preserve the result-view funnel event that previously came from the final navigation.
      try {if (window.umami) Promise.resolve(window.umami.track('paper_result_view')).catch(function () {});} catch (_) {}
      if (window.SPChat) SPChat.typeset(wrapper);
      window.scrollTo(0, scrollY);
      if (box.dataset.tree === '1') {var closest = document.getElementById('closest'); if (closest) closest.scrollIntoView();}
    } catch (_) {awaitingRetry = true; failed(fr ? 'Synthèse prête. Réessayez de charger les outils ; vos résultats restent affichés.' : 'Synthesis ready. Retry loading the tools; your results remain visible.');}
    finally {installing = false;}
  }
  function update(state) {
    if (state.metadata && state.metadata.title) {var title = box.querySelector("#live-title"); title.textContent = state.metadata.title; title.hidden = false;}
    fields.forEach(function (field) {if (state[field] != null) section(field, state[field], field === 'lay_summary' ? state.lay_summary_html : null);});
    if (state.source_scope) scope(state.source_scope);
    if (state.progress) {
      var p = state.progress;
      progress.value = Math.max(0, Math.min(100, Number(p.progress) || 0));
      status.textContent = (p.loading_message || '') + (p.progress ? ' · ' + p.progress + '%' : '');
      if (p.progress === 0) failed(p.loading_message);
      if (p.progress === 100 && !awaitingRetry) install();
    }
  }
  async function snapshot() {
    if (complete) return;
    try {var r = await fetch(box.dataset.state, {cache: 'no-store'}); if (r.ok) update(await r.json());} catch (_) { /* Socket remains the primary transport. */ }
  }
  function connect() {
    if (complete) return;
    socket = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/arxivid/' + box.dataset.paper + '/' + box.dataset.lang + '/');
    socket.onopen = function () {if (pendingCommand) {socket.send(JSON.stringify({command: pendingCommand})); pendingCommand = ''; } else snapshot();};
    socket.onmessage = function (event) {
      var data; try {data = JSON.parse(event.data);} catch (_) {return;}
      if (data.type === 'synthesis_snapshot') {update(data.state); return;}
      if (data.type === 'progress_arxiv_update') {update({metadata: data.message}); return;}
      if (data.type === 'progress_text_update') {update({progress: data.message}); return;}
      if (data.type === 'progress_source_update') {scope(data.message); return;}
      var field = {progress_sum_update: 'summary', progress_notes_update: 'notes', progress_laysum_update: 'lay_summary', progress_blog_update: 'blog'}[data.type];
      if (field) section(field, data.message, data.html);
    };
    socket.onclose = function () {if (!complete) {snapshot(); reconnect = setTimeout(function () {connect();}, 3000);}};
  }
  box.querySelectorAll('[data-live-tab]').forEach(function (tab) {tab.addEventListener('click', function () {manuallySelected = true; select(Number(tab.dataset.liveTab));});});
  retry.addEventListener('click', function () {retry.hidden = true; awaitingRetry = false; snapshot(); if (progress.value === 100) install();});
  // Do not replay an HTTP-authorized start on reconnect.
  connect(); poll = setInterval(snapshot, 5000);
  window.addEventListener('pagehide', stop, {once: true});
})();
