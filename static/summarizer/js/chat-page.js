// The chat page: the papers to discuss (search, identifier, or a reference of a selected paper), what the
// assistant reads of each, the conversation (answers streamed, then the passage behind each claim), the
// comparison table and the exports. The conversation lives in the page address (?c=room&ids=...): opening it
// again shows it (summarizer/chat_page.py). Paper titles and model output are always set as text.
(function () {
  'use strict';
  var root = document.getElementById('chat-page');
  if (!root) return;
  var T = root.dataset;
  var MAX = Number(T.max) || 10;
  var WS = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host;
  var $ = function (selector) {return root.querySelector(selector);};
  var tools = window.SPChatTools;

  var search = $('#search'), searchResults = $('#search-results');
  var searchNote = $('#search-note'), spinner = $('#spinner');
  var addForm = $('#add-paper-form'), paperId = $('#paper-id'), addNote = $('#add-note');
  var list = $('#selected-papers'), nopap = $('#nopap'), count = $('#selected-count'), footer = $('#selection-footer');
  var reading = $('#chat-reading'), messages = $('.chatbot-messages'), suggestions = $('#chat-suggestions');
  var form = $('.chatbot-form'), input = form.querySelector('textarea');
  var sendButton = $('#send-btn'), stopButton = $('#stop-btn'), loading = $('#loading-spinner');
  var tableToggle = $('#table-toggle'), builder = $('#table-builder'), columnsInput = $('#table-columns');
  var tableCost = $('#table-cost'), tableAbstract = $('#table-abstract'), exportButton = $('#export-conversation');
  var shareButton = $('#share-conversation'), shareBox = $('#share-box');
  var defaultColumns = JSON.parse(document.getElementById('chat-table-columns').textContent);

  var selected = [];  // {id, title, author, published, label, state: 'reading'|'ready'|'failed', supply, item}
  var room = T.room || newRoom();
  var socket = null, queue = [], busy = false, lastQuestion = '', lastPapers = null;
  var currentResponse = null, pendingFigures = null, checking = {}, currentTable = null;

  // --- helpers --------------------------------------------------------------

  function node(tag, className, text) {
    var element = document.createElement(tag);
    if (className) element.className = className;
    if (text) element.textContent = text;
    return element;
  }
  function fmt(template, values) {
    return String(template).replace(/\{(\w+)\}/g, function (match, key) {return key in values ? values[key] : match;});
  }
  function note(element, text, kind) {
    element.textContent = text || '';
    element.hidden = !text;
    element.className = 'chat-note' + (kind ? ' ' + kind : '');
  }
  function newRoom() {
    var bytes = new Uint8Array(12);
    crypto.getRandomValues(bytes);
    return Array.prototype.map.call(bytes, function (b) {return ('0' + b.toString(16)).slice(-2);}).join('');
  }
  function canonical(id) {return String(id).trim().replace(/\//g, '--');}
  function external(id) {return /^(doi_|hal_|pmc_)/.test(id);}
  function version(id) {var m = /v(\d+)$/.exec(id); return !external(id) && m ? m[1] : '';}
  function paperUrl(id) {return '/' + T.lang + '/paper/' + encodeURIComponent(id) + '/';}
  function find(id) {return selected.filter(function (e) {return e.id === id;})[0];}
  function readyIds() {return selected.filter(function (e) {return e.state === 'ready';}).map(function (e) {return e.id;});}
  function papersKey(ids) {return ids.slice().sort().join(',');}
  function safeLink(value) {
    try {var u = new URL(value); return u.protocol === 'https:' && u.hostname === 'arxiv.org';} catch (_) {return false;}
  }
  function typeset(element) {
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([element]).catch(function () {});
  }
  function day(iso) {
    if (!iso) return '';
    var date = new Date(iso.length === 10 ? iso + 'T00:00:00Z' : iso);
    return isNaN(date) ? '' : date.toLocaleDateString(T.lang, {year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC'});
  }
  function download(name, text, type) {
    var link = node('a');
    link.href = URL.createObjectURL(new Blob([text], {type: type}));
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(function () {URL.revokeObjectURL(link.href);}, 1000);
  }
  function scroll() {messages.scrollTop = messages.scrollHeight;}

  // --- finding papers ---------------------------------------------------------

  // the relay of arXiv (and of the other sources) answers in arXiv's Atom form
  function fetchFeed(params) {
    return fetch(T.api + '?' + new URLSearchParams(params)).then(function (response) {
      if (!response.ok) throw new Error(response.status);
      return response.text();
    }).then(function (text) {
      var xml = new DOMParser().parseFromString(text, 'application/xml');
      return Array.prototype.map.call(xml.getElementsByTagName('entry'), function (entry) {
        function text(tag) {var e = entry.getElementsByTagName(tag)[0]; return e ? e.textContent.replace(/\s+/g, ' ').trim() : '';}
        var authors = entry.getElementsByTagName('author'), id = text('id');
        var first = authors.length ? (authors[0].getElementsByTagName('name')[0] || {}).textContent || '' : '';
        return {id: id.indexOf('/abs/') >= 0 ? canonical(id.split('/abs/')[1]) : '', title: text('title'),
                author: first.trim() + (authors.length > 1 ? ' et al.' : ''), published: text('published').slice(0, 10),
                label: text('label'), venue: text('venue'), summary: text('summary')};
      }).filter(function (paper) {return paper.id && paper.title;});
    });
  }

  var searchTimer = null, searchSeq = 0;
  function runSearch() {
    var query = search.value.trim(), seq = ++searchSeq;
    note(searchNote, '');
    if (query.length < 3) {
      searchResults.replaceChildren();
      note(searchNote, query ? T.atLeast : '', 'error');
      return;
    }
    spinner.hidden = false;
    // arXiv and the journals and preprint servers at once, as the search of the site
    fetchFeed({search_query: query, max_results: 10, src: 'all'}).then(function (papers) {
      if (seq !== searchSeq) return;
      searchResults.replaceChildren.apply(searchResults, papers.map(resultItem));
      if (!papers.length) note(searchNote, T.noneFound, 'error');
      typeset(searchResults);
    }).catch(function () {
      if (seq === searchSeq) note(searchNote, T.unreachable, 'error');
    }).then(function () {
      if (seq === searchSeq) spinner.hidden = true;
    });
  }
  search.addEventListener('input', function () {clearTimeout(searchTimer); searchTimer = setTimeout(runSearch, 400);});
  $('#search-help').querySelectorAll('[data-example]').forEach(function (example) {
    example.addEventListener('click', function () {search.value = example.dataset.example; runSearch();});
  });
  $('#paper-id-help').querySelectorAll('[data-id]').forEach(function (example) {
    example.addEventListener('click', function () {paperId.value = example.dataset.id; paperId.focus();});
  });

  // a search result: its title, authors and source, its abstract on demand, and Add
  function resultItem(paper) {
    var item = node('div', 'paper-item result-item');
    item.dataset.id = paper.id;
    var add = node('button', 'result-add', T.add);
    add.type = 'button';
    add.addEventListener('click', function () {
      var result = addPaper(paper);
      if (result === 'full') note(searchNote, T.fullSelection, 'error');
    });
    item.appendChild(add);
    item.appendChild(node('strong', '', paper.title));
    if (paper.author) item.appendChild(node('small', '', T.firstAuthor + ' ' + paper.author));
    var venue = /^(arXiv|OpenAlex|Crossref|Europe PMC)?$/.test(paper.venue || '') ? '' : paper.venue;  // a journal, not a source
    item.appendChild(node('small', '', [T.published + ' ' + day(paper.published), venue, paper.label].filter(Boolean).join(' · ')));
    if (paper.summary) {
      item.appendChild(node('p', 'result-abstract', paper.summary));
      var more = node('button', 'result-more', T.more);
      more.type = 'button';
      more.addEventListener('click', function () {more.textContent = item.classList.toggle('expanded') ? T.less : T.more;});
      item.appendChild(more);
    }
    mark(item);
    return item;
  }
  function mark(item) {
    var chosen = !!find(item.dataset.id), add = item.querySelector('.result-add');
    item.classList.toggle('active', chosen);
    add.disabled = chosen;
    add.textContent = chosen ? T.added : T.add;
  }
  function markResults() {searchResults.querySelectorAll('.result-item').forEach(mark);}

  // an arXiv id without version gives the latest version: arXiv answers with it
  function addById(value, quiet) {
    var id = String(value || '').trim();
    if (!id) return Promise.resolve('');
    return fetchFeed({id_list: id}).then(function (papers) {
      if (!papers.length) {
        if (!quiet) note(addNote, T.notFound, 'error');
        return 'unknown';
      }
      var result = addPaper(papers[0]);
      if (!quiet || result === 'full') {
        note(addNote, {added: T.addedOk, already: T.already, full: T.fullSelection}[result], result === 'added' ? 'ok' : 'error');
      }
      return result;
    }).catch(function () {
      if (!quiet) note(addNote, T.notFound, 'error');
      return 'unknown';
    });
  }
  addForm.addEventListener('submit', function (event) {
    event.preventDefault();
    addById(paperId.value).then(function (result) {if (result === 'added') paperId.value = '';});
  });

  // --- the selection ----------------------------------------------------------

  function addPaper(paper) {
    var id = canonical(paper.id);
    if (find(id)) return 'already';
    if (selected.length >= MAX) return 'full';
    var entry = {id: id, title: paper.title, author: paper.author || '', published: paper.published || '',
                 label: paper.label || '', state: 'reading', supply: null};
    entry.item = paperItem(entry);
    selected.push(entry);
    list.appendChild(entry.item);
    typeset(entry.item);
    ingest(entry);
    selectionChanged();
    return 'added';
  }

  var dragged = null;
  // entry goes just before target in the selection
  function move(entry, target) {
    if (!target || entry === target) return;
    selected.splice(selected.indexOf(entry), 1);
    selected.splice(selected.indexOf(target), 0, entry);
    selected.forEach(function (e) {list.appendChild(e.item);});
    selectionChanged();
  }

  function removePaper(id) {
    var entry = find(id);
    if (!entry) return;
    selected.splice(selected.indexOf(entry), 1);
    entry.item.remove();
    selectionChanged();
  }

  // the paper is read once (its text stored), on a socket of its own so that papers are read side by side
  function ingest(entry) {
    entry.state = 'reading';
    entry.supply = null;
    badges(entry);
    var socket = new WebSocket(WS + '/ws/emb/');
    socket.addEventListener('open', function () {socket.send(JSON.stringify({paper_id: entry.id}));});
    socket.addEventListener('message', function (event) {
      var data;
      try {data = JSON.parse(event.data);} catch (_) {return;}
      if (data.paper_id !== entry.id || (data.message !== 'completed' && data.message !== 'failed')) return;  // 'reading': still at it
      entry.state = data.message === 'completed' ? 'ready' : 'failed';
      socket.close();
      badges(entry);
      selectionChanged();
    });
    socket.addEventListener('close', function () {
      if (entry.state !== 'reading') return;
      entry.state = 'failed';
      badges(entry);
      selectionChanged();
    });
  }

  function paperItem(entry) {
    var item = node('li', 'paper-item');
    var title = node('strong'), link = node('a', '', entry.title);
    link.href = paperUrl(entry.id);
    link.target = '_blank';
    link.title = T.paperPage;
    title.appendChild(link);
    item.appendChild(title);
    if (entry.author) item.appendChild(node('small', '', entry.author));
    var publication = node('small', 'paper-publication');
    publication.hidden = true;
    item.appendChild(publication);
    var when = [entry.published ? (external(entry.id) ? T.published : T.arxivDate) + ' ' + day(entry.published) : '', entry.label].filter(Boolean).join(' · ');
    if (when) item.appendChild(node('small', '', when));
    item.appendChild(node('div', 'paper-badges'));
    var actions = node('div', 'paper-actions');
    if (!external(entry.id)) {
      var pdf = node('a', '', T.pdf);
      pdf.href = 'https://arxiv.org/pdf/' + entry.id.replace(/--/g, '/');
      pdf.target = '_blank';
      pdf.rel = 'noopener';
      pdf.setAttribute('data-pdf-panel', '');
      actions.appendChild(pdf);
    }
    var up = node('button', 'move-up', '↑ ' + T.moveUp);
    up.type = 'button';
    up.addEventListener('click', function () {move(entry, selected[selected.indexOf(entry) - 1]);});
    actions.appendChild(up);
    var refs = node('button', '', T.references);
    refs.type = 'button';
    refs.addEventListener('click', function () {references(entry);});
    actions.appendChild(refs);
    item.appendChild(actions);
    var remove = node('button', 'remove-symbol');
    remove.type = 'button';
    remove.id = entry.id;  // chat-figures.js reads the selected papers from these ids
    remove.title = T.remove;
    remove.setAttribute('aria-label', T.remove);
    remove.appendChild(node('i', 'fas fa-times'));
    remove.addEventListener('click', function () {removePaper(entry.id);});
    item.appendChild(remove);
    // the order of the papers is the order of the answers and of the table's rows
    item.draggable = true;
    item.title = T.drag;
    item.addEventListener('dragstart', function (event) {dragged = entry; item.classList.add('dragging'); event.dataTransfer.effectAllowed = 'move';});
    item.addEventListener('dragend', function () {dragged = null; item.classList.remove('dragging');});
    item.addEventListener('dragover', function (event) {if (dragged && dragged !== entry) {event.preventDefault(); item.classList.add('drop-before');}});
    item.addEventListener('dragleave', function () {item.classList.remove('drop-before');});
    item.addEventListener('drop', function (event) {event.preventDefault(); item.classList.remove('drop-before'); if (dragged) move(dragged, entry);});
    return item;
  }

  // what the assistant reads of the paper, its version, and its tag in answers about several papers
  function badges(entry) {
    var box = entry.item.querySelector('.paper-badges');
    box.replaceChildren();
    entry.item.classList.toggle('failed', entry.state === 'failed');
    function badge(className, text, title) {
      var b = node('span', 'paper-badge ' + className, text);
      if (title) b.title = title;
      box.appendChild(b);
      return b;
    }
    var supply = entry.supply;
    var publication = entry.item.querySelector('.paper-publication');
    publication.textContent = supply && supply.publication_year ? T.published + ' ' + supply.publication_year : '';
    publication.hidden = !publication.textContent;
    if (supply && supply.tag && readyIds().length > 1) badge('tag', supply.tag);
    if (entry.state === 'reading') badge('', T.reading).prepend(node('i', 'fas fa-spinner fa-spin'));
    else if (entry.state === 'failed') {
      badge('failed', T.failed);
    } else if (supply) {
      if (supply.kind === 'missing') badge('abstract', T.missing);
      else if (supply.kind === 'restricted') badge('abstract', T.restricted, T.restrictedTitle);
      else if (supply.kind === 'abstract') badge('abstract', T.abstract, T.abstractTitle);
      else if (supply.coverage >= 100 || (supply.left_out || []).join() === 'references') badge('full', T.full);
      else {
        var parts = (supply.left_out || []).map(function (k) {return T['left' + k.charAt(0).toUpperCase() + k.slice(1)];});
        badge('partial', fmt(T.partial, {n: supply.coverage}), fmt(T.partialTitle, {parts: parts.join(', ')}));
      }
    }
    var v = version(entry.id);
    if (v) badge('', fmt(T.version, {v: v}), T.versionTitle);
    if (entry.state === 'failed' && !entry.item.querySelector('.retry')) {
      var retry = node('button', 'retry', '↻');
      retry.type = 'button';
      retry.addEventListener('click', function () {retry.remove(); ingest(entry); selectionChanged();});
      entry.item.querySelector('.paper-actions').appendChild(retry);
    }
  }

  function references(entry) {
    var box = entry.item.querySelector('.paper-references');
    if (box) {box.remove(); return;}
    box = node('div', 'paper-references');
    box.appendChild(node('p', 'chat-note', T.referencesLoading));
    entry.item.appendChild(box);
    fetch(T.referencesUrl.replace('PAPER_ID', encodeURIComponent(entry.id))).then(function (r) {return r.json();}).then(function (data) {
      box.replaceChildren();
      if (data.status !== 'done') {box.appendChild(node('p', 'chat-note', T.referencesUnavailable)); return;}
      if (!data.references.length) {box.appendChild(node('p', 'chat-note', T.referencesNone)); return;}
      var filter = node('input');
      filter.type = 'search';
      filter.placeholder = T.referencesFilter;
      filter.setAttribute('aria-label', T.referencesFilter);
      box.appendChild(filter);
      var rows = data.references.map(function (ref) {
        var row = node('div', 'paper-reference'), add = node('button', '', T.add);
        add.type = 'button';
        add.addEventListener('click', function () {
          add.disabled = true;
          addById(ref.ref, true).then(function (result) {
            var done = result === 'added' || result === 'already';
            add.textContent = done ? T.added : T.add;
            add.disabled = done;
            if (result === 'unknown') add.textContent = '✕';
          });
        });
        row.append(add, node('span', '', ref.title + ' — ' + [ref.authors, ref.year].filter(Boolean).join(', ')));
        box.appendChild(row);
        return {row: row, text: (ref.title + ' ' + ref.authors + ' ' + ref.year).toLowerCase()};
      });
      filter.addEventListener('input', function () {
        var words = filter.value.toLowerCase();
        rows.forEach(function (r) {r.row.hidden = !!words && r.text.indexOf(words) < 0;});
      });
    }).catch(function () {box.replaceChildren(node('p', 'chat-note', T.referencesUnavailable));});
  }

  var sourcesTimer = null;
  function selectionChanged() {
    nopap.hidden = selected.length > 0;
    selected.forEach(function (e, i) {e.item.querySelector('.move-up').hidden = i === 0;});
    footer.hidden = selected.length === 0;
    count.textContent = selected.length ? selected.length + '/' + MAX : '';
    var ids = selected.map(function (e) {return e.id;}).join(',');
    footer.querySelectorAll('[data-export]').forEach(function (link) {
      link.href = T.exportUrl.replace('FMT', link.dataset.export) + '?ids=' + encodeURIComponent(ids);
    });
    markResults();
    updateUrl();
    composer();
    clearTimeout(sourcesTimer);
    sourcesTimer = setTimeout(refreshSources, 300);
  }
  $('#clear-selection').addEventListener('click', function () {
    selected.slice().forEach(function (e) {removePaper(e.id);});
  });

  // what the assistant will read of each paper, asked again whenever the selection changes
  var sourcesSeq = 0;
  function refreshSources() {
    var ids = readyIds();
    if (!ids.length || selected.some(function (e) {return e.state === 'reading';})) return;
    var seq = ++sourcesSeq;
    fetch(T.sourcesUrl + '?ids=' + encodeURIComponent(ids.join(','))).then(function (r) {return r.ok ? r.json() : null;})
      .then(function (data) {if (data && seq === sourcesSeq) applySources(data);}).catch(function () {});
  }
  function applySources(data) {
    (data.papers || []).forEach(function (supply) {
      var entry = find(supply.arxiv_id);
      if (entry) entry.supply = supply;
    });
    (data.missing || []).forEach(function (id) {
      var entry = find(id);
      if (entry) entry.supply = {kind: 'missing'};
    });
    selected.forEach(badges);
    messages.querySelectorAll('.chatbot-response').forEach(function (r) {tools.linkTags(r, tagLinks());});
    if (currentTable && data.papers) {currentTable.papers = data.papers; drawTable(currentTable);}
    composer();
  }

  // the question box, the suggestions and the line saying what the assistant reads
  function composer() {
    var ready = readyIds(), waiting = selected.some(function (e) {return e.state === 'reading';});
    var can = ready.length > 0 && !waiting;
    input.disabled = !ready.length;
    sendButton.disabled = !can || busy;
    tableToggle.disabled = ready.length < 2 || waiting || busy;
    if (tableToggle.disabled) builder.hidden = true;
    if (tableCost) tableCost.textContent = fmt(tableCost.dataset.text, {n: ready.length});
    var abstractOnly = selected.filter(function (e) {return e.state === 'ready' && e.supply && /^(abstract|restricted)$/.test(e.supply.kind);});
    if (tableAbstract) {
      tableAbstract.textContent = fmt(T.tableAbstract, {papers: abstractOnly.map(function (e) {return e.title;}).join(' ; ')});
      tableAbstract.hidden = !abstractOnly.length;
    }
    suggestions.hidden = !can || busy;
    suggestions.querySelector('[data-papers="one"]').hidden = ready.length !== 1;
    suggestions.querySelector('[data-papers="several"]').hidden = ready.length < 2;
    var text = T.selectFirst, warn = false;
    if (waiting) text = T.wait;
    else if (ready.length) {
      var kinds = selected.map(function (e) {return e.supply && e.supply.kind;});
      var partial = selected.some(function (e) {return e.supply && e.supply.kind === 'full' && e.supply.coverage < 100 && (e.supply.left_out || []).join() !== 'references';});
      var missing = selected.filter(function (e) {return e.state === 'failed' || (e.supply && e.supply.kind === 'missing');});
      text = [partial ? T.readsPartial : '', kinds.some(function (k) {return k === 'abstract' || k === 'restricted';}) ? T.readsAbstract : '',
              missing.length ? T.readsMissing + ' ' + missing.map(function (e) {return e.title;}).join(' ; ') : ''].filter(Boolean).join(' ');
      warn = !!text;
    }
    reading.textContent = text;
    reading.hidden = !text;
    reading.classList.toggle('warn', warn);
  }

  function updateUrl() {
    var params = new URLSearchParams(location.search);
    params.delete('c');
    params.delete('ids');
    if (messages.querySelector('.chatbot-message')) params.set('c', room);
    if (selected.length) params.set('ids', selected.map(function (e) {return e.id;}).join(','));
    var query = params.toString();
    window.history.replaceState(null, '', location.pathname + (query ? '?' + query : ''));
  }

  // --- the conversation -------------------------------------------------------

  function send(payload) {
    if (socket && socket.readyState === WebSocket.OPEN) {socket.send(JSON.stringify(payload)); return;}
    queue.push(payload);
    if (socket && socket.readyState === WebSocket.CONNECTING) return;
    socket = new WebSocket(WS + '/ws/arxivid/' + room + '/' + T.lang + '/');
    socket.addEventListener('message', receive);
    socket.addEventListener('open', function () {
      queue.splice(0).forEach(function (p) {socket.send(JSON.stringify(p));});
    });
    socket.addEventListener('close', lost);
    socket.addEventListener('error', lost);
  }
  function lost(event) {
    if (event && event.target !== socket) return;
    if (busy) messages.appendChild(node('p', 'chat-note error', T.error));
    socket = null; queue = []; currentResponse = null; pendingFigures = null; checking = {}; currentTable = null;
    finish(null);
  }

  function start() {
    busy = true;
    stopButton.hidden = false;
    loading.style.display = 'block';
    composer();
  }
  function finish(response) {
    busy = false;
    stopButton.hidden = true;
    loading.style.display = 'none';
    if (response) {
      actions(response, lastQuestion);
      tools.linkTags(response, tagLinks());
      refreshQuota();
    }
    exportButton.disabled = shareButton.disabled = !messages.querySelector('.chatbot-response');
    composer();
  }

  function asked(text, ids) {
    var key = papersKey(ids);
    if (lastPapers !== null && lastPapers !== key && messages.querySelector('.chatbot-response')) {
      messages.appendChild(node('p', 'chat-divider', T.papersChanged));
    }
    lastPapers = key;
    messages.appendChild(node('div', 'chatbot-message', T.user + ' ' + text));
    updateUrl();
    scroll();
  }

  function ask(question) {
    question = String(question || '').trim();
    var ids = readyIds();
    if (!question || busy || sendButton.disabled) return;
    lastQuestion = question;
    asked(question, ids);
    input.value = '';
    grow();
    start();
    send({message: question, selectedpapers: ids});
  }

  // the messages of the room's socket: what the papers supply, the answer streamed, its figures, its sources,
  // and the rows of a comparison table
  function receive(event) {
    var data;
    try {data = JSON.parse(event.data);} catch (_) {return;}
    if (data.type === 'chat_sources') {applySources(data); return;}
    if (data.type === 'chat_busy') return;
    if (data.type === 'chat_stopped') {stopped(); return;}
    if (data.type && data.type.indexOf('table_') === 0) {tableMessage(data); return;}
    if (data.type === 'chat_figures') {pendingFigures = data; return;}
    if (data.type === 'chat_evidence') {
      if (checking[data.answer_id]) SPChat.evidence(checking[data.answer_id], data);
      delete checking[data.answer_id];
      return;
    }
    if (data.type && data.type !== 'chat_delta') return;
    if (data.type !== 'chat_delta' && typeof data.message !== 'string') return;
    if (!currentResponse) {currentResponse = document.createElement('div'); currentResponse.className = 'chatbot-response'; messages.appendChild(currentResponse);}
    if (data.type === 'chat_delta') {SPChat.append(currentResponse, data.delta, root.dataset.ai); scroll(); return;}
    SPChat.render(currentResponse, data.message, root.dataset.ai);
    SPChat.typeset(currentResponse);
    if (pendingFigures && window.SPChatFigures) SPChatFigures.render(currentResponse, pendingFigures.figures, pendingFigures.status);
    // the claims of the answer are then checked against the papers, while the reader goes on
    if (data.evidence === 'checking') {checking[data.answer_id] = currentResponse; SPChat.evidence(currentResponse, {status: 'checking'});}
    var done = currentResponse;
    currentResponse = null; pendingFigures = null; finish(done); scroll();
  }

  function stopped() {
    var where = currentResponse || (currentTable && currentTable.element);
    if (where) where.appendChild(node('p', 'chat-stopped', T.stopped));
    else messages.appendChild(node('p', 'chat-stopped', T.stopped));
    currentResponse = null; pendingFigures = null; currentTable = null;
    finish(null);
    scroll();
  }
  stopButton.addEventListener('click', function () {
    if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({stop: true}));
  });

  // Copy and Ask again under each answer; a table is copied as Markdown
  function actions(response, question) {
    tools.actions(response, {question: question, ask: ask, labels: T, text: response.classList.contains('chat-table-block')
      ? function () {return tableMarkdown(response.table);} : null});
  }
  function sourcesMarkdown(claims) {return tools.sourcesMarkdown(claims, T);}
  // the tags of the selected papers ([Smith et al. 2021]) that answers turn into links to their pages
  function tagLinks() {
    return selected.filter(function (e) {return e.supply && e.supply.tag;})
      .map(function (e) {return {tag: e.supply.tag, url: paperUrl(e.id), title: e.title};});
  }
  var refreshQuota = tools.quota($('#chat-quota'), T.quotaUrl);
  var grow = tools.questionBox(input, function () {ask(input.value);});
  form.addEventListener('submit', function (event) {event.preventDefault(); ask(input.value);});
  suggestions.querySelectorAll('button').forEach(function (button) {
    button.addEventListener('click', function () {ask(button.dataset.question);});
  });

  $('#new-conversation').addEventListener('click', function () {
    if (socket) {var old = socket; socket = null; old.close();}
    room = newRoom();
    busy = false; queue = []; currentResponse = null; pendingFigures = null; checking = {}; currentTable = null;
    lastPapers = null;
    shareBox.hidden = true;
    messages.replaceChildren();
    finish(null);
    updateUrl();
  });

  // --- the comparison table ---------------------------------------------------

  tableToggle.addEventListener('click', function () {
    builder.hidden = !builder.hidden;
    if (!builder.hidden && columnsInput && !columnsInput.value.trim()) columnsInput.value = defaultColumns.join('\n');
  });
  var tableBuild = $('#table-build');  // absent without Premium: the builder then says what Premium gives
  if (tableBuild) $('#table-reset').addEventListener('click', function () {columnsInput.value = defaultColumns.join('\n');});
  if (tableBuild) tableBuild.addEventListener('click', function () {
    var columns = columnsInput.value.split('\n').map(function (c) {return c.trim();}).filter(Boolean);
    if (!columns.length || columns.length > 8) {alert(T.tableColumnsInvalid); return;}
    buildTable(columns);
  });
  // rows already filled for the same columns are reused by the server: building again reads only the others
  function buildTable(columns) {
    var ids = readyIds();
    if (busy || tableToggle.disabled) return;
    builder.hidden = true;
    lastQuestion = '';
    asked(T.tableTitle + ': ' + columns.join(' · '), ids);
    var element = node('div', 'chatbot-response chat-table-block');
    messages.appendChild(element);
    currentTable = {element: element, columns: columns, rows: [], papers: selected.filter(function (e) {return e.state === 'ready';})
      .map(function (e) {return {arxiv_id: e.id, title: e.title, tag: e.supply && e.supply.tag || '', label: e.label};})};
    drawTable(currentTable);
    start();
    send({table: {columns: columns}, selectedpapers: ids});
  }

  function tableMessage(data) {
    if (data.type === 'table_row' && currentTable) {
      currentTable.rows[data.index] = data.row;
      drawTable(currentTable);
    } else if (data.type === 'table_done' && currentTable) {
      var element = currentTable.element;
      renderTable(element, data.table);
      currentTable = null;
      finish(element);
    } else if (data.type === 'table_error') {
      if (currentTable) currentTable.element.remove();
      messages.appendChild(node('p', 'chat-note error', data.premium ? T.tablePremium : data.message || T.tableFailed));
      currentTable = null;
      finish(null);
    }
    scroll();
  }

  // the table while it fills: the rows already read, the others pending
  function drawTable(state) {
    var rows = state.papers.map(function (paper, i) {
      return state.rows[i] || {arxiv_id: paper.arxiv_id, title: paper.title, tag: paper.tag, label: paper.label,
                               cells: state.columns.map(function (c) {return {column: c};})};
    });
    renderTable(state.element, {columns: state.columns, rows: rows}, true);
  }

  function supplyNote(row) {
    if (row.kind === 'restricted') return T.restricted;
    if (row.kind === 'abstract') return T.abstract;
    if (row.kind === 'full' && row.coverage < 100 && (row.left_out || []).join() !== 'references') return fmt(T.partial, {n: row.coverage});
    return '';
  }

  function cellView(cell) {
    var td = node('td', 'cell-' + (cell.status || 'pending'));
    if (!cell.status) {td.appendChild(node('span', 'cell-pending', T.tablePending)); return td;}
    if (!cell.value) {
      td.appendChild(node('span', 'cell-empty', {missing: T.tableMissing, unknown: T.tableUnknown, error: T.tableError}[cell.status] || ''));
      return td;
    }
    td.appendChild(node('span', '', cell.value));
    if (cell.numbers_absent && cell.numbers_absent.length) {
      td.appendChild(node('p', 'chat-claim-warning', fmt(T.tableNumbers, {numbers: cell.numbers_absent.join(', ')})));
    }
    var details = node('details');
    details.appendChild(node('summary', '', cell.status === 'verified' ? T.tableVerified : T.tableNotFound));
    if (cell.quote) details.appendChild(node('blockquote', '', cell.quote));
    var labels = document.getElementById('chat-evidence-labels'), where = [];
    if (cell.section === 'Abstract') where.push(labels ? labels.dataset.abstract : 'Abstract');
    else if (cell.section) where.push(fmt(T.tableSection, {name: cell.section}));
    if (cell.figure) where.push(cell.figure);
    if (where.length) details.appendChild(node('p', '', where.join(' · ')));
    if (cell.link && safeLink(cell.link)) {
      var link = node('a', '', T.tableOpen);
      link.href = cell.link; link.target = '_blank'; link.rel = 'nofollow noopener noreferrer';
      details.appendChild(link);
    }
    td.appendChild(details);
    return td;
  }

  function renderTable(element, table, pending) {
    element.table = table;
    var label = node('span', 'chat-label', T.ai), wrap = node('div', 'chat-table'), grid = node('table');
    var head = node('tr');
    head.appendChild(node('th', '', T.tablePaper));
    table.columns.forEach(function (c) {head.appendChild(node('th', '', c));});
    var thead = node('thead'), tbody = node('tbody');
    thead.appendChild(head);
    (table.rows || []).forEach(function (row) {
      var tr = node('tr'), paper = node('td', 'paper'), link = node('a', '', row.title);
      link.href = paperUrl(row.arxiv_id);
      link.target = '_blank';
      if (row.tag) paper.appendChild(node('strong', '', row.tag + ' '));
      paper.appendChild(link);
      var extra = [row.label, supplyNote(row)].filter(Boolean).join(' · ');
      if (extra) paper.appendChild(node('small', '', extra));
      tr.appendChild(paper);
      row.cells.forEach(function (cell) {tr.appendChild(cellView(cell));});
      tbody.appendChild(tr);
    });
    grid.append(thead, tbody);
    wrap.appendChild(grid);
    var parts = [label, wrap];
    var lists = node('div', 'chat-table-lists');
    [['agreements', T.tableAgreements], ['disagreements', T.tableDisagreements], ['gaps', T.tableGaps]].forEach(function (pair) {
      if (!table[pair[0]] || !table[pair[0]].length) return;
      lists.appendChild(node('h4', '', pair[1]));
      var ul = node('ul');
      table[pair[0]].forEach(function (item) {ul.appendChild(node('li', '', item));});
      lists.appendChild(ul);
    });
    if (lists.children.length) parts.push(lists);
    if (!pending) {
      var bar = node('div', 'chat-answer-actions'), csv = node('button', '', T.tableCsv), md = node('button', '', T.tableMarkdown);
      csv.type = md.type = 'button';
      csv.addEventListener('click', function () {download('comparison.csv', '﻿' + tableCsv(table), 'text/csv;charset=utf-8');});
      md.addEventListener('click', function () {download('comparison.md', tableMarkdown(table), 'text/markdown;charset=utf-8');});
      bar.append(csv, md);
      if (tableBuild && table.rows.some(function (r) {return r.cells.some(function (c) {return c.status === 'error';});})) {
        var retry = node('button', '', T.tableRetry);
        retry.type = 'button';
        retry.addEventListener('click', function () {buildTable(table.columns);});
        bar.appendChild(retry);
      }
      parts.push(bar);
    }
    element.replaceChildren.apply(element, parts);
    typeset(element);
  }

  function cellText(cell) {
    return cell.value || {missing: T.tableMissing, unknown: T.tableUnknown, error: T.tableError}[cell.status] || '';
  }
  function tableCsv(table) {
    function quote(value) {return '"' + String(value == null ? '' : value).replace(/"/g, '""') + '"';}
    var header = [T.tablePaper, 'Title', 'Identifier'];
    table.columns.forEach(function (c) {header.push(c, c + ' — ' + T.tableQuote);});
    var lines = [header.map(quote).join(',')];
    table.rows.forEach(function (row) {
      var line = [row.tag, row.title, row.label];
      row.cells.forEach(function (cell) {line.push(cellText(cell), cell.quote || '');});
      lines.push(line.map(quote).join(','));
    });
    return lines.join('\r\n');
  }
  function tableMarkdown(table) {
    if (!table) return '';
    function esc(value) {return String(value || '').replace(/\|/g, '\\|').replace(/\n/g, ' ');}
    var lines = ['| ' + T.tablePaper + ' | ' + table.columns.map(esc).join(' | ') + ' |',
                 '|' + new Array(table.columns.length + 2).join('---|')];
    table.rows.forEach(function (row) {
      lines.push('| ' + esc(row.tag + ' ' + row.title) + ' | ' + row.cells.map(function (c) {return esc(cellText(c));}).join(' | ') + ' |');
    });
    [['agreements', T.tableAgreements], ['disagreements', T.tableDisagreements], ['gaps', T.tableGaps]].forEach(function (pair) {
      if (table[pair[0]] && table[pair[0]].length) lines.push('', '**' + pair[1] + '**', '', table[pair[0]].map(function (i) {return '- ' + i;}).join('\n'));
    });
    var quotes = [];
    table.rows.forEach(function (row) {
      row.cells.forEach(function (cell) {
        if (cell.quote) quotes.push('- ' + row.tag + ' — ' + cell.column + ': > ' + cell.quote + (cell.section ? ' (' + cell.section + ')' : ''));
      });
    });
    if (quotes.length) lines.push('', '**' + T.tableQuote + '**', '', quotes.join('\n'));
    return lines.join('\n');
  }

  tools.share(root, function () {return room;});

  // the recent conversations: rename (an empty name goes back to the first question) and delete
  var recent = root.querySelector('.recent-conversations');
  if (recent) recent.addEventListener('click', function (event) {
    var button = event.target.closest('button'), line = button && button.closest('li');
    if (!line) return;
    var url = function (pattern) {return pattern.replace('ROOM', encodeURIComponent(line.dataset.room));};
    if (button.hasAttribute('data-delete')) {
      if (!confirm(T.deleteConfirm)) return;
      tools.post(root, url(T.deleteUrl)).then(function () {
        line.remove();
        if (!recent.querySelector('li')) recent.remove();
      }).catch(function () {alert(T.error);});
    } else if (button.hasAttribute('data-rename')) {
      var name = prompt(T.renamePrompt, button.dataset.rename);
      if (name === null) return;
      tools.post(root, url(T.renameUrl), {title: name}).then(function (data) {
        button.dataset.rename = data.title;
        if (data.title) line.querySelector('strong').textContent = data.title;
      }).catch(function () {alert(T.error);});
    }
  });

  // --- the export of the conversation -------------------------------------------

  exportButton.addEventListener('click', function () {
    var ids = selected.map(function (e) {return e.id;});
    var lines = ['# ' + T.exportTitle, '', new Date().toISOString().slice(0, 10), ''];
    if (selected.length) {
      lines.push('## ' + T.exportPapers, '');
      selected.forEach(function (e) {
        lines.push('- ' + [e.supply && e.supply.tag, e.title, e.label || e.id].filter(Boolean).join(' — '));
      });
      lines.push('');
    }
    Array.prototype.forEach.call(messages.children, function (child) {
      if (child.classList.contains('chatbot-message')) lines.push('## ' + child.textContent.trim(), '');
      else if (child.classList.contains('chat-table-block')) lines.push(tableMarkdown(child.table), '');
      else if (child.classList.contains('chatbot-response')) lines.push(SPChat.raw(child) + sourcesMarkdown(SPChat.claims(child)), '');
      else if (child.textContent.trim()) lines.push('_' + child.textContent.trim() + '_', '');
    });
    var bibtex = ids.length ? fetch(T.exportUrl.replace('FMT', 'bib') + '?ids=' + encodeURIComponent(ids.join(',')))
      .then(function (r) {return r.ok ? r.text() : '';}).catch(function () {return '';}) : Promise.resolve('');
    bibtex.then(function (bib) {
      if (bib) lines.push('## ' + T.exportBibtex, '', '```bibtex', bib.trim(), '```', '');
      download('conversation.md', lines.join('\n'), 'text/markdown;charset=utf-8');
    });
  });

  // --- start ------------------------------------------------------------------

  $('#chatbot-toggle').addEventListener('click', function (event) {
    event.preventDefault();
    var help = $('.chatbot-instructions');
    help.style.display = help.style.display === 'none' ? 'block' : 'none';
  });

  // the stored conversation: answers rendered (SPChat.init is idempotent), tables drawn, actions added
  if (window.SPChat) SPChat.init(messages);
  messages.querySelectorAll('.chatbot-response').forEach(function (response) {
    if (response.classList.contains('chat-table-block')) {
      try {renderTable(response, JSON.parse(response.dataset.table).table);} catch (_) {return;}
    }
    actions(response, response.dataset.question);
    if (response.dataset.papers) lastPapers = response.dataset.papers;
  });
  exportButton.disabled = shareButton.disabled = !messages.querySelector('.chatbot-response');
  scroll();

  // the papers of the stored conversation, or of a collection's "chat with these" (?ids=)
  JSON.parse(document.getElementById('chat-preset').textContent).forEach(function (paper) {
    if (paper.title) addPaper(paper);
    else addById(paper.id, true);
  });
  selectionChanged();
  // a question proposed by another tool (the press kit's questions for the authors): the reader sends it
  var proposedQuestion = new URLSearchParams(location.search).get('ask');
  if (proposedQuestion && !input.value) input.value = proposedQuestion.slice(0, 600);
})();
