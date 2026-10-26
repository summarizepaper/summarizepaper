(function () {
  'use strict';
  var imageHosts = ['arxiv.org', 'www.arxiv.org', 'pmc.ncbi.nlm.nih.gov', 'cdn.ncbi.nlm.nih.gov'];
  function imageURL(value) {
    try { var u = new URL(value); return u.protocol === 'https:' && imageHosts.includes(u.hostname) &&
      !u.username && !u.password && (!u.port || u.port === '443'); } catch (_) { return false; }
  }
  function sourceURL(value) {
    try {var u = new URL(value); return u.protocol === 'https:' && !u.username && !u.password;} catch (_) {return false;}
  }
  function node(tag, text) {
    var e = document.createElement(tag);
    if (text) e.textContent = text;
    return e;
  }
  function canonical(id) {
    id = id.replace(/\//g, '--');
    return /^(?:\d{4}\.\d{4,5}|[a-z.-]+--\d{7})$/i.test(id) ? id + 'v1' : id;
  }
  function mount(box) {
    if (box.dataset.figuresMounted) return;
    box.dataset.figuresMounted = "1";
    var select = box.querySelector('select');
    var picker = box.querySelector('.chat-figures-picker');
    var status = box.querySelector('.chat-figures-status');
    var grid = box.querySelector('.chat-figures-grid');
    var retry = box.querySelector('.chat-figures-retry');
    var source = box.querySelector('.chat-figures-source');
    var dialog = box.parentElement.querySelector('dialog');
    var enlarged = dialog.querySelector('.chat-figure-enlarged');
    var selected = document.getElementById('selected-papers');
    var version = 0, controller, current = '', state = '', signature = '', opener, questionPrefix = '';
    var responses = new Map();

    function clear(element) {
      if (window.MathJax && typeof window.MathJax.typesetClear === 'function') window.MathJax.typesetClear([element]);
      element.replaceChildren();
    }
    function typeset(element) {
      if (window.SPChat) window.SPChat.typeset(element);
    }

    function close() {
      if (!dialog.open) return;
      if (typeof dialog.close === 'function') dialog.close();
      else dialog.removeAttribute('open');
    }
    function question(figure, data) {
      close();
      var input = box.closest('.chatbot-body').querySelector('.chatbot-form textarea, .chatbot-form input');
      var prefix = box.dataset.question.replace('{figure}', figure.label.replace(/[:：]\s*$/, '')).replace('{paper}', data.title).replace('{id}', data.paper_id);
      var draft = questionPrefix && input.value.indexOf(questionPrefix) === 0 ? input.value.slice(questionPrefix.length) : input.value;
      questionPrefix = prefix;
      input.value = (prefix + draft).slice(0, input.maxLength > 0 ? input.maxLength : 6000);
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.focus();
    }
    function pictures(figure, parent, large) {
      (figure.images || []).filter(imageURL).slice(0, large ? 8 : 1).forEach(function (url) {
        var img = node('img'); img.src = url; img.alt = figure.label;
        img.loading = 'lazy'; img.referrerPolicy = 'no-referrer';
        img.addEventListener('error', function () {
          img.replaceWith(node('p', box.dataset.imageError));
        }, {once: true});
        parent.appendChild(img);
      });
    }
    function original(url) {
      if (!sourceURL(url)) return node('span');
      var a = node('a', box.dataset.source); a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer';
      return a;
    }
    function createCard(figure, data) {
        var card = node('figure');
        var button = node('button'); button.type = 'button'; button.className = 'chat-figure-thumb';
        button.setAttribute('aria-label', box.dataset.enlarge + ' — ' + figure.label);
        pictures(figure, button, false); card.appendChild(button);
        card.appendChild(node('figcaption', figure.caption || figure.label));
        var ask = node('button', box.dataset.ask); ask.type = 'button'; ask.className = 'btn btn-sm';
        ask.addEventListener('click', function () {question(figure, data);}); card.appendChild(ask);
        button.addEventListener('click', function () {
          clear(enlarged);
          enlarged.appendChild(node('h3', data.title + ' — ' + figure.label));
          pictures(figure, enlarged, true);
          enlarged.appendChild(node('p', figure.caption));
          enlarged.appendChild(original(figure.source_url || data.source_url));
          opener = button;
          if (typeof dialog.showModal === 'function') dialog.showModal();
          else dialog.setAttribute('open', '');
          typeset(enlarged);
        });
      return card;
    }
    function draw(data) {
      state = data.status;
      clear(grid);
      status.classList.remove('is-loading');
      status.textContent = state === 'ready' ? data.title :
        (state === 'license' ? box.dataset.license : state === 'not_ready' ? box.dataset.pending :
          state === 'unavailable' ? box.dataset.error : box.dataset.empty);
      retry.hidden = !['not_ready', 'unavailable'].includes(state);
      source.hidden = !sourceURL(data.source_url);
      if (!source.hidden) {source.href = data.source_url; source.textContent = box.dataset.source;}
      (data.figures || []).forEach(function (figure) {
        if (!(figure.images || []).some(imageURL)) return;
        grid.appendChild(createCard(figure, data));
      });
      typeset(grid);
    }
    function load(force) {
      if (!box.open) return;
      var id = select.value;
      version++; var mine = version;
      if (controller) controller.abort();
      close(); clear(grid); retry.hidden = true; source.hidden = true;
      current = id;
      if (!id) {state = ''; status.classList.remove('is-loading'); status.textContent = box.dataset.none; return;}
      if (!force && responses.has(id)) {draw(responses.get(id)); return;}
      state = 'loading';
      controller = new AbortController();
      status.textContent = box.dataset.loading; status.classList.add('is-loading');
      fetch(box.dataset.url.replace('PAPER_ID', encodeURIComponent(id)), {signal: controller.signal, credentials: 'same-origin'})
        .then(function (response) {return response.json();})
        .then(function (data) {
          if (mine !== version || select.value !== id || !box.open) return;
          if (!data || !['ready', 'empty', 'license', 'unsupported', 'not_ready', 'unavailable'].includes(data.status)) throw new Error('invalid response');
          if (!['not_ready', 'unavailable'].includes(data.status)) responses.set(id, data);
          draw(data);
        }).catch(function (error) {
          if (error.name !== 'AbortError' && mine === version && box.open) draw({status: 'unavailable'});
        });
    }
    function refresh(skipLoad) {
      var papers = [];
      if (box.dataset.paper) papers.push({id: canonical(box.dataset.paper), title: box.dataset.title || box.dataset.paper});
      else if (selected) selected.querySelectorAll('li').forEach(function (item) {
        var remove = item.querySelector('.remove-symbol[id]'), title = item.querySelector('strong');
        if (remove && title && !papers.some(function (p) {return p.id === canonical(remove.id);}))
          papers.push({id: canonical(remove.id), title: title.textContent});
      });
      var next = JSON.stringify(papers);
      if (next === signature) {if (!skipLoad && box.open && state === 'not_ready') load(true); return;}
      signature = next;
      select.replaceChildren();
      papers.slice(0, 10).forEach(function (p) {var option = node('option', p.title); option.value = p.id; select.appendChild(option);});
      if (papers.some(function (p) {return p.id === current;})) select.value = current;
      picker.hidden = papers.length <= 1;
      select.disabled = papers.length <= 1;
      if (!skipLoad && box.open) load(false);
    }
    box.addEventListener('toggle', function () {
      if (box.open) {refresh(true); load(false);}
      else {version++; if (controller) controller.abort(); close(); status.classList.remove('is-loading');}
    });
    select.addEventListener('change', function () {load(false);});
    retry.addEventListener('click', function () {load(true);});
    dialog.querySelector('.chat-figure-close').addEventListener('click', close);
    dialog.addEventListener('close', function () {if (opener && opener.isConnected) opener.focus();});
    if (selected) new MutationObserver(function () {refresh(false);}).observe(selected, {childList: true, subtree: true});
    window.SPChatFigures.render = function (element, items, result) {
      if (result === 'not_requested' || !result) return;
      var section = node('div'); section.className = 'chat-visual-evidence';
      if (result !== 'ready') {
        section.appendChild(node('p', box.dataset.visionUnavailable));
      } else {
        section.appendChild(node('p', box.dataset.visionUsed));
        var cards = node('div'); cards.className = 'chat-inline-figures';
        (items || []).slice(0, 2).forEach(function (item) {
          if (!item.figure || !(item.figure.images || []).some(imageURL)) return;
          var card = createCard(item.figure, item);
          card.appendChild(node('small', item.title + ' (' + item.paper_id + ')'));
          cards.appendChild(card);
        });
        section.appendChild(cards);
      }
      element.appendChild(section);
      typeset(section);
    };
    refresh();
  }
  window.SPChatFigures = window.SPChatFigures || {};
  window.SPChatFigures.init = function (root) {(root || document).querySelectorAll('.chat-figures').forEach(mount);};
  window.SPChatFigures.init(document);
})();
