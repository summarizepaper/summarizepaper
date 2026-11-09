(function () {
  var deck = document.getElementById('deck');
  if (!deck) return;
  var shared = !!deck.dataset.shared;
  var status = document.getElementById('deck-status');
  var saveButton = document.getElementById('deck-save');
  var pdfButton = document.getElementById('deck-pdf');
  var exitButton = document.getElementById('deck-exit');
  var template = document.getElementById('slide-template');
  var figuresNode = document.getElementById('deck-figures');
  var figures = figuresNode ? JSON.parse(figuresNode.textContent) : {};
  var dirty = false, saving = false, version = 0, current = 0, presenter = null, started = null, clock = null;

  function items() { return Array.from(deck.querySelectorAll('.slide-item')); }
  function lines(value) { return value.split('\n').map(function (s) { return s.trim(); }).filter(Boolean); }
  function field(item, name) { return item.querySelector('[data-edit="' + name + '"]'); }
  function value(item, name, fallback) { var input = field(item, name); return input ? input.value : fallback; }
  function csrf() { var input = document.querySelector('[name=csrfmiddlewaretoken]'); return input ? input.value : ''; }
  function page(item) {
    var origin = item.dataset.origin === '' ? null : Number(item.dataset.origin);
    var data = {origin: origin, title: value(item, 'title', ''), bullets: lines(value(item, 'bullets', '')),
      secondary: lines(value(item, 'secondary', '')), notes: value(item, 'notes', ''),
      layout: value(item, 'layout', 'right'), figure_size: Number(value(item, 'figure_size', 50)),
      text_size: Number(value(item, 'text_size', 30))};
    if (field(item, 'figure')) data.figure = value(item, 'figure', '');
    return data;
  }
  function collect() { return items().map(page); }
  function list(target, values) {
    if (!target) return;
    target.replaceChildren();
    values.forEach(function (text) { var li = document.createElement('li'); li.textContent = text; target.appendChild(li); });
  }
  function fit(item) {
    var slide = item.querySelector('.slide'), body = item.querySelector('.slide-body');
    var heading = item.querySelector('[data-preview-title]');
    var size = Number(value(item, 'text_size', slide.style.getPropertyValue('--slide-font').replace('px', '') || 30));
    body.style.fontSize = size + 'px';
    // Keep a long title within three lines before fitting body text.
    var titleSize = slide.dataset.kind === 'title' ? 48 : 40;
    heading.style.fontSize = titleSize + 'px';
    while (heading.offsetHeight > 140 && titleSize > 26) { titleSize -= 2; heading.style.fontSize = titleSize + 'px'; }
    function overflows() {
      return body.scrollHeight > body.clientHeight + 2 || body.scrollWidth > body.clientWidth + 2 ||
        slide.scrollHeight > slide.clientHeight + 2 ||
        Array.from(body.querySelectorAll('.slide-copy, .slide-secondary')).some(function (el) {
          return el.scrollHeight > el.clientHeight + 2 || el.scrollWidth > el.clientWidth + 2;
        });
    }
    while (overflows() && size > 20) { size -= 1; body.style.fontSize = size + 'px'; }
    item.classList.toggle('has-overflow', overflows());
  }
  function fitAll() {
    var all = items();
    all.forEach(function (item, i) {
      fit(item);
      var number = item.querySelector('[data-slide-number]');
      if (number) number.textContent = (i + 1) + ' / ' + all.length;
    });
    var overflow = all.some(function (item) { return item.classList.contains('has-overflow'); });
    document.getElementById('deck-overflow').hidden = !overflow;
    return !overflow;
  }
  function showFigure(item) {
    var body = item.querySelector('.slide-body'), id = value(item, 'figure', null);
    if (id === null) return;
    var figure = body.querySelector('figure'), chosen = figures[id];
    if (!chosen) { if (figure) figure.remove(); }
    else {
      if (!figure) {
        figure = document.createElement('figure');
        figure.appendChild(document.createElement('img'));
        figure.appendChild(document.createElement('figcaption'));
        body.appendChild(figure);
      }
      var img = figure.querySelector('img');
      if (img.getAttribute('src') !== chosen.url) { img.src = chosen.url; img.alt = chosen.caption; }
      figure.querySelector('figcaption').textContent = chosen.credit;
    }
    body.classList.toggle('with-figure', !!chosen);
    item.classList.toggle('has-figure', !!chosen);
  }
  function preview(item) {
    var slide = item.querySelector('.slide');
    item.querySelector('[data-preview-title]').textContent = value(item, 'title', '');
    list(item.querySelector('[data-preview-bullets]'), lines(value(item, 'bullets', '')));
    list(item.querySelector('[data-preview-secondary]'), lines(value(item, 'secondary', '')));
    var notes = item.querySelector('[data-preview-notes]');
    notes.textContent = value(item, 'notes', ''); notes.hidden = !notes.textContent.trim();
    slide.dataset.layout = value(item, 'layout', 'right');
    slide.style.setProperty('--figure-share', Number(value(item, 'figure_size', 50)) + '%');
    showFigure(item);
    fitAll();
    if (presenter) updatePresenter();
  }
  function changed() {
    dirty = true; version += 1;
    if (saveButton) saveButton.disabled = saving;
    status.textContent = deck.dataset.unsaved;
  }
  function watchFrame(item) {
    var frame = item.querySelector('.slide-frame');
    function resize() { frame.style.setProperty('--deck-scale', frame.clientWidth / 1280); }
    resize();
    if (window.ResizeObserver) new ResizeObserver(resize).observe(frame);
    else window.addEventListener('resize', resize);
  }
  items().forEach(watchFrame);

  // Editing: the text, layout and figure of a slide, and the order of the slides.
  deck.addEventListener('input', function (e) {
    var item = e.target.closest('.slide-item');
    if (!item || !e.target.matches('[data-edit]')) return;
    changed(); preview(item);
  });
  function fresh(from) {
    var item;
    if (from) {
      item = from.cloneNode(true);
      // a clone keeps the values typed in its fields, which cloneNode does not copy
      Array.from(from.querySelectorAll('[data-edit]')).forEach(function (input, i) {
        item.querySelectorAll('[data-edit]')[i].value = input.value;
      });
    } else {
      item = template.content.firstElementChild.cloneNode(true);
      field(item, 'title').value = deck.dataset.newTitle;
    }
    return item;
  }
  deck.addEventListener('click', function (e) {
    var button = e.target.closest('[data-action]');
    if (!button || deck.classList.contains('presenting')) return;
    var item = button.closest('.slide-item'), action = button.dataset.action, added = null;
    if (action === 'up' && item.previousElementSibling && item.previousElementSibling.classList.contains('slide-item')) item.parentNode.insertBefore(item, item.previousElementSibling);
    else if (action === 'down' && item.nextElementSibling && item.nextElementSibling.classList.contains('slide-item')) item.parentNode.insertBefore(item.nextElementSibling, item);
    else if (action === 'duplicate') { added = fresh(item); item.after(added); }
    else if (action === 'add') { added = fresh(null); item.after(added); }
    else if (action === 'delete') {
      if (items().length < 2 || !window.confirm(deck.dataset.confirmDelete)) return;
      item.remove();
    } else return;
    if (added) { watchFrame(added); preview(added); }
    changed(); fitAll();
    (added || item).scrollIntoView({block: 'nearest'});
  });

  function send(url, body) {
    return fetch(url, {method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf()}, body: JSON.stringify(body)});
  }
  if (saveButton) {
    document.getElementById('deck-edit').addEventListener('click', function () {
      var editing = deck.classList.toggle('editing'); this.setAttribute('aria-pressed', String(editing));
    });
    saveButton.addEventListener('click', function () {
      if (saving || !dirty) return;
      saving = true; saveButton.disabled = true; status.textContent = deck.dataset.saving;
      var submitted = version;
      send(deck.dataset.saveUrl, {revision: deck.dataset.revision, pages: collect()}).then(function (r) {
        if (r.status === 409) throw new Error('conflict');
        if (!r.ok) throw new Error('save');
        return r.json();
      }).then(function (data) {
        deck.dataset.revision = data.revision; dirty = version !== submitted;
        status.textContent = dirty ? deck.dataset.unsaved : deck.dataset.saved;
      }).catch(function (error) { status.textContent = error.message === 'conflict' ? deck.dataset.conflict : deck.dataset.error; })
        .finally(function () { saving = false; saveButton.disabled = !dirty; });
    });
  }

  // Rewriting a slide: the answer fills its fields, to check and save.
  deck.addEventListener('click', function (e) {
    var button = e.target.closest('[data-rewrite]');
    if (!button) return;
    var item = button.closest('.slide-item'), buttons = item.querySelectorAll('[data-rewrite]');
    buttons.forEach(function (b) { b.disabled = true; });
    status.textContent = deck.dataset.rewriting;
    send(deck.dataset.rewriteUrl, {how: button.dataset.rewrite, page: page(item)}).then(function (r) {
      if (r.status === 429) throw new Error(deck.dataset.rewriteQuota);
      if (!r.ok) throw new Error(deck.dataset.rewriteError);
      return r.json();
    }).then(function (data) {
      field(item, 'title').value = data.title;
      field(item, 'bullets').value = data.bullets.join('\n');
      field(item, 'notes').value = data.notes;
      changed(); preview(item);
      status.textContent = deck.dataset.rewritten + (data.new_numbers.length ? ' ' + deck.dataset.newNumbers + ' ' + data.new_numbers.join(', ') : '');
    }).catch(function (error) { status.textContent = error.message || deck.dataset.rewriteError; })
      .finally(function () { buttons.forEach(function (b) { b.disabled = false; }); });
  });

  var powerpointButton = document.getElementById('deck-powerpoint');
  if (powerpointButton) powerpointButton.addEventListener('click', function () {
    if (powerpointButton.disabled || !fitAll()) return;
    powerpointButton.disabled = true; status.textContent = deck.dataset.exporting;
    send(deck.dataset.powerpointUrl, {revision: deck.dataset.revision, pages: collect()}).then(function (r) {
      if (r.status === 409) throw new Error(deck.dataset.conflict);
      if (!r.ok) return r.json().then(function (data) { throw new Error(data.message || deck.dataset.exportError); });
      return r.blob();
    }).then(function (blob) {
      var url = URL.createObjectURL(blob), link = document.createElement('a');
      link.href = url; link.download = deck.dataset.exportName || 'SummarizePaper-JournalClub.pptx';
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 60000);
      status.textContent = deck.dataset.exported;
    }).catch(function (error) { status.textContent = error.message || deck.dataset.exportError; })
      .finally(function () { powerpointButton.disabled = false; });
  });

  function exportBeamer(button, toOverleaf) {
    if (button.disabled || !fitAll()) return;
    // Opened now, while the click allows it; the form posts the project to it once the server has written it.
    var tab = toOverleaf ? window.open('', 'overleaf-export') : null;
    if (tab) try { tab.document.title = 'Overleaf'; tab.document.body.textContent = deck.dataset.beamerExporting; } catch (_) { /* another origin already */ }
    button.disabled = true; status.textContent = deck.dataset.beamerExporting;
    send(deck.dataset.beamerUrl, {revision: deck.dataset.revision, pages: collect()}).then(function (r) {
      if (r.status === 409) throw new Error(deck.dataset.conflict);
      if (!r.ok) return r.json().then(function (data) { throw new Error(data.message || deck.dataset.beamerError); });
      return r.blob();
    }).then(function (blob) {
      if (!toOverleaf) {
        var url = URL.createObjectURL(blob), link = document.createElement('a');
        link.href = url; link.download = (deck.dataset.exportName || 'SummarizePaper-JournalClub.pptx').replace(/\.pptx$/, '-beamer.zip');
        document.body.appendChild(link); link.click(); link.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 60000);
        status.textContent = deck.dataset.beamerExported;
        return;
      }
      return new Promise(function (resolve, reject) {
        var reader = new FileReader();
        reader.onload = function () { resolve(reader.result); };
        reader.onerror = reject;
        reader.readAsDataURL(new Blob([blob], {type: 'application/zip'}));
      }).then(function (dataUrl) {
        var form = document.createElement('form');
        form.method = 'post'; form.action = deck.dataset.overleafDocs; form.target = tab ? 'overleaf-export' : '_blank';
        [['snip_uri', dataUrl], ['main_document', 'main.tex'], ['engine', 'xelatex']].forEach(function (pair) {
          var input = document.createElement('input');
          input.type = 'hidden'; input.name = pair[0]; input.value = pair[1];
          form.appendChild(input);
        });
        document.body.appendChild(form); form.submit(); form.remove();
        status.textContent = deck.dataset.overleafOpened;
      });
    }).catch(function (error) {
      if (tab) tab.close();
      status.textContent = error.message || deck.dataset.beamerError;
    }).finally(function () { button.disabled = false; });
  }
  var overleafButton = document.getElementById('deck-overleaf');
  if (overleafButton) overleafButton.addEventListener('click', function () { exportBeamer(overleafButton, true); });
  var beamerButton = document.getElementById('deck-beamer');
  if (beamerButton) beamerButton.addEventListener('click', function () { exportBeamer(beamerButton, false); });

  var share = document.getElementById('deck-share');
  if (share) share.addEventListener('click', function (e) {
    var button = e.target.closest('[data-share]');
    if (!button) return;
    if (button.dataset.share === 'share' && dirty) { status.textContent = share.dataset.unsaved; return; }
    button.disabled = true;
    SP.post(share.dataset.url, {action: button.dataset.share}).then(function (json) {
      if (json.status !== 'ok') throw new Error();
      var link = share.querySelector('.deck-share-link');
      link.hidden = !json.url; link.querySelector('input').value = json.url;
      link.querySelector('[data-share-timing]').textContent = json.timing || '';
      share.querySelector('[data-share="unshare"]').hidden = !json.url;
      if (json.url) link.querySelector('input').select();
    }).catch(function () { status.textContent = share.dataset.error; })
      .finally(function () { button.disabled = false; });
  });

  document.getElementById('deck-notes').addEventListener('change', function () { deck.classList.toggle('hide-notes', !this.checked); });
  document.getElementById('deck-sources').addEventListener('change', function () { document.body.classList.toggle('hide-sources', !this.checked); });

  // Presenting, with an optional presenter window: notes, next slide and timer.
  function show(index) {
    var all = items();
    current = Math.max(0, Math.min(all.length - 1, index));
    all.forEach(function (item, i) { item.classList.toggle('is-current', i === current); });
    fitAll();
    if (presenter) updatePresenter();
  }
  function exit() {
    deck.classList.remove('presenting'); exitButton.hidden = true;
    if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(function () {});
  }
  function present() {
    deck.classList.add('presenting'); exitButton.hidden = false; show(current);
    if (deck.requestFullscreen) deck.requestFullscreen().catch(function () {});
  }
  function minutes(ms) { var s = Math.floor(ms / 1000); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); }
  function updatePresenter() {
    if (!presenter || presenter.closed) { presenter = null; return; }
    var all = items(), item = all[current], next = all[current + 1], doc = presenter.document;
    doc.getElementById('p-count').textContent = (current + 1) + ' / ' + all.length;
    doc.getElementById('p-title').textContent = value(item, 'title', item.querySelector('[data-preview-title]').textContent);
    doc.getElementById('p-notes').textContent = value(item, 'notes', item.querySelector('[data-preview-notes]').textContent) || '—';
    doc.getElementById('p-next').textContent = next ? deck.dataset.nextLabel + ' → ' + next.querySelector('[data-preview-title]').textContent : deck.dataset.endLabel;
    var planned = Number(deck.dataset.minutes) * 60000, elapsed = started ? Date.now() - started : 0;
    var timer = doc.getElementById('p-timer');
    timer.textContent = deck.dataset.elapsedLabel + ' ' + minutes(elapsed) + ' · ' + deck.dataset.plannedLabel + ' ' + minutes(planned);
    timer.className = elapsed > planned ? 'late' : elapsed > planned * (current + 1) / all.length + 60000 ? 'behind' : '';
  }
  function openPresenter() {
    presenter = window.open('', 'sp-presenter', 'popup,width=900,height=600');
    if (!presenter) { status.textContent = deck.dataset.presenterBlocked; return; }
    var doc = presenter.document;
    doc.open();
    doc.write('<!doctype html><html lang="' + deck.dataset.lang + '"><head><meta charset="utf-8"><title>Presenter</title><style>' +
      'body{font:18px/1.5 system-ui,sans-serif;margin:24px;background:#111;color:#eee}' +
      '#p-head{display:flex;justify-content:space-between;color:#aaa}#p-title{font-size:30px;margin:12px 0}' +
      '#p-notes{font-size:24px;white-space:pre-wrap;background:#1d1d1d;padding:16px;border-radius:8px}' +
      '#p-next{margin-top:18px;color:#aaa}#p-timer{font-variant-numeric:tabular-nums}.behind{color:#f5b041}.late{color:#ff6b6b}' +
      'button{font:inherit;margin-right:8px}</style></head><body><div id="p-head"><span id="p-count"></span><span id="p-timer"></span></div>' +
      '<h1 id="p-title"></h1><div id="p-notes"></div><p id="p-next"></p>' +
      '<p><button id="p-prev">←</button><button id="p-go">→</button></p></body></html>');
    doc.close();
    doc.getElementById('p-prev').addEventListener('click', function () { show(current - 1); });
    doc.getElementById('p-go').addEventListener('click', function () { show(current + 1); });
    doc.addEventListener('keydown', keys);
    started = Date.now();
    clearInterval(clock);
    clock = setInterval(function () { if (presenter && !presenter.closed) updatePresenter(); else clearInterval(clock); }, 1000);
    updatePresenter();
  }
  document.getElementById('deck-present').addEventListener('click', function () { current = 0; present(); });
  document.getElementById('deck-presenter').addEventListener('click', function () { current = 0; present(); openPresenter(); });
  exitButton.addEventListener('click', exit);
  document.addEventListener('fullscreenchange', function () { if (!document.fullscreenElement) exit(); });
  function keys(e) {
    if (!deck.classList.contains('presenting') && !(presenter && e.view === presenter)) return;
    if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); show(current + 1); }
    if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); show(current - 1); }
    if (e.key === 'Escape') exit();
  }
  document.addEventListener('keydown', keys);
  deck.addEventListener('click', function (e) {
    if (deck.classList.contains('presenting') && !e.target.closest('button')) show(current + (e.clientX > window.innerWidth / 3 ? 1 : -1));
  });
  pdfButton.addEventListener('click', function () {
    if (!fitAll()) return;
    pdfButton.disabled = true;
    var images = Array.from(deck.querySelectorAll('img'));
    var ready = Promise.all(images.map(function (img) {
      return new Promise(function (resolve) {
        if (img.complete) return resolve();
        img.addEventListener('load', resolve, {once: true}); img.addEventListener('error', resolve, {once: true});
        setTimeout(resolve, 5000);
      });
    }));
    ready.then(function () {
      if (images.some(function (img) { return !img.complete || !img.naturalWidth; })) {
        status.textContent = deck.dataset.imageError; return;
      }
      return (document.fonts ? document.fonts.ready : Promise.resolve()).then(function () { if (fitAll()) window.print(); });
    }).finally(function () { pdfButton.disabled = false; });
  });
  window.addEventListener('beforeunload', function (e) { if (dirty && !shared) { e.preventDefault(); e.returnValue = ''; } });
  window.addEventListener('beforeprint', fitAll);
  if (document.fonts) document.fonts.ready.then(fitAll);
  fitAll();
}());
