(function () {
  var deck = document.getElementById('deck');
  if (!deck) return;
  var items = Array.from(deck.querySelectorAll('.slide-item'));
  var status = document.getElementById('deck-status');
  var saveButton = document.getElementById('deck-save');
  var pdfButton = document.getElementById('deck-pdf');
  var exitButton = document.getElementById('deck-exit');
  var dirty = false, saving = false, version = 0, current = 0;

  function lines(value) { return value.split('\n').map(function (s) { return s.trim(); }).filter(Boolean); }
  function field(item, name) { return item.querySelector('[data-edit="' + name + '"]'); }
  function value(item, name, fallback) { var input = field(item, name); return input ? input.value : fallback; }
  function collect() {
    return items.map(function (item) {
      return {title: value(item, 'title', ''), bullets: lines(value(item, 'bullets', '')),
        secondary: lines(value(item, 'secondary', '')), notes: value(item, 'notes', ''),
        layout: value(item, 'layout', 'right'), figure_size: Number(value(item, 'figure_size', 50)),
        text_size: Number(value(item, 'text_size', 30))};
    });
  }
  function list(target, values) {
    if (!target) return;
    target.replaceChildren();
    values.forEach(function (text) { var li = document.createElement('li'); li.textContent = text; target.appendChild(li); });
  }
  function fit(item) {
    var slide = item.querySelector('.slide'), body = item.querySelector('.slide-body');
    var heading = item.querySelector('[data-preview-title]');
    var size = Number(value(item, 'text_size', 30));
    body.style.fontSize = size + 'px';
    heading.style.fontSize = (slide.dataset.kind === 'title' ? 48 : 40) + 'px';
    // Keep a long title within three lines before fitting body text.
    var titleSize = slide.dataset.kind === 'title' ? 48 : 40;
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
    items.forEach(fit);
    var overflow = items.some(function (item) { return item.classList.contains('has-overflow'); });
    document.getElementById('deck-overflow').hidden = !overflow;
    return !overflow;
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
    fitAll();
  }
  items.forEach(function (item) {
    item.querySelectorAll('[data-edit]').forEach(function (input) {
      input.addEventListener('input', function () {
        dirty = true; version += 1; saveButton.disabled = saving; status.textContent = deck.dataset.unsaved; preview(item);
      });
    });
    var frame = item.querySelector('.slide-frame');
    function resize() { frame.style.setProperty('--deck-scale', frame.clientWidth / 1280); }
    resize();
    if (window.ResizeObserver) new ResizeObserver(resize).observe(frame);
    else window.addEventListener('resize', resize);
  });
  document.getElementById('deck-edit').addEventListener('click', function () {
    var editing = deck.classList.toggle('editing'); this.setAttribute('aria-pressed', String(editing));
  });
  saveButton.addEventListener('click', function () {
    if (saving || !dirty) return;
    saving = true; saveButton.disabled = true; status.textContent = deck.dataset.saving;
    var submitted = version;
    fetch(deck.dataset.saveUrl, {method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json', 'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value},
      body: JSON.stringify({revision: deck.dataset.revision, pages: collect()})
    }).then(function (r) {
      if (r.status === 409) throw new Error('conflict');
      if (!r.ok) throw new Error('save');
      return r.json();
    }).then(function (data) {
      deck.dataset.revision = data.revision; dirty = version !== submitted;
      status.textContent = dirty ? deck.dataset.unsaved : deck.dataset.saved;
    }).catch(function (error) { status.textContent = error.message === 'conflict' ? deck.dataset.conflict : deck.dataset.error; })
      .finally(function () { saving = false; saveButton.disabled = !dirty; });
  });
  var powerpointButton = document.getElementById('deck-powerpoint');
  if (powerpointButton) powerpointButton.addEventListener('click', function () {
    if (powerpointButton.disabled || !fitAll()) return;
    powerpointButton.disabled = true; status.textContent = deck.dataset.exporting;
    fetch(deck.dataset.powerpointUrl, {method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json', 'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value},
      body: JSON.stringify({revision: deck.dataset.revision, pages: collect()})
    }).then(function (r) {
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
  document.getElementById('deck-notes').addEventListener('change', function () { deck.classList.toggle('hide-notes', !this.checked); });
  function show(index) {
    current = Math.max(0, Math.min(items.length - 1, index));
    items.forEach(function (item, i) { item.classList.toggle('is-current', i === current); });
    fitAll();
  }
  function exit() {
    deck.classList.remove('presenting'); exitButton.hidden = true;
    if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(function () {});
  }
  document.getElementById('deck-present').addEventListener('click', function () {
    deck.classList.add('presenting'); exitButton.hidden = false; show(0);
    if (deck.requestFullscreen) deck.requestFullscreen().catch(function () {});
  });
  exitButton.addEventListener('click', exit);
  document.addEventListener('fullscreenchange', function () { if (!document.fullscreenElement) exit(); });
  document.addEventListener('keydown', function (e) {
    if (!deck.classList.contains('presenting')) return;
    if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); show(current + 1); }
    if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); show(current - 1); }
    if (e.key === 'Escape') exit();
  });
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
  window.addEventListener('beforeunload', function (e) { if (dirty) { e.preventDefault(); e.returnValue = ''; } });
  window.addEventListener('beforeprint', fitAll);
  if (document.fonts) document.fonts.ready.then(fitAll);
  fitAll();
}());
