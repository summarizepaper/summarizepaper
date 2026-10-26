// Digest settings: find a category, show the chosen ones, and preview what the settings would send.
(function () {
  var form = document.getElementById('digest-form');
  if (!form) { return; }
  var boxes = Array.prototype.slice.call(form.querySelectorAll('input[name="categories"]'));
  var chips = form.querySelector('[data-chips]');
  var search = document.getElementById('cat-search');

  function plain(text) {
    return (text || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
  }

  function showChosen() {
    chips.textContent = '';
    boxes.filter(function (box) { return box.checked; }).forEach(function (box) {
      var chip = document.createElement('span');
      chip.textContent = box.dataset.label || box.value;
      chips.appendChild(chip);
    });
  }

  if (search) {
    search.addEventListener('input', function () {
      var wanted = plain(search.value.trim());
      form.querySelectorAll('[data-cat-group]').forEach(function (group) {
        var shown = 0;
        group.querySelectorAll('label').forEach(function (label) {
          var box = label.querySelector('input');
          var hit = !wanted || plain(box.value + ' ' + (box.dataset.label || '') + ' ' + (box.dataset.name || '')).indexOf(wanted) !== -1;
          label.hidden = !hit;
          shown += hit ? 1 : 0;
        });
        group.hidden = !shown;
        if (wanted) { group.open = shown > 0; }
      });
    });
  }

  // the live preview
  var panel = document.getElementById('digest-try');
  var body = panel && panel.querySelector('[data-try-body]');
  var timer = null;
  var sequence = 0;

  function refresh() {
    if (!body) { return; }
    var data = new FormData(form);
    data.delete('action');
    var mine = ++sequence;
    panel.setAttribute('aria-busy', 'true');
    fetch(form.dataset.tryUrl, {method: 'POST', body: data, credentials: 'same-origin'})
      .then(function (response) { return response.ok ? response.text() : ''; })
      .then(function (html) {
        if (mine !== sequence) { return; }
        panel.removeAttribute('aria-busy');
        if (html) { body.innerHTML = html; }
      })
      .catch(function () { panel.removeAttribute('aria-busy'); });
  }

  function later() {
    clearTimeout(timer);
    timer = setTimeout(refresh, 900);
  }

  form.addEventListener('change', function (event) {
    if (event.target.name === 'categories') { showChosen(); }
    if (event.target.id !== 'cat-search') { later(); }
  });
  form.addEventListener('input', function (event) {
    if (event.target.matches('#keywords, #profile, #reference_papers')) { later(); }
  });
  if (boxes.some(function (box) { return box.checked; })) { refresh(); }
}());
