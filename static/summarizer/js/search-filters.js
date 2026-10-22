(function () {
  document.querySelectorAll('[data-journal-picker]').forEach(function (picker) {
    var input = picker.querySelector('[data-journal-query]');
    var results = picker.querySelector('[data-journal-results]');
    var status = picker.querySelector('[role="status"]');
    var timer, controller, sequence = 0;
    function remove(button) { button.closest('.journal-chip').remove(); }
    picker.querySelectorAll('[data-remove-journal]').forEach(function (button) {
      button.addEventListener('click', function () { remove(button); });
    });
    function choose(item, name) {
      var list = picker.querySelector('[data-journal-list="' + name + '"]');
      var existing = picker.querySelector('.journal-chip[data-id="' + item.id + '"]');
      if (existing && existing.parentElement === list) return;
      if (list.children.length >= 5) { status.textContent = picker.dataset.limit; return; }
      if (existing) existing.remove();
      var chip = document.createElement('span'); chip.className = 'journal-chip'; chip.dataset.id = item.id;
      var hidden = document.createElement('input'); hidden.type = 'hidden'; hidden.name = name; hidden.value = item.id;
      var text = document.createElement('span'); text.textContent = item.name;
      var button = document.createElement('button'); button.type = 'button'; button.textContent = '×';
      button.setAttribute('aria-label', picker.dataset.remove + ' ' + item.name);
      button.addEventListener('click', function () { remove(button); });
      chip.append(hidden, text, button); list.append(chip);
      input.value = ''; results.replaceChildren(); status.textContent = '';
      sequence += 1;
      input.focus();
    }
    input.addEventListener('input', function () {
      clearTimeout(timer); if (controller) controller.abort();
      var current = ++sequence;
      results.replaceChildren(); status.textContent = '';
      var query = input.value.trim(); if (query.length < 2) return;
      timer = setTimeout(async function () {
        controller = new AbortController(); status.textContent = picker.dataset.loading;
        try {
          var response = await fetch(picker.dataset.url + '?q=' + encodeURIComponent(query), {signal: controller.signal});
          if (!response.ok) throw new Error('unavailable');
          var data = await response.json(); if (current !== sequence) return;
          results.replaceChildren(); status.textContent = data.results.length ? '' : picker.dataset.empty;
          data.results.forEach(function (item) {
            if (!/^S\d{1,15}$/.test(item.id)) return;
            var row = document.createElement('li'); var text = document.createElement('span'); text.textContent = item.name;
            var detail = document.createElement('small'); detail.textContent = [item.issn, item.publisher].filter(Boolean).join(' · ');
            text.append(detail); row.append(text);
            [['journals', picker.dataset.include], ['exclude_journals', picker.dataset.exclude]].forEach(function (choice) {
              var button = document.createElement('button'); button.type = 'button'; button.textContent = choice[1];
              button.setAttribute('aria-label', choice[1] + ' ' + item.name);
              button.addEventListener('click', function () { choose(item, choice[0]); }); row.append(button);
            });
            results.append(row);
          });
        } catch (error) { if (error.name !== 'AbortError' && current === sequence) status.textContent = picker.dataset.error; }
      }, 400);
    });
  });
}());
