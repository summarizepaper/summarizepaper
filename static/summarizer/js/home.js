(function () {
  document.querySelectorAll('form[data-research-search]').forEach(function (form) {
    var input = form.elements.arxiv_id || form.elements.q;
    var buttons = Array.from(form.querySelectorAll('button[type="submit"]'));
    var button = buttons[0];
    var status = form.querySelector('[data-search-status]');
    var label = button.textContent;
    function reset() {
      form.removeAttribute('aria-busy');
      buttons.forEach(function (item) { item.disabled = false; });
      button.textContent = label;
      if (status) status.hidden = true;
    }
    form.addEventListener('submit', function (event) {
      var hasFilter = Array.from(form.querySelectorAll('input[name=orcid], input[name=journals]')).some(function (item) { return item.value.trim(); });
      if ((!input.value.trim() && !hasFilter) || form.getAttribute('aria-busy') === 'true') {
        event.preventDefault();
        if (!input.value.trim()) input.focus();
        return;
      }
      form.setAttribute('aria-busy', 'true');
      button.textContent = button.dataset.loading;
      buttons.forEach(function (item) { item.disabled = true; });
      if (status) status.hidden = false;
    });
    var examples = form.querySelector('.search-examples');
    if (examples) {
      examples.hidden = false;
      examples.querySelectorAll('button[data-query]').forEach(function (example) {
        example.addEventListener('click', function () {
          input.value = example.dataset.query;
          form.querySelector('input[name="field"][value="' + example.dataset.field + '"]').checked = true;
          input.focus();
        });
      });
    }
    window.addEventListener('pageshow', reset);
  });
}());
