(function () {
  document.querySelectorAll('form[data-research-search]').forEach(function (form) {
    var input = form.elements.arxiv_id;
    var button = form.querySelector('button[type="submit"]');
    var status = form.querySelector('[role="status"]');
    var label = button.textContent;
    function reset() {
      form.removeAttribute('aria-busy');
      button.disabled = false;
      button.textContent = label;
      if (status) status.hidden = true;
    }
    form.addEventListener('submit', function (event) {
      if (!input.value.trim() || form.getAttribute('aria-busy') === 'true') {
        event.preventDefault();
        if (!input.value.trim()) input.focus();
        return;
      }
      form.setAttribute('aria-busy', 'true');
      button.textContent = button.dataset.loading;
      button.disabled = true;
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
