(function () {
  document.querySelectorAll('form[data-watch-activation]').forEach(function (form) {
    var button = form.querySelector('button[type="submit"]');
    var label = form.querySelector('[data-watch-label]');
    var spinner = form.querySelector('.watch-spinner');
    var status = form.querySelector('[data-watch-status]');
    var original = label.textContent;
    function reset() {
      form.removeAttribute('aria-busy');
      button.disabled = false;
      label.textContent = original;
      spinner.hidden = true;
      status.hidden = true;
    }
    form.addEventListener('submit', function (event) {
      if (form.getAttribute('aria-busy') === 'true') {
        event.preventDefault();
        return;
      }
      form.setAttribute('aria-busy', 'true');
      button.disabled = true;
      label.textContent = button.dataset.loading;
      spinner.hidden = false;
      status.hidden = false;
    });
    window.addEventListener('pageshow', reset);
  });
}());

(function () {
  // The preview reads OpenAlex, which may take seconds: the page asks for it once it has loaded.
  var box = document.querySelector('[data-preview-url]');
  if (!box || !window.fetch) {
    return;
  }
  fetch(box.getAttribute('data-preview-url'), {credentials: 'same-origin', headers: {'X-Requested-With': 'fetch'}})
    .then(function (response) {
      if (!response.ok) {
        throw new Error(response.status);
      }
      return response.text();
    })
    .then(function (html) {
      box.innerHTML = html;
    })
    .catch(function () {
      box.querySelector('[data-preview-status]').hidden = true;
      box.querySelector('[data-preview-failed]').hidden = false;
    });
}());
