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
