(function () {
  var form = document.getElementById('search-form');
  if (!form) return;
  form.addEventListener('submit', function (event) {
    if (!form.elements.arxiv_id.value.trim()) {
      event.preventDefault();
      form.elements.arxiv_id.focus();
      return;
    }
    form.setAttribute('aria-busy', 'true');
  });
  window.addEventListener('pageshow', function () { form.removeAttribute('aria-busy'); });
}());
