// The reading status buttons next to a paper (templatetags/reading_marks.py): saved without leaving the page.
(function () {
  if (window.readingMarkReady) return;
  window.readingMarkReady = true;
  var style = document.createElement('style');
  style.textContent = '.reading-mark{display:inline-flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0}' +
    '.reading-mark button{font:inherit;font-size:13px;padding:3px 10px;border:1px solid var(--border,#ccc);border-radius:999px;background:var(--surface,#fff);color:var(--text,#16161d);cursor:pointer}' +
    '.reading-mark button[aria-pressed=true]{background:var(--accent,#4f46e5);border-color:var(--accent,#4f46e5);color:#fff}' +
    '.reading-mark a{font-size:13px}';
  document.head.appendChild(style);
  document.addEventListener('submit', function (event) {
    var form = event.target;
    if (!form.matches || !form.matches('[data-reading-mark]') || !window.fetch) return;
    event.preventDefault();
    var data = new FormData(form);
    if (event.submitter) data.set('status', event.submitter.value);
    fetch(form.action, {method: 'POST', body: data, headers: {'X-Requested-With': 'fetch'}, credentials: 'same-origin'})
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r); })
      .then(function (answer) {
        form.querySelectorAll('button[data-status]').forEach(function (b) {
          var on = b.dataset.status === answer.reading;
          b.setAttribute('aria-pressed', on ? 'true' : 'false');
          b.value = on ? '' : b.dataset.status;
        });
      })
      .catch(function () {  // without the clicked button's value, the form would clear the status
        var input = document.createElement('input');
        input.type = 'hidden'; input.name = 'status'; input.value = data.get('status') || '';
        form.querySelectorAll('button[name=status]').forEach(function (b) { b.removeAttribute('name'); });
        form.appendChild(input);
        form.submit();
      });
  });
})();
