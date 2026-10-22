document.addEventListener('DOMContentLoaded', function () {
  var form = document.getElementById('deck-language-form');
  if (!form) return;
  var select = document.getElementById('deck-language');
  var button = document.getElementById('deck-go');
  var panel = document.getElementById('deck-working');
  var error = document.getElementById('deck-language-error');
  var start = document.getElementById('deck-start');
  var busy = false;
  function pending(value) {
    busy = value; button.disabled = value; select.disabled = value;
    form.setAttribute('aria-busy', String(value));
  }
  function failed() {
    pending(false); error.textContent = form.dataset.error; error.hidden = false;
  }
  function open(url) { pending(false); window.location.assign(url); }
  function follow(id, url) {
    pending(true);
    if (start) start.hidden = true;
    SP.track(panel, id, function () { open(url); }, failed);
  }
  form.addEventListener('submit', function (event) {
    event.preventDefault();
    if (busy) return;
    var language = select.value;
    pending(true); error.hidden = true; panel.dataset.started = '';
    SP.post(form.dataset.startUrl, {deck_lang: language}).then(function (json) {
      if (json.status === 'done' && json.url) return open(json.url);
      if (json.id && json.url && json.status === 'pending') return follow(json.id, json.url);
      failed();
    }).catch(failed);
  });
  if (form.dataset.pendingId) follow(Number(form.dataset.pendingId), form.dataset.returnUrl);
});
