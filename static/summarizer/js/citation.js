/* Delegated clicks also cover links inserted after a live synthesis. */
(function () {
  'use strict';
  function init() {
    var dialog = document.getElementById('citation-dialog');
    if (!dialog || dialog.dataset.mounted || typeof dialog.showModal !== 'function') return;
    dialog.dataset.mounted = '1';
    var text = dialog.querySelector('[data-citation-text]'), status = dialog.querySelector('[data-citation-status]');
    var copy = dialog.querySelector('[data-citation-copy]'), download = dialog.querySelector('[data-citation-download]');
    var retry = dialog.querySelector('[data-citation-retry]'), opener, url, version = 0;
    function close() {dialog.close();}
    function load() {
      var request = ++version;
      text.value = ''; text.hidden = true; copy.disabled = true; retry.hidden = true;
      status.textContent = dialog.dataset.loading;
      fetch(url, {headers: {'Accept': 'text/plain'}}).then(function (response) {
        if (!response.ok || !(response.headers.get('Content-Type') || '').startsWith('text/plain')) throw Error('citation unavailable');
        return response.text();
      }).then(function (value) {
        if (request !== version || !dialog.open) return;
        if (!value.trim()) throw Error('empty citation');
        text.value = value; text.hidden = false; copy.disabled = false; status.textContent = '';
      }).catch(function () {
        if (request !== version || !dialog.open) return;
        status.textContent = dialog.dataset.failed; retry.hidden = false;
      });
    }
    document.addEventListener('click', function (event) {
      var link = event.target.closest && event.target.closest('a[data-citation]');
      if (!link || event.defaultPrevented || event.button > 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault(); opener = link; url = link.href;
      download.href = url + (url.indexOf('?') < 0 ? '?' : '&') + 'download=1';
      dialog.showModal(); load();
    });
    dialog.querySelector('[data-citation-close]').addEventListener('click', close);
    dialog.addEventListener('click', function (event) {if (event.target === dialog) {
      var rect = dialog.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) close();
    }});
    dialog.addEventListener('close', function () {version++; if (opener && opener.isConnected) opener.focus();});
    retry.addEventListener('click', load);
    copy.addEventListener('click', function () {
      var request = version, value = text.value;
      if (!value || copy.disabled) return;
      Promise.resolve().then(function () {return navigator.clipboard.writeText(value);}).then(function () {
        if (request === version && dialog.open) status.textContent = dialog.dataset.copied;
      }).catch(function () {
        if (request !== version || !dialog.open) return;
        text.focus(); text.select(); status.textContent = dialog.dataset.copyFailed;
      });
    });
  }
  init();
  document.addEventListener('DOMContentLoaded', init);
})();
