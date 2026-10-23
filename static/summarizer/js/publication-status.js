/* Publication metadata is checked in a background job; no model call or kit regeneration. */
(function () {
  'use strict';
  document.addEventListener('DOMContentLoaded', function () {
    var box = document.getElementById('publication-status');
    if (!box) return;
    var busy = false;
    function state(working, error) {
      box.querySelector('[data-publication-refresh]').disabled = working;
      box.querySelector('[data-publication-progress]').hidden = !working;
      box.querySelector('[data-publication-error]').hidden = !error;
    }
    function failed() {busy = false; state(false, true);}
    function result() {
      fetch(box.dataset.result, {cache: 'no-store', credentials: 'same-origin'})
        .then(function (r) {if (!r.ok) throw r; return r.json();})
        .then(function (data) {
          var parsed = new DOMParser().parseFromString(data.html, 'text/html');
          parsed.querySelectorAll('script').forEach(function (script) {script.remove();});
          if (window.MathJax && MathJax.typesetClear) MathJax.typesetClear([box]);
          box.replaceChildren.apply(box, Array.from(parsed.body.childNodes).map(function (node) {return document.importNode(node, true);}));
          var questions = document.getElementById('author-questions');
          if (questions && Array.isArray(data.questions)) {
            if (window.MathJax && MathJax.typesetClear) MathJax.typesetClear([questions]);
            questions.replaceChildren();
            data.questions.forEach(function (question) {var li = document.createElement('li'); li.textContent = question; questions.appendChild(li);});
          }
          busy = false; state(false, false);
          if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise(questions ? [box, questions] : [box]).catch(function () {});
        }).catch(failed);
    }
    function check() {
      if (busy) return;
      busy = true; state(true, false);
      SP.post(box.dataset.start, {}).then(function (data) {
        if (!data.id) return failed();
        SP.wait(data.id, result, failed);
      }).catch(failed);
    }
    box.addEventListener('click', function (event) {if (event.target.closest('[data-publication-refresh]')) check();});
    if (box.dataset.auto === '1') check();
  });
})();
