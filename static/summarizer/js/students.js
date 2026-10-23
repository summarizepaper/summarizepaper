(function () {
  'use strict';
  function init() {
    document.querySelectorAll('[data-student-shortlist]').forEach(function (form) {
      if (form.dataset.mounted) return;
      form.dataset.mounted = '1';
      var status = form.querySelector('[data-student-save-status]');
      var button = form.querySelector('[data-student-save]');
      var initial = [form.elements.status.value, form.elements.notes.value];
      var original = status.textContent;
      function changed() {
        var dirty = form.elements.status.value !== initial[0] || form.elements.notes.value !== initial[1];
        status.textContent = dirty ? form.dataset.unsaved : original;
        status.classList.toggle('notice', dirty);
      }
      form.addEventListener('change', changed);
      form.addEventListener('input', changed);
      form.addEventListener('submit', function (event) {
        if (event.submitter && event.submitter.value === 'remove') return;
        button.disabled = true;
        status.textContent = form.dataset.saving;
      });
      window.addEventListener('pageshow', function () {button.disabled = false; changed();});
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
