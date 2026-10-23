(function () {
  'use strict';
  function init() {
    document.querySelectorAll('[data-student-shortlist]').forEach(function (form) {
      if (form.dataset.mounted) return;
      form.dataset.mounted = '1';
      var status = form.querySelector('[data-student-save-status]');
      var button = form.querySelector('[data-student-save]');
      var fields = ['status', 'notes', 'remind_on'];
      var saved = snapshot(), timer, running = false, blocked = false, removing = false;
      function snapshot() {return JSON.stringify(fields.map(function (name) {return form.elements[name].value;}));}
      function dirty() {return snapshot() !== saved;}
      function notice(message, warning) {status.textContent = message; status.classList.toggle('notice', !!warning);}
      async function save() {
        clearTimeout(timer);
        if (running || blocked || removing || !dirty() || !form.reportValidity()) return;
        running = true;
        var sent = snapshot();
        var body = new FormData(form);
        notice(form.dataset.saving, false);
        try {
          var response = await fetch(form.getAttribute('action'), {method:'POST', body:body, headers:{Accept:'application/json'}, credentials:'same-origin'});
          if (response.status === 409) {
            blocked = true;
            notice(form.dataset.conflict, true);
            return;
          }
          if (!response.ok || response.redirected) throw new Error('save');
          var result = await response.json();
          if (!Number.isInteger(result.revision)) throw new Error('revision');
          form.elements.revision.value = result.revision;
          saved = sent;
          notice(dirty() ? form.dataset.unsaved : form.dataset.saved, dirty());
          if (dirty()) timer = setTimeout(save, 0);
        } catch (error) {notice(form.dataset.error, true);}
        finally {running = false; button.disabled = false;}
      }
      function changed(event) {
        clearTimeout(timer);
        if (blocked) return;
        notice(dirty() ? form.dataset.unsaved : form.dataset.saved, dirty());
        if (dirty()) timer = setTimeout(save, event.type === 'change' ? 0 : 700);
      }
      form.addEventListener('change', changed);
      form.addEventListener('input', changed);
      form.addEventListener('submit', function (event) {
        clearTimeout(timer);
        if (event.submitter && event.submitter.value === 'remove') {removing = true; return;}
        event.preventDefault(); save();
      });
      window.addEventListener('beforeunload', function (event) {
        if (dirty() && !removing) {event.preventDefault(); event.returnValue = '';}
      });
      window.addEventListener('pageshow', function () {removing = false; button.disabled = false;});
    });
    document.querySelectorAll('[data-student-question]').forEach(function (button) {
      button.addEventListener('click', function () {
        var card = button.closest('[data-student-card]');
        var draft = card.querySelector('[data-student-draft]');
        var message = card.querySelector('[data-student-draft-status]');
        var snippet = button.dataset.studentQuestion;
        if (!draft.value.includes(snippet) && !(button.dataset.studentQuestionText && draft.value.includes(button.dataset.studentQuestionText))) {
          var placeholder = draft.dataset.questionPlaceholder;
          var closing = draft.dataset.questionClosing;
          if (placeholder && draft.value.includes(placeholder)) draft.value = draft.value.replace(placeholder, function () {return button.dataset.studentPaperTitle && draft.value.includes(button.dataset.studentPaperTitle) ? button.dataset.studentQuestionText : snippet;});
          else if (closing && draft.value.includes(closing)) draft.value = draft.value.replace(closing, function () {return snippet + '\n\n' + closing;});
          else draft.value += '\n\n' + snippet;
        }
        draft.closest('details').open = true;
        message.textContent = message.dataset.added;
        draft.focus();
      });
    });
    document.querySelectorAll('[data-student-source]').forEach(function (form) {
      form.addEventListener('submit', function () {
        form.querySelector('button').disabled = true;
        form.querySelector('[role=status]').textContent = form.dataset.saving;
      });
      window.addEventListener('pageshow', function () {form.querySelector('button').disabled = false;});
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
