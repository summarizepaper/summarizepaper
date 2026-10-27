(function () {
  'use strict';
  function init() {
    document.querySelectorAll('[data-student-shortlist]').forEach(function (form) {
      if (form.dataset.mounted) return;
      form.dataset.mounted = '1';
      var status = form.querySelector('[data-student-save-status]');
      var button = form.querySelector('[data-student-save]');
      var fields = ['status', 'notes', 'remind_on', 'email_reminder', 'draft'].filter(function (name) {return form.elements[name];});
      var saved = snapshot(), timer, running = false, blocked = false, removing = false, reminderSet = false;
      function value(name) {var field = form.elements[name]; return field.type === 'checkbox' ? field.checked : field.value;}
      function snapshot() {return JSON.stringify(fields.map(value));}
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
          notice(dirty() ? form.dataset.unsaved : (reminderSet ? form.dataset.reminderSet + ' ' : '') + form.dataset.saved, dirty());
          reminderSet = false;
          if (dirty()) timer = setTimeout(save, 0);
        } catch (error) {notice(form.dataset.error, true);}
        finally {running = false; button.disabled = false;}
      }
      function changed(event) {
        clearTimeout(timer);
        if (blocked) return;
        // A first contact or a follow-up gets a reminder ten days later, unless one is set.
        var target = event.target;
        if (target && target.name === 'status' && (target.value === 'contacted' || target.value === 'followed_up')
            && form.elements.remind_on && !form.elements.remind_on.value) {
          var day = new Date();
          day.setDate(day.getDate() + 10);
          form.elements.remind_on.value = day.getFullYear() + '-' + String(day.getMonth() + 1).padStart(2, '0') + '-' + String(day.getDate()).padStart(2, '0');
          reminderSet = true;
        }
        notice(dirty() ? form.dataset.unsaved : form.dataset.saved, dirty());
        if (dirty()) timer = setTimeout(save, event.type === 'change' ? 0 : 700);
      }
      form.addEventListener('change', changed);
      form.addEventListener('input', changed);
      // The draft lives in the card, outside the form it belongs to.
      fields.forEach(function (name) {
        var field = form.elements[name];
        if (form.contains && field.addEventListener && !form.contains(field)) {
          field.addEventListener('input', changed);
          field.addEventListener('change', changed);
        }
      });
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
        if (typeof Event === 'function' && draft.dispatchEvent) draft.dispatchEvent(new Event('input', {bubbles: true}));
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
  function filters() {
    document.querySelectorAll('[data-student-filters]').forEach(function (box) {
      var inputs = Array.prototype.slice.call(box.querySelectorAll('input[data-filter]'));
      function apply() {
        var wanted = inputs.filter(function (input) {return input.checked;}).map(function (input) {return input.dataset.filter;});
        document.querySelectorAll('[data-overview-row]').forEach(function (row) {
          var shown = wanted.every(function (name) {return row.dataset[name] === '1';});
          row.hidden = !shown;
          var card = document.getElementById(row.dataset.card);
          if (card) card.hidden = !shown;
        });
        document.querySelectorAll('[data-overview-team]').forEach(function (row) {row.hidden = wanted.length > 0;});
      }
      inputs.forEach(function (input) {input.addEventListener('change', apply);});
    });
  }
  function openTarget() {
    var target = location.hash && document.getElementById(location.hash.slice(1));
    var lead = target && target.closest('details[data-student-lead]');
    if (lead) lead.open = true;
  }
  function start() {
    init();
    if (typeof location === 'undefined' || !document.getElementById) return;
    filters();
    openTarget();
    window.addEventListener('hashchange', openTarget);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
