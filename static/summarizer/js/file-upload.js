/* Progressive enhancement of native multipart inputs. A drop never submits a form. */
(function () {
  'use strict';
  function hasFiles(event) {
    return event.dataTransfer && Array.from(event.dataTransfer.types || []).indexOf('Files') >= 0;
  }
  function mount(box) {
    if (box.dataset.uploadMounted) return;
    box.dataset.uploadMounted = '1';
    var input = box.querySelector('input[type="file"]'), zone = box.querySelector('[data-file-drop]');
    var status = box.querySelector('[data-file-status]'), error = box.querySelector('[data-file-error]');
    var clear = box.querySelector('[data-file-clear]'), saved = input.files, depth = 0;
    var maximum = Number(box.dataset.maxBytes);
    function size(bytes) {
      var unit = bytes >= 1048576 ? 1048576 : bytes >= 1024 ? 1024 : 1;
      return new Intl.NumberFormat(document.documentElement.lang || 'en', {maximumFractionDigits: 1}).format(bytes / unit) + ' ' +
        (unit === 1048576 ? box.dataset.mb : unit === 1024 ? box.dataset.kb : box.dataset.bytes);
    }
    function render() {
      var file = input.files && input.files[0];
      status.hidden = clear.hidden = !file;
      status.textContent = file ? box.dataset.selected.replace('{name}', file.name).replace('{size}', size(file.size)) : '';
      if (file) clear.setAttribute('aria-label', box.dataset.remove.replace('{name}', file.name));
      box.classList.toggle('has-file', Boolean(file));
    }
    function message(value) {error.textContent = value || ''; error.hidden = !value;}
    function validate(files) {
      if (files.length !== 1) return box.dataset.errorCount;
      var file = files[0], name = file.name.toLowerCase(), type = (file.type || '').toLowerCase();
      var required = box.dataset.requiredExtension;
      var accepted = (input.accept || '').toLowerCase().split(',').map(function (s) {return s.trim();}).filter(Boolean);
      var matches = !accepted.length || accepted.some(function (rule) {
        if (rule[0] === '.') return name.endsWith(rule);
        if (rule.endsWith('/*')) return type.startsWith(rule.slice(0, -1));
        return type === rule;
      });
      if (!matches || (required && !name.endsWith(required))) return box.dataset.errorType.replace('{formats}', required || input.accept);
      if (maximum > 0 && file.size > maximum) return box.dataset.errorSize.replace('{max}', box.dataset.maxMb);
      return '';
    }
    function resetDrag() {depth = 0; zone.classList.remove('is-dragover');}
    input.addEventListener('change', function () {
      if (!input.files.length) {saved = input.files; render(); message(''); return;}
      var problem = validate(input.files);
      if (problem) {
        // A mistaken replacement must not discard the reader's previous valid file.
        try {input.files = saved;} catch (_) {input.value = '';}
        render(); message(problem); return;
      }
      saved = input.files; render(); message('');
    });
    zone.addEventListener('dragenter', function (event) {
      if (!hasFiles(event)) return;
      event.preventDefault(); depth++; zone.classList.add('is-dragover');
    });
    zone.addEventListener('dragover', function (event) {
      if (!hasFiles(event)) return;
      event.preventDefault(); event.dataTransfer.dropEffect = 'copy';
    });
    zone.addEventListener('dragleave', function (event) {
      if (!hasFiles(event)) return;
      depth = Math.max(0, depth - 1); if (!depth) zone.classList.remove('is-dragover');
    });
    zone.addEventListener('drop', function (event) {
      if (!hasFiles(event)) return;
      event.preventDefault(); resetDrag();
      var files = event.dataTransfer.files || [];
      var problem = validate(files);
      if (problem) {message(problem); return;}
      try {input.files = files;} catch (_) {message(box.dataset.errorBrowser); return;}
      input.dispatchEvent(new Event('change', {bubbles: true}));
    });
    clear.addEventListener('click', function () {
      input.value = ''; input.dispatchEvent(new Event('change', {bubbles: true})); input.focus();
    });
    if (input.form) input.form.addEventListener('reset', function () {
      setTimeout(function () {saved = input.files; resetDrag(); render(); message('');}, 0);
    });
    document.addEventListener('dragend', resetDrag);
    render();
  }
  function init() {
    var boxes = document.querySelectorAll('[data-file-upload]');
    if (!boxes.length) return;
    boxes.forEach(mount);
    // Do not let an accidental drop elsewhere navigate away and lose the draft.
    ['dragover', 'drop'].forEach(function (type) {
      document.addEventListener(type, function (event) {
        if (!hasFiles(event)) return;
        event.preventDefault();
        if (type === 'dragover' && !(event.target.closest && event.target.closest('[data-file-drop]'))) event.dataTransfer.dropEffect = 'none';
      });
    });
  }
  document.addEventListener('DOMContentLoaded', init);
})();
