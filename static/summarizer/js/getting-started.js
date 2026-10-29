/* A resumable walkthrough. Only fixed step labels go to analytics, never questions or account data. */
(function () {
  'use strict';
  var root = document.querySelector('.getting-started');
  if (!root) return;
  var panels = Array.from(root.querySelectorAll('[data-panel]'));
  if (!panels.length) return;
  var steps = ['summary', 'figures', 'question'];
  function track(step) {
    try { if (window.umami) Promise.resolve(window.umami.track('getting_started_step', {step:step})).catch(function () {}); } catch (_) {}
  }
  function show(step, focus) {
    if (!steps.includes(step)) step = 'summary';
    panels.forEach(function (panel) {panel.hidden = panel.dataset.panel !== step;});
    root.querySelectorAll('[data-step]').forEach(function (button) {
      if (button.dataset.step === step) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
    if (step === 'figures') root.querySelector('.chat-figures').open = true;
    if (focus) root.querySelector('#step-' + step + ' h2').focus();
    track(step);
  }
  function go(step) {
    if (location.hash === '#' + step) show(step, true);
    else location.hash = step;
  }
  root.querySelector('.getting-started-steps').hidden = false;
  root.querySelector('.getting-started-suggestions').hidden = false;
  root.querySelectorAll('[data-next]').forEach(function (button) {button.hidden = false; button.addEventListener('click', function () {go(button.dataset.next);});});
  root.querySelectorAll('[data-step]').forEach(function (button) {button.addEventListener('click', function () {go(button.dataset.step);});});
  root.querySelectorAll('[data-question]').forEach(function (button) {
    button.addEventListener('click', function () {var input = root.querySelector('textarea'); input.value = button.dataset.question; input.focus();});
  });
  window.addEventListener('hashchange', function () {show(location.hash.slice(1), true);});
  show(location.hash.slice(1), false);
})();
