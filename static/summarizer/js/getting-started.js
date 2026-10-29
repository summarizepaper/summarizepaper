/* A resumable walkthrough. Only fixed step/domain labels go to analytics. */
(function () {
  'use strict';
  var root = document.querySelector('.getting-started');
  if (!root) return;
  var panels = Array.from(root.querySelectorAll('[data-panel]'));
  if (!panels.length) return;
  var steps = ['summary', 'figures', 'question'];
  var domain = ['astronomy', 'biology', 'computing'].includes(root.dataset.domain) ? root.dataset.domain : 'astronomy';
  var navigation = root.querySelector('.getting-started-steps');
  function track(name, step) {
    try { if (window.umami) Promise.resolve(window.umami.track(name, {step:step, domain:domain})).catch(function () {}); } catch (_) {}
  }
  function show(step, focus) {
    if (!steps.includes(step)) step = 'summary';
    panels.forEach(function (panel) {panel.hidden = panel.dataset.panel !== step;});
    root.querySelectorAll('[data-step]').forEach(function (button) {
      if (button.dataset.step === step) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
    if (step === 'figures') root.querySelector('.chat-figures').open = true;
    if (focus) {
      root.querySelector('#step-' + step + ' h2').focus({preventScroll:true});
      navigation.scrollIntoView({block:'start', behavior:'auto'});
    }
    track('getting_started_step', step);
  }
  function go(step) {
    if (location.hash === '#' + step) show(step, true);
    else location.hash = step;
  }
  navigation.hidden = false;
  root.querySelector('.getting-started-suggestions').hidden = false;
  root.querySelectorAll('[data-next]').forEach(function (button) {button.hidden = false; button.addEventListener('click', function () {go(button.dataset.next);});});
  root.querySelectorAll('[data-step]').forEach(function (button) {button.addEventListener('click', function () {go(button.dataset.step);});});
  root.querySelectorAll('.getting-started-suggestions [data-question]').forEach(function (button) {
    button.addEventListener('click', function () {var input = root.querySelector('textarea'); input.value = button.dataset.question; input.focus();});
  });
  var sourceOpened = false;
  root.querySelectorAll('.getting-started-evidence').forEach(function (evidence) {
    evidence.addEventListener('toggle', function () {
      if (evidence.open && !sourceOpened) {
        sourceOpened = true;
        track('getting_started_source_open', 'summary');
      }
    });
  });
  window.addEventListener('hashchange', function () {show(location.hash.slice(1), true);});
  // A return from the chat or a shared step link should also open at the actual step.
  show(location.hash.slice(1), steps.includes(location.hash.slice(1)));
})();
