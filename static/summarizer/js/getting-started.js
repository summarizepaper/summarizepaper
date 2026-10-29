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
  function track(name, step, format) {
    var data = {step:step, domain:domain};
    if (['summary', 'plain', 'points', 'blog', 'abstract'].includes(format)) data.format = format;
    try { if (window.umami) Promise.resolve(window.umami.track(name, data)).catch(function () {}); } catch (_) {}
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
  root.querySelectorAll('[data-reading-widget]').forEach(function (widget) {
    var tabs = Array.from(widget.querySelectorAll('[data-reading]'));
    var contents = Array.from(widget.querySelectorAll('[data-reading-panel]'));
    if (!tabs.length) return;
    widget.querySelector('.getting-started-reading-tabs').hidden = false;
    function select(tab, focus, measured) {
      tabs.forEach(function (button) {
        var active = button === tab;
        button.setAttribute('aria-selected', active ? 'true' : 'false');
        button.setAttribute('tabindex', active ? '0' : '-1');
      });
      contents.forEach(function (panel) {
        panel.hidden = panel.dataset.readingPanel !== tab.dataset.reading;
        if (!panel.hidden && window.SPChat) window.SPChat.typeset(panel);
      });
      if (focus) tab.focus();
      if (measured) track('getting_started_reading_format', 'summary', tab.dataset.reading);
    }
    tabs.forEach(function (tab, index) {
      tab.addEventListener('click', function () {select(tab, false, true);});
      tab.addEventListener('keydown', function (event) {
        var target;
        if (event.key === 'ArrowRight') target = tabs[(index + 1) % tabs.length];
        else if (event.key === 'ArrowLeft') target = tabs[(index + tabs.length - 1) % tabs.length];
        else if (event.key === 'Home') target = tabs[0];
        else if (event.key === 'End') target = tabs[tabs.length - 1];
        if (target) {event.preventDefault(); select(target, true, true);}
      });
    });
    select(tabs[0], false, false);
  });
  window.addEventListener('hashchange', function () {show(location.hash.slice(1), true);});
  // A return from the chat or a shared step link should also open at the actual step.
  show(location.hash.slice(1), steps.includes(location.hash.slice(1)));
})();
