// The paper's PDF beside the summary and the assistant, on wide screens: links marked data-pdf-panel open
// it in a panel fixed on the right, that the reader can resize; on narrow screens they open a new tab.
(function () {
  var script = document.currentScript;
  var label = function (name) { return script.getAttribute('data-' + name); };
  var wide = window.matchMedia('(min-width: 1100px)');
  var MIN = 320;
  var panel, frame;

  function stored(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }
  function store(key, value) {
    try { value === null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch (e) { /* private window */ }
  }

  function setWidth(px) {
    var max = Math.round(window.innerWidth * 0.7);
    var width = Math.max(MIN, Math.min(max, Math.round(px)));
    document.documentElement.style.setProperty('--pdf-width', width + 'px');
    return width;
  }

  function build() {
    panel = document.createElement('aside');
    panel.className = 'pdf-panel';
    panel.setAttribute('aria-label', label('title'));
    panel.innerHTML =
      '<div class="pdf-resize" role="separator" aria-orientation="vertical" tabindex="0"></div>' +
      '<div class="pdf-panel-head"><strong></strong>' +
      '<a target="_blank" rel="noopener"><i class="fas fa-external-link-alt"></i></a>' +
      '<button type="button"><i class="fas fa-times"></i></button></div>' +
      '<iframe></iframe>';
    panel.querySelector('strong').textContent = label('title');
    var tab = panel.querySelector('a');
    tab.title = label('new-tab');
    tab.setAttribute('aria-label', label('new-tab'));
    var close = panel.querySelector('button');
    close.title = label('close');
    close.setAttribute('aria-label', label('close'));
    close.addEventListener('click', hide);
    frame = panel.querySelector('iframe');
    frame.title = label('title');

    var handle = panel.querySelector('.pdf-resize');
    handle.setAttribute('aria-label', label('resize'));
    handle.addEventListener('pointerdown', function (event) {
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      panel.classList.add('resizing');  // the iframe would swallow the pointer otherwise
    });
    handle.addEventListener('pointermove', function (event) {
      if (panel.classList.contains('resizing')) setWidth(window.innerWidth - event.clientX);
    });
    handle.addEventListener('pointerup', function (event) {
      panel.classList.remove('resizing');
      handle.releasePointerCapture(event.pointerId);
      store('pdfPanelWidth', panel.offsetWidth);
    });
    handle.addEventListener('keydown', function (event) {
      var step = event.key === 'ArrowLeft' ? 40 : event.key === 'ArrowRight' ? -40 : 0;
      if (!step) return;
      event.preventDefault();
      store('pdfPanelWidth', setWidth(panel.offsetWidth + step));
    });
    document.body.appendChild(panel);
  }

  function show(url) {
    if (!panel) build();
    setWidth(Number(stored('pdfPanelWidth')) || window.innerWidth * 0.42);
    if (frame.getAttribute('src') !== url) frame.setAttribute('src', url);
    panel.querySelector('a').href = url;
    document.body.classList.add('pdf-open');
    store('pdfPanelOpen', '1');
  }

  function hide() {
    document.body.classList.remove('pdf-open');
    store('pdfPanelOpen', null);
  }

  document.addEventListener('click', function (event) {
    var link = event.target.closest('a[data-pdf-panel]');
    if (!link || !wide.matches || event.metaKey || event.ctrlKey || event.shiftKey) return;
    event.preventDefault();
    if (document.body.classList.contains('pdf-open') && frame.getAttribute('src') === link.href) hide();
    else show(link.href);
  });
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && document.body.classList.contains('pdf-open') && !document.querySelector('.nav-dropdown.open, .lang-switch.open, dialog[open]')) hide();
  });
  wide.addEventListener('change', function () {
    if (!wide.matches) document.body.classList.remove('pdf-open');
  });

  // left open on the last paper: open again on this one
  var first = document.querySelector('a[data-pdf-panel]');
  if (first && wide.matches && stored('pdfPanelOpen')) show(first.href);
})();
