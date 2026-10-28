(function () {
  'use strict';
  var dialog = document.getElementById('guide-image-dialog');
  if (!dialog || typeof dialog.showModal !== 'function') return;
  var image = dialog.querySelector('img'), caption = dialog.querySelector('figcaption'), opener;
  function close() {if (dialog.open) dialog.close();}
  document.addEventListener('click', function (event) {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    var link = event.target.closest('.guide-screenshot a');
    if (!link) return;
    var thumbnail = link.querySelector('img');
    if (!thumbnail) return;
    event.preventDefault();
    opener = link;
    image.src = thumbnail.currentSrc || thumbnail.src;
    image.alt = thumbnail.alt;
    var label = link.closest('figure').querySelector('figcaption');
    caption.textContent = label ? label.textContent : thumbnail.alt;
    dialog.showModal();
  });
  dialog.querySelector('.guide-image-close').addEventListener('click', close);
  dialog.addEventListener('cancel', function (event) {event.preventDefault(); close();});
  dialog.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {event.preventDefault(); event.stopPropagation(); close();}
  });
  dialog.addEventListener('click', function (event) {
    if (event.target !== dialog) return;
    var rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) close();
  });
  dialog.addEventListener('close', function () {if (opener && opener.isConnected) opener.focus();});
})();
