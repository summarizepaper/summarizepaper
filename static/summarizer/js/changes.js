/* Keep the chosen pair, its explanatory text and its saved result in sync. */
(function () {
  function init() {
    var form = document.getElementById('changes-versions');
    if (!form) return;
    var from = form.querySelector('[name="from"]');
    var to = form.querySelector('[name="to"]');
    form.addEventListener('change', function (event) {
      if (event.target !== from && event.target !== to) return;
      if (Number(from.value) >= Number(to.value)) {
        if (event.target === from) to.value = String(Number(from.value) + 1);
        else from.value = String(Number(to.value) - 1);
      }
      form.requestSubmit();
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}());
