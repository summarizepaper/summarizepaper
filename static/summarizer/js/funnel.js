/* Only fixed event names and plan periods: never send form values or account IDs. */
(function () {
  function track(name, data) {
    try {
      if (window.umami) Promise.resolve(window.umami.track(name, data)).catch(function () {});
    } catch (_) { /* Analytics must never prevent navigation or payment. */ }
  }
  document.addEventListener('click', function (event) {
    var link = event.target.closest('a[data-funnel]');
    if (link) track(link.dataset.funnel);
  });
  document.addEventListener('submit', function (event) {
    var form = event.target;
    if (!form.dataset.funnel) return;
    var period = event.submitter && event.submitter.value;
    track(form.dataset.funnel, period === 'monthly' || period === 'yearly' ? {period: period} : {});
  });
  document.querySelectorAll('[data-funnel-view]').forEach(function (element) {
    track(element.dataset.funnelView);
  });
  var path = window.location.pathname;
  if (/\/premium\/$/.test(path)) {
    track('premium_view');
    var params = new URLSearchParams(window.location.search);
    if (params.get('success') === '1') track('checkout_return');
    if (params.get('cancelled') === '1') track('checkout_cancel');
  }
  if (/\/register\/$/.test(path)) track('signup_view');
  if (/\/account_activation_sent\/$/.test(path)) track('activation_email_pending');
}());
