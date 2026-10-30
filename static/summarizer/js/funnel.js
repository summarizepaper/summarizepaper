/* Only fixed UI labels and plan periods: never send form values or account IDs. */
(function () {
  function track(name, data) {
    try {
      if (window.umami) Promise.resolve(window.umami.track(name, data)).catch(function () {});
    } catch (_) { /* Analytics must never prevent navigation or payment. */ }
  }
  function context(element) {
    var data = {};
    if (['hero', 'needs', 'examples', 'offer', 'recent', 'free', 'preview'].includes(element.dataset.placement)) data.placement = element.dataset.placement;
    if (['paper', 'digest', 'refcheck', 'missed', 'scoops', 'literature', 'chat', 'letters', 'trends', 'saved_searches'].includes(element.dataset.tool)) data.tool = element.dataset.tool;
    if (['astronomy', 'biology', 'computing'].includes(element.dataset.domain)) data.domain = element.dataset.domain;
    return data;
  }
  document.addEventListener('click', function (event) {
    var link = event.target.closest('a[data-funnel]');
    if (link) track(link.dataset.funnel, context(link));
  });
  document.addEventListener('submit', function (event) {
    var form = event.target;
    if (!form.dataset.funnel || event.defaultPrevented) return;
    if (form.dataset.funnel === 'home_search_submit' && !form.elements.arxiv_id.value.trim()) return;
    var period = event.submitter && event.submitter.value;
    var data = period === 'monthly' || period === 'yearly' ? {period: period} : {};
    if (form.dataset.funnel === 'checkout_start' && ['eur', 'usd'].includes(form.dataset.currency)) data.currency = form.dataset.currency;
    track(form.dataset.funnel, data);
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
