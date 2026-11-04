// Tells the server a person is on this page (visits.seen), at their first scroll, touch, key or mouse move:
// the robots that open pages in a real browser never do, and are then left out of the staff activity page.
(function () {
  if (navigator.webdriver) return;
  var events = ['pointermove', 'pointerdown', 'scroll', 'keydown', 'touchstart', 'wheel'];
  var sent = false;
  function seen() {
    if (sent) return;
    sent = true;
    events.forEach(function (name) { window.removeEventListener(name, seen, true); });
    var body = new URLSearchParams({path: location.pathname});
    try {
      if (!(navigator.sendBeacon && navigator.sendBeacon('/visit/', body))) {
        fetch('/visit/', {method: 'POST', body: body, keepalive: true, credentials: 'same-origin'}).catch(function () {});
      }
    } catch (e) { /* never breaks the page */ }
  }
  events.forEach(function (name) { window.addEventListener(name, seen, {capture: true, passive: true}); });
})();
