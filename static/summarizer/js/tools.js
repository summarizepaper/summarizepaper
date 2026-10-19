/* Helpers shared by the paper tools: POST a form with the CSRF token, and wait for a
   result written in the background (see generated.py). */
(function () {
  var lang = document.documentElement.lang || 'en';

  function csrf() {
    var match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
    if (match) return decodeURIComponent(match[1]);
    var input = document.querySelector('input[name=csrfmiddlewaretoken]');
    return input ? input.value : '';
  }

  function post(url, data) {
    var body = new URLSearchParams();
    Object.keys(data || {}).forEach(function (k) { body.append(k, data[k]); });
    return fetch(url, {
      method: 'POST', body: body, credentials: 'same-origin',
      headers: {'X-CSRFToken': csrf(), 'X-Requested-With': 'fetch'}
    }).then(function (r) {
      return r.json().catch(function () { return {status: 'error'}; }).then(function (json) {
        json.httpStatus = r.status;
        return json;
      });
    });
  }

  // Calls done(data) once the result is ready, failed() if it fails or takes more than 10 minutes.
  function wait(id, done, failed) {
    var started = Date.now();
    function tick() {
      fetch('/' + lang + '/ai/' + id + '/', {credentials: 'same-origin'})
        .then(function (r) { return r.ok ? r.json() : {status: 'failed'}; })
        .then(function (json) {
          if (json.status === 'done') return done(json.data);
          if (json.status === 'failed' || Date.now() - started > 600000) return failed && failed();
          setTimeout(tick, 3000);
        })
        .catch(function () { setTimeout(tick, 5000); });
    }
    tick();
  }

  function text(tag, content, className) {
    var el = document.createElement(tag);
    if (content) el.textContent = content;
    if (className) el.className = className;
    return el;
  }

  window.SP = {lang: lang, post: post, wait: wait, text: text};
})();
