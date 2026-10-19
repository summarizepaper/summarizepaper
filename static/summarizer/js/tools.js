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

  // A waiting panel (templates/summarizer/working.html): spinner, the work's real steps, elapsed time.
  // Shows the panel and follows result `id`; calls done(data) or shows the failure message.
  function track(panel, id, done) {
    panel.hidden = false;
    panel.classList.remove('is-failed');
    var tracking = Date.now();
    // the work may have started before this page was opened
    var started = panel.dataset.started ? Math.min(Date.parse(panel.dataset.started), tracking) : tracking;
    var elapsed = panel.querySelector('.working-elapsed');
    var steps = Array.prototype.slice.call(panel.querySelectorAll('[data-step]'));
    var timer = setInterval(function () {
      var s = Math.floor((Date.now() - started) / 1000);
      if (elapsed) elapsed.textContent = Math.floor(s / 60) + ':' + ('0' + s % 60).slice(-2);
    }, 1000);
    function mark(step) {
      var index = steps.findIndex(function (li) { return li.dataset.step === step; });
      if (index < 0) return;
      steps.forEach(function (li, i) {
        li.classList.toggle('is-done', i < index);
        li.classList.toggle('is-current', i === index);
      });
    }
    if (steps.length) mark(steps[0].dataset.step);
    function tick() {
      fetch('/' + lang + '/ai/' + id + '/', {credentials: 'same-origin'})
        .then(function (r) { return r.ok ? r.json() : {status: 'failed'}; })
        .then(function (json) {
          if (json.step) mark(json.step);
          if (json.status === 'done') {
            clearInterval(timer);
            steps.forEach(function (li) { li.classList.remove('is-current'); li.classList.add('is-done'); });
            return done(json.data);
          }
          if (json.status === 'failed' || Date.now() - tracking > 600000) {
            clearInterval(timer);
            panel.classList.add('is-failed');
            return;
          }
          setTimeout(tick, 2500);
        })
        .catch(function () { setTimeout(tick, 5000); });
    }
    tick();
  }

  window.SP = {lang: lang, post: post, wait: wait, text: text, track: track};
})();
