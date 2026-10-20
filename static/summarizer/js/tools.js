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

  // Paper search next to a field that takes paper ids (templates/summarizer/arxiv_picker.html):
  // typing searches arXiv, or the open-access journals and preprints, through our relay (views.arxiv_api);
  // clicking a result adds its id to the field: an arXiv id, or a key such as doi_10.1093--mnras--stad1234.
  var ARXIV_ID = /^(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?\/\d{7})(v\d+)?$/i;
  var PAPER_KEY = /^(doi|hal|pmc)_[a-z0-9._-]+$/i;
  var LINK = /^(10\.\d{4,9}\/|https?:\/\/|doi:|pmc\d|pmid)/i;

  function picker(box) {
    var field = document.getElementById(box.dataset.target);
    var input = box.querySelector('.arxiv-picker-input');
    var status = box.querySelector('.arxiv-picker-status');
    var list = box.querySelector('.arxiv-picker-results');
    var source = box.querySelector('.arxiv-picker-source');
    var comma = box.dataset.separator === 'comma';
    var timer = null, seq = 0;
    if (!field || !input) return;

    function ids() {
      return field.value.split(/[\s,]+/).map(function (s) {
        if (PAPER_KEY.test(s)) return s.toLowerCase();
        var m = s.match(/(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?\/\d{7})/i);
        return m ? m[1].replace(/v\d+$/, '') : '';
      }).filter(Boolean);
    }

    function add(id) {
      if (ids().indexOf(id) >= 0) return;
      var value = field.value.replace(/[\s,]+$/, '');
      field.value = value ? value + (comma ? ', ' : '\n') + id : id;
      field.dispatchEvent(new Event('input', {bubbles: true}));
    }

    function say(message) {
      status.textContent = message || '';
      status.hidden = !message;
    }

    // arXiv's own syntax (au:, ti:, AND...) is passed as is; plain words must all appear.
    function query(q) {
      if (ARXIV_ID.test(q) || PAPER_KEY.test(q) || LINK.test(q)) return {id_list: q};
      if (source && source.value === 'open') return {search_query: q, src: 'open'};
      if (/\b(all|ti|au|abs|cat|id):/.test(q)) return {search_query: q};
      return {search_query: q.split(/\s+/).map(function (w) { return 'all:' + w.replace(/[():"]/g, ''); })
        .filter(function (w) { return w !== 'all:'; }).join(' AND ')};
    }

    function show(xml) {
      list.innerHTML = '';
      var entries = Array.prototype.slice.call(xml.getElementsByTagName('entry')).filter(function (e) {
        return e.getElementsByTagName('title').length && e.getElementsByTagName('id')[0].textContent.indexOf('/abs/') > 0;
      });
      if (!entries.length) { list.hidden = true; return say(box.dataset.none); }
      say('');
      entries.forEach(function (entry) {
        var id = entry.getElementsByTagName('id')[0].textContent.split('/abs/')[1];
        if (!PAPER_KEY.test(id)) id = id.replace(/v\d+$/, '');
        var label = entry.getElementsByTagName('label')[0];
        var authors = Array.prototype.map.call(entry.getElementsByTagName('author'), function (a) {
          return a.getElementsByTagName('name')[0].textContent;
        });
        var published = entry.getElementsByTagName('published')[0];
        var li = text('li');
        var info = text('div', '', 'arxiv-picker-info');
        info.appendChild(text('strong', entry.getElementsByTagName('title')[0].textContent.replace(/\s+/g, ' ').trim()));
        info.appendChild(text('div', [
          authors.slice(0, 3).join(', ') + (authors.length > 3 ? ' et al.' : ''),
          published ? published.textContent.slice(0, 4) : '',
          label ? label.textContent : 'arXiv:' + id
        ].filter(Boolean).join(' · '), 'meta'));
        var button = text('button', '', 'btn btn-sm btn-ghost');
        button.type = 'button';
        function mark() {
          var added = ids().indexOf(id) >= 0;
          button.textContent = added ? '✓ ' + box.dataset.added : '+ ' + box.dataset.add;
          button.disabled = added;
        }
        mark();
        button.addEventListener('click', function () { add(id); mark(); });
        li.appendChild(info);
        li.appendChild(button);
        list.appendChild(li);
      });
      list.hidden = false;
    }

    function run() {
      var q = input.value.trim();
      var mine = ++seq;
      if (q.length < 3) { list.hidden = true; return say(''); }
      var params = new URLSearchParams(query(q));
      params.append('max_results', '8');
      box.classList.add('is-loading');
      fetch(box.dataset.url + '?' + params, {credentials: 'same-origin'})
        .then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); })
        .then(function (body) {
          if (mine !== seq) return;
          show(new DOMParser().parseFromString(body, 'application/xml'));
        })
        .catch(function () { if (mine === seq) { list.hidden = true; say(box.dataset.error); } })
        .then(function () { if (mine === seq) box.classList.remove('is-loading'); });
    }

    // arXiv asks for at most one request every 3 s: wait until typing pauses
    input.addEventListener('input', function () { clearTimeout(timer); timer = setTimeout(run, 600); });
    if (source) source.addEventListener('change', function () { if (input.value.trim()) run(); });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); clearTimeout(timer); run(); }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    Array.prototype.forEach.call(document.querySelectorAll('.arxiv-picker'), picker);
  });

  window.SP = {lang: lang, post: post, wait: wait, text: text, track: track};
})();
