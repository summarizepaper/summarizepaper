/* Papers around a paper (templates/summarizer/neighbours.html): select some to chat with them or keep them in a
   collection, and the sentence saying what links each one to the paper, asked for once the list is in view. */
window.SPNeighbours = {init: function () {
  var box = document.querySelector('section.neighbours');
  if (!box || box.dataset.mounted) return;
  box.dataset.mounted = '1';
  var d = box.dataset;
  var relations = JSON.parse(d.relations || '{}');
  var actions = box.querySelector('.nb-actions');
  var count = box.querySelector('.nb-count');
  var chat = box.querySelector('.nb-chat');
  var status = box.querySelector('.nb-status');
  var checks = Array.prototype.slice.call(box.querySelectorAll('.nb-check'));

  function el(tag, text) {
    var node = document.createElement(tag);
    if (text) node.textContent = text;
    return node;
  }
  function whenReady(fn) {
    if (window.SP) fn(); else document.addEventListener('DOMContentLoaded', fn);
  }
  function selected() {
    var refs = [];
    checks.forEach(function (c) { if (c.checked && refs.indexOf(c.value) < 0) refs.push(c.value); });
    return refs;
  }
  function show(message, link) {
    status.textContent = message + (link ? ' ' : '');
    if (link) {
      var a = document.createElement('a');
      a.href = link.href;
      a.textContent = link.text;
      status.appendChild(a);
    }
  }
  function update() {
    var refs = selected();
    checks.forEach(function (c) { c.closest('.nb-item').classList.toggle('is-selected', c.checked); });
    actions.hidden = !refs.length;
    count.textContent = refs.length === 1 ? d.selectedOne : d.selectedMany.replace('{n}', refs.length);
    var tooMany = refs.length > parseInt(d.chatMax, 10);
    chat.href = d.base + '/chat/?ids=' + encodeURIComponent([d.paper].concat(refs).join(','));
    chat.classList.toggle('is-disabled', tooMany);
    chat.setAttribute('aria-disabled', tooMany ? 'true' : 'false');
    status.textContent = tooMany ? d.chatTooMany : '';
  }
  checks.forEach(function (c) {
    c.addEventListener('change', function () {
      // the same paper may be in two lists of the map page: keep both boxes alike
      checks.forEach(function (o) { if (o !== c && o.value === c.value) o.checked = c.checked; });
      update();
    });
  });
  chat.addEventListener('click', function (event) {
    if (chat.getAttribute('aria-disabled') === 'true') event.preventDefault();
  });

  var go = box.querySelector('.nb-collection-go');
  if (go) {
    var pick = box.querySelector('.nb-collection-pick');
    go.addEventListener('click', function () {
      go.disabled = true;
      SP.post(d.base + '/collections/add/', {ids: [d.paper].concat(selected()).join(','), collection: pick.value, name: go.dataset.name})
        .then(function (json) {
          go.disabled = false;
          if (json.status === 'added') {
            if (pick.value === 'new') {
              var option = document.createElement('option');
              option.value = json.collection;
              option.textContent = json.name;
              pick.appendChild(option);
              pick.value = json.collection;
            }
            return show(d.added + ' « ' + json.name + ' ».', {href: json.url, text: d.open});
          }
          if (json.status === 'already' || json.status === 'duplicate') return show(d.already, {href: json.url, text: d.open});
          if (json.status === 'full') return show(d.full);
          if (json.status === 'too_many') return show(d.tooMany, {href: d.base + '/premium/', text: d.premiumLink});
          show(d.failed);
        }, function () { go.disabled = false; show(d.failed); });
    });
  }

  // a paper marked not relevant leaves the list, for this reader now and for everyone once enough agree
  box.querySelectorAll('.nb-hide').forEach(function (button) {
    button.addEventListener('click', function () {
      var li = button.closest('.nb-item');
      var ref = li.dataset.ref;
      var check = li.querySelector('.nb-check');
      if (check.checked) { check.checked = false; update(); }
      var kept = Array.prototype.slice.call(li.childNodes);
      var note = el('span', d.hidden + ' ');
      var undo = el('button', d.undo);
      undo.type = 'button';
      note.appendChild(undo);
      kept.forEach(function (n) { n.remove(); });
      li.appendChild(note);
      li.classList.add('is-hidden');
      SP.post(d.markUrl, {ref: ref});
      undo.addEventListener('click', function () {
        note.remove();
        kept.forEach(function (n) { li.appendChild(n); });
        li.classList.remove('is-hidden');
        SP.post(d.markUrl, {ref: ref, undo: '1'});
      });
    });
  });

  // what links each paper to this one
  function fill(data) {
    var byRef = (data && data.by_ref) || {};
    box.querySelectorAll('.nb-item').forEach(function (li) {
      var reason = byRef[li.dataset.ref];
      if (!reason) return;
      var why = li.querySelector('.nb-why');
      if (reason.why) { why.textContent = reason.why; why.hidden = false; }
      var relation = li.querySelector('.nb-relation');
      if (relations[reason.relation]) { relation.textContent = relations[reason.relation]; relation.hidden = false; }
    });
    box.dispatchEvent(new CustomEvent('neighbours:why', {detail: byRef}));
  }
  function ask() {
    SP.post(d.whyUrl, {}).then(function (json) {
      if (json.status === 'done') return fill(json.data);
      if (json.id) SP.wait(json.id, fill);
    });
  }
  if (d.whyReady !== '1' && checks.length) {
    if (!('IntersectionObserver' in window)) return whenReady(ask);
    var seen = new IntersectionObserver(function (entries) {
      if (!entries.some(function (e) { return e.isIntersecting; })) return;
      seen.disconnect();
      whenReady(ask);
    });
    seen.observe(box);
  }
}};
SPNeighbours.init();
