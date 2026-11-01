/* Zotero on the integrations page (the collections of the library, each opened here or checked; the collection
   single papers go to) and the "from Zotero" pickers that fill reference papers (digest, scoop watch). */
(function () {
  var lang = document.documentElement.lang || 'en';
  var base = '/' + lang;

  function loadCollections() {
    return fetch(base + '/collections/zotero/', {credentials: 'same-origin'}).then(function (r) { return r.json(); });
  }

  function option(value, label) {
    var o = document.createElement('option');
    o.value = value;
    o.textContent = label;
    return o;
  }

  // --- the integrations page ----------------------------------------------------------------------
  var hub = document.getElementById('zotero-hub');
  var inbox = document.getElementById('zotero-inbox');
  if (hub) {
    var t = hub.dataset;
    var status = document.getElementById('zotero-hub-status');
    var list = document.getElementById('zotero-hub-list');
    loadCollections().then(function (json) {
      if (json.status === 'unavailable') return status.textContent = t.unavailable;
      if (json.status !== 'done') return status.textContent = t.failed;
      if (!json.collections.length) return status.textContent = t.empty;
      status.textContent = '';
      json.collections.forEach(function (c) {
        var li = document.createElement('li');
        var name = document.createElement('strong');
        name.textContent = c.name + (c.items ? ' (' + c.items + ')' : '');
        var open = document.createElement('button');
        open.type = 'button';
        open.className = 'btn btn-sm';
        open.textContent = t.openLabel;
        var check = document.createElement('a');
        check.className = 'btn btn-sm btn-ghost';
        check.href = t.refcheck + encodeURIComponent(c.key);
        check.textContent = t.checkLabel;
        var said = document.createElement('span');
        said.className = 'tool-status';
        said.setAttribute('aria-live', 'polite');
        open.addEventListener('click', function () {
          open.disabled = true;
          said.textContent = t.opening;
          SP.post(t.open, {key: c.key, name: c.name.split(' / ').pop()}).then(function (json) {
            if (json.status === 'too_many') { open.disabled = false; return said.textContent = t.tooMany; }
            if (!json.id) { open.disabled = false; return said.textContent = t.failed; }
            SP.wait(json.id, function () { location.href = json.url; },
                    function () { open.disabled = false; said.textContent = t.failed; });
          });
        });
        li.appendChild(name);
        li.appendChild(document.createTextNode(' '));
        li.appendChild(open);
        li.appendChild(document.createTextNode(' '));
        li.appendChild(check);
        li.appendChild(document.createTextNode(' '));
        li.appendChild(said);
        list.appendChild(li);
        if (inbox && c.key !== inbox.dataset.selected) inbox.appendChild(option(c.key, c.name));
      });
    }).catch(function () { status.textContent = t.unavailable; });
  }
  if (inbox) inbox.addEventListener('change', function () {
    var chosen = inbox.options[inbox.selectedIndex];
    document.getElementById('zotero-inbox-name').value = inbox.value ? chosen.textContent : '';
  });

  // --- "from Zotero" pickers -----------------------------------------------------------------------
  Array.prototype.slice.call(document.querySelectorAll('[data-zotero-fill]')).forEach(function (box) {
    var button = box.querySelector('button');
    var select = box.querySelector('select');
    var said = box.querySelector('.tool-status');
    var target = document.getElementById(box.dataset.target);
    var comma = box.dataset.separator === 'comma';
    button.addEventListener('click', function () {
      said.textContent = box.dataset.loading;
      loadCollections().then(function (json) {
        if (json.status === 'setup') { said.textContent = ''; var a = document.createElement('a'); a.href = json.url; a.textContent = box.dataset.setup; return said.appendChild(a); }
        if (json.status !== 'done') return said.textContent = box.dataset.unavailable;
        if (!json.collections.length) return said.textContent = box.dataset.empty;
        select.appendChild(option('', '—'));
        json.collections.forEach(function (c) { select.appendChild(option(c.key, c.name + (c.items ? ' (' + c.items + ')' : ''))); });
        said.textContent = '';
        button.hidden = true;
        select.hidden = false;
      }).catch(function () { said.textContent = box.dataset.unavailable; });
    });
    select.addEventListener('change', function () {
      if (!select.value) return;
      said.textContent = box.dataset.loading;
      fetch(base + '/zotero/references/?n=' + (box.dataset.n || 5) + '&key=' + encodeURIComponent(select.value), {credentials: 'same-origin'})
        .then(function (r) { return r.json(); }).then(function (json) {
          if (json.status !== 'done') return said.textContent = box.dataset.unavailable;
          var have = target.value.split(comma ? /\s*,\s*/ : /\s+/).filter(Boolean);
          json.refs.forEach(function (r) { if (have.indexOf(r) < 0) have.push(r); });
          target.value = have.slice(0, Number(box.dataset.n || 5)).join(comma ? ', ' : '\n');
          said.textContent = json.refs.length ? box.dataset.filled.replace('%(n)s', json.refs.length) : box.dataset.none;
        }).catch(function () { said.textContent = box.dataset.unavailable; });
    });
  });
})();
