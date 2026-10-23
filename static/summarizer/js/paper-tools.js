
window.SPPaperTools = {init: function () {
  var box = document.getElementById("tools");
  if (!box || box.dataset.mounted) return;
  box.dataset.mounted = "1";
  var arxivId = box.dataset.paper;
  var base = box.dataset.base;
  var t = document.getElementById('tools-text').dataset;

  function show(el, message, link) {
    el.textContent = message + (link ? ' ' : '');
    if (link) {
      var a = document.createElement('a');
      a.href = link.href;
      a.textContent = link.text;
      el.appendChild(a);
    }
  }
  function busy(el, message) {
    el.textContent = '';
    el.appendChild(SP.text('span', '', 'spinner spinner-sm'));
    el.appendChild(document.createTextNode(message));
  }
  function refused(el, json) {
    if (json.status === 'premium') return show(el, t.premium, {href: base + '/premium/', text: t.premiumLink}), true;
    if (json.status === 'login') return show(el, t.login, {href: base + '/login/?next=' + encodeURIComponent(location.pathname), text: '→'}), true;
    return false;
  }

  // reading levels
  var readingStatus = document.getElementById('reading-status');
  var readingOutput = document.getElementById('reading-output');
  function renderReading(data) {
    readingOutput.innerHTML = '';
    var box = SP.text('div', '', 'reading-output');
    if (data.title) box.appendChild(SP.text('h4', data.title));
    data.paragraphs.forEach(function (p) { box.appendChild(SP.text('p', p)); });
    readingOutput.appendChild(box);
    readingStatus.textContent = '';
  }
  var readingGo = document.getElementById('reading-go');
  readingGo.addEventListener('click', function () {
    readingGo.disabled = true;
    busy(readingStatus, t.working);
    function finish(fn) { return function (x) { readingGo.disabled = false; return fn(x); }; }
    SP.post(base + '/reading/' + arxivId + '/', {
      level: document.getElementById('reading-level').value,
      lang: document.getElementById('reading-lang').value
    }).then(finish(function (json) {
      if (json.status === 'quota') return show(readingStatus, t.quota, {href: base + '/premium/', text: t.premiumLink});
      if (json.status === 'done') return renderReading(json.data);
      if (!json.id) return show(readingStatus, t.failed);
      readingGo.disabled = true;
      SP.wait(json.id, finish(renderReading), finish(function () { show(readingStatus, t.failed); }));
    }));
  });

  // collections
  var pick = document.getElementById('collection-pick');
  if (pick) {
    var nameInput = document.getElementById('collection-name');
    var collectionStatus = document.getElementById('collection-status');
    pick.addEventListener('change', function () { nameInput.hidden = pick.value !== 'new'; });
    document.getElementById('collection-go').addEventListener('click', function () {
      SP.post(base + '/collections/add/', {arxiv_id: arxivId, collection: pick.value, name: nameInput.value})
        .then(function (json) {
          if (refused(collectionStatus, json)) return;
          if (json.status === 'added') return show(collectionStatus, t.added + ' « ' + json.name + ' ».', {href: json.url, text: t.open});
          if (json.status === 'already') return show(collectionStatus, t.already, {href: json.url, text: t.open});
          if (json.status === 'full') return show(collectionStatus, t.full);
          if (json.status === 'too_many') return show(collectionStatus, t.tooMany, {href: base + '/premium/', text: t.premiumLink});
          show(collectionStatus, t.failed);
        });
    });
  }

  // Zotero
  var zoteroStatus = document.getElementById('zotero-status');
  document.getElementById('zotero-go').addEventListener('click', function () {
    busy(zoteroStatus, t.sending);
    SP.post(base + '/zotero/send/', {arxiv_id: arxivId}).then(function (json) {
      if (refused(zoteroStatus, json)) return;
      if (json.status === 'setup') return show(zoteroStatus, t.setup, {href: json.url, text: '→'});
      if (!json.id) return show(zoteroStatus, t.failed);
      SP.wait(json.id, function () { show(zoteroStatus, t.sent); }, function () { show(zoteroStatus, t.failed); });
    });
  });

  // citation alerts
  var followBtn = document.getElementById('follow-go');
  var followStatus = document.getElementById('follow-status');
  followBtn.addEventListener('click', function () {
    var following = followBtn.dataset.following === '1';
    SP.post(base + '/citations/follow/', {arxiv_id: arxivId, action: following ? 'unfollow' : 'follow'}).then(function (json) {
      if (refused(followStatus, json)) return;
      if (json.status === 'too_many') return show(followStatus, t.full);
      following = json.status === 'following';
      followBtn.dataset.following = following ? '1' : '0';
      followBtn.textContent = following ? t.unfollow : t.follow;
      show(followStatus, following ? t.following : t.unfollowed, following ? {href: base + '/citations/', text: '→'} : null);
    });
  });
}};
SPPaperTools.init();
