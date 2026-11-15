/* The ready-made post of share_post.html: each button posts the text of the box, with the page link tagged for
   its network (utm_source). X and Bluesky count the link in their limit: they get the shorter post unless it was
   edited. LinkedIn no longer documents a pre-filled text, so the post is also copied for pasting. */
(function () {
  var NETWORKS = {
    linkedin: function (p) { return 'https://www.linkedin.com/feed/?shareActive=true&text=' + encodeURIComponent(p.withLink); },
    x: function (p) { return 'https://x.com/intent/post?text=' + encodeURIComponent(p.withoutLink) + '&url=' + encodeURIComponent(p.link); },
    bluesky: function (p) { return 'https://bsky.app/intent/compose?text=' + encodeURIComponent(p.withLink); }
  };

  function tidy(text) {
    return text.replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
  }

  // The post for a network: {link, withLink, withoutLink}
  function post(box, network) {
    var area = box.querySelector('[data-share-text]');
    var edited = area.value.trim() !== area.defaultValue.trim();
    var text = (network === 'x' || network === 'bluesky') && !edited ? box.dataset.short : area.value;
    var bare = box.dataset.url;
    var link = box.getAttribute('data-link-' + network) || bare;
    var has = text.indexOf(bare) !== -1;
    return {
      link: link,
      withLink: tidy(has ? text.split(bare).join(link) : text + '\n\n' + link),
      withoutLink: tidy(text.split(bare).join(''))
    };
  }

  function copy(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    var field = document.createElement('textarea');
    field.value = text;
    field.setAttribute('readonly', '');
    field.style.position = 'fixed';
    field.style.opacity = '0';
    document.body.appendChild(field);
    field.select();
    var done = document.execCommand('copy');
    document.body.removeChild(field);
    return done ? Promise.resolve() : Promise.reject();
  }

  function say(box, message) {
    var status = box.querySelector('.share-status');
    if (status) status.textContent = message;
  }

  document.addEventListener('click', function (event) {
    var button = event.target.closest('[data-share-network]');
    if (!button) return;
    var box = button.closest('[data-share]');
    var network = button.dataset.shareNetwork;
    var p = post(box, network);
    if (NETWORKS[network]) {
      button.href = NETWORKS[network](p);  // the default action then opens the edited post
      if (network === 'linkedin') copy(p.withLink).then(function () { say(box, box.dataset.linkedinCopied); }, function () {});
    } else if (network === 'copy') {
      copy(p.withLink).then(function () { say(box, box.dataset.copied); }, function () {
        var area = box.querySelector('[data-share-text]');
        area.focus();
        area.select();
      });
    } else if (network === 'native' && navigator.share) {
      navigator.share({text: p.withoutLink, url: p.link}).catch(function () {});
    }
  });

  if (navigator.share) {
    document.querySelectorAll('[data-share] [data-share-network="native"]').forEach(function (button) { button.hidden = false; });
  }
  // the paper page's "Share" action opens the box it points to
  function openFromHash() {
    var box = location.hash === '#share' && document.getElementById('share');
    if (box) box.open = true;
  }
  window.addEventListener('hashchange', openFromHash);
  openFromHash();
})();
