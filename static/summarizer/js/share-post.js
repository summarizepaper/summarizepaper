/* The ready-made posts of share_post.html: each button posts the text of its box, with the page link tagged for
   its network (utm_source). X and Bluesky count the link in their limit: they get the shorter post unless it was
   edited. LinkedIn no longer documents a pre-filled text, so the post is also copied for pasting.
   A page can hold several boxes (one per rising topic) and so load this file several times: it runs once. */
(function () {
  if (window.SPSharePost) return;
  window.SPSharePost = true;

  var NETWORKS = {
    linkedin: function (p) { return 'https://www.linkedin.com/feed/?shareActive=true&text=' + encodeURIComponent(p.withLink); },
    x: function (p) { return 'https://x.com/intent/post?text=' + encodeURIComponent(p.withoutLink) + '&url=' + encodeURIComponent(p.link); },
    bluesky: function (p) { return 'https://bsky.app/intent/compose?text=' + encodeURIComponent(p.withLink); },
    email: function (p, box) { return 'mailto:?subject=' + encodeURIComponent(box.dataset.subject || '') + '&body=' + encodeURIComponent(p.withLink); }
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

  // Opening a box asks for its picture once: a paper's figure is fetched in the background, ready when the
  // network reads the link a minute later.
  function warm(box) {
    if (!box.dataset.image || box.dataset.warmed) return;
    box.dataset.warmed = '1';
    try { fetch(box.dataset.image, {mode: 'no-cors', credentials: 'omit'}).catch(function () {}); } catch (_) { /* old browsers */ }
  }

  function openBox(id, scroll) {
    var box = document.getElementById(id);
    if (!box || !box.hasAttribute('data-share')) return;
    box.open = true;
    warm(box);
    if (scroll) box.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  }

  document.addEventListener('click', function (event) {
    var opener = event.target.closest('[data-share-open], a[href="#share"]');
    if (opener) {
      event.preventDefault();
      openBox(opener.dataset.shareOpen || 'share', true);
      return;
    }
    var button = event.target.closest('[data-share-network]');
    if (!button) return;
    var box = button.closest('[data-share]');
    var network = button.dataset.shareNetwork;
    var p = post(box, network);
    if (NETWORKS[network]) {
      button.href = NETWORKS[network](p, box);  // the default action then opens the edited post
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

  document.addEventListener('toggle', function (event) {
    if (event.target.hasAttribute && event.target.hasAttribute('data-share') && event.target.open) warm(event.target);
  }, true);

  function ready() {
    if (navigator.share) {
      document.querySelectorAll('[data-share] [data-share-network="native"]').forEach(function (button) { button.hidden = false; });
    }
    document.querySelectorAll('[data-share][open]').forEach(warm);
    // a link to a box (the "Share" action, an email's "Share" link) opens it
    if (location.hash.length > 1) openBox(decodeURIComponent(location.hash.slice(1)), false);
    // a shared topic: its card first
    var shared = document.querySelector('.topic-card.is-shared');
    if (shared && !location.hash) shared.scrollIntoView({block: 'start'});
  }
  window.addEventListener('hashchange', function () { if (location.hash.length > 1) openBox(decodeURIComponent(location.hash.slice(1)), true); });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready); else ready();
})();
