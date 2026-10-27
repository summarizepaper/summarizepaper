/* Press kit page: the reader's role (remembered on this browser), copying a passage with its reference,
   the read-only share link, and reporting a mistake. Generation itself is the inline script of presskit.html. */
(function () {
  'use strict';
  var STORE = 'presskit-role';

  // usage of the page (fixed labels only, as funnel.js): which roles, copies, shares and reports are used
  function track(name, data) {
    try { if (window.umami) Promise.resolve(window.umami.track(name, data)).catch(function () {}); } catch (_) {}
  }

  function remember(role) { try { localStorage.setItem(STORE, role); } catch (_) {} }
  function remembered() { try { return localStorage.getItem(STORE); } catch (_) { return null; } }

  function copy(text, button, root) {
    function done() {
      if (!button) return;
      var label = button.textContent;
      button.textContent = root.dataset.copied || 'Copied';
      setTimeout(function () { button.textContent = label; }, 1600);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(function () { fallback(text); done(); });
    } else { fallback(text); done(); }
  }

  function fallback(text) {
    var area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.appendChild(area);
    area.select();
    try { document.execCommand('copy'); } catch (_) {}
    area.remove();
  }

  function blockText(node) {
    var items = node.querySelectorAll('li');
    if (items.length) return Array.prototype.map.call(items, function (li) { return li.textContent.trim(); }).join('\n\n');
    var paragraphs = node.querySelectorAll('h4, p');
    if (paragraphs.length) return Array.prototype.map.call(paragraphs, function (p) { return p.textContent.trim(); }).join('\n\n');
    return node.textContent.trim();
  }

  document.addEventListener('DOMContentLoaded', function () {
    var root = document.getElementById('presskit');
    if (!root) return;
    var tabs = root.querySelectorAll('[data-role-tab]');

    function show(role, keep) {
      if (!root.querySelector('[data-role-panel="' + role + '"]')) return;
      root.dataset.role = role;
      tabs.forEach(function (tab) { tab.setAttribute('aria-selected', tab.dataset.roleTab === role ? 'true' : 'false'); });
      root.querySelectorAll('[data-role-panel]').forEach(function (panel) { panel.hidden = panel.dataset.rolePanel !== role; });
      root.querySelectorAll('[data-role-link]').forEach(function (link) {
        var url = new URL(link.href, location.href);
        url.searchParams.set('role', role);
        link.href = url.pathname + url.search;
      });
      if (keep) {
        remember(role);
        var page = new URL(location.href);
        page.searchParams.set('role', role);
        history.replaceState(null, '', page.pathname + page.search + page.hash);
      }
    }
    tabs.forEach(function (tab) {
      tab.addEventListener('click', function () { show(tab.dataset.roleTab, true); track('presskit_role', {role: tab.dataset.roleTab}); });
    });
    var stored = remembered();
    if (stored && !new URLSearchParams(location.search).get('role')) show(stored, false);

    root.addEventListener('click', function (event) {
      var button = event.target.closest('[data-copy-quote]');
      if (button) {
        var where = [root.dataset.citation, button.dataset.copyWhere].filter(Boolean).join(', ');
        copy('“' + button.dataset.copyQuote + '”' + (where ? ' (' + where + ')' : ''), button, root);
        track('presskit_copy', {what: 'quote'});
        return;
      }
      button = event.target.closest('[data-copy-block]');
      if (button) {
        var node = document.getElementById(button.dataset.copyBlock);
        if (node) copy(blockText(node), button, root);
        track('presskit_copy', {what: button.dataset.copyBlock.replace(/^kit-|-\d+$/g, '')});
        return;
      }
      button = event.target.closest('[data-copy-field]');
      if (button) {
        var field = document.getElementById(button.dataset.copyField);
        if (field && field.value) copy(field.value, button, root);
      }
    });

    // read-only link for a press office or co-authors
    var shareBox = root.querySelector('[data-kit-share-box]');
    function share(action) {
      if (!root.dataset.share) return;
      track('presskit_share', {action: action});
      SP.post(root.dataset.share, {action: action}).then(function (json) {
        if (json.status !== 'ok') return;
        var field = document.getElementById('kit-share-url');
        field.value = json.url || '';
        shareBox.hidden = !json.url;
        if (action === 'update') shareBox.querySelector('[data-kit-share-outdated]').hidden = true;
        if (json.url) field.select();
      });
    }
    var update = root.querySelector('[data-kit-share-update]');
    if (update) update.addEventListener('click', function () { share('update'); });
    var shareButton = root.querySelector('[data-kit-share]');
    if (shareButton) shareButton.addEventListener('click', function () { share('share'); });
    var unshare = root.querySelector('[data-kit-unshare]');
    if (unshare) unshare.addEventListener('click', function () { share('unshare'); });

    // a mistake reported on a card, the experts, the novelty...
    var dialog = document.getElementById('kit-report-dialog');
    if (!dialog || !root.dataset.report || typeof dialog.showModal !== 'function') return;
    var about = dialog.querySelector('[data-report-about]'), text = dialog.querySelector('textarea');
    var error = dialog.querySelector('[data-report-error]'), current = {};
    root.addEventListener('click', function (event) {
      var button = event.target.closest('[data-report-section]');
      if (!button) return;
      current = {section: button.dataset.reportSection, item: button.dataset.reportItem || '', button: button};
      about.textContent = current.item;
      text.value = '';
      error.hidden = true;
      dialog.showModal();
      text.focus();
    });
    dialog.querySelector('[data-report-cancel]').addEventListener('click', function () { dialog.close(); });
    dialog.querySelector('form').addEventListener('submit', function (event) {
      event.preventDefault();
      if (!text.value.trim()) return;
      SP.post(root.dataset.report, {section: current.section, item: current.item, text: text.value}).then(function (json) {
        if (json.status !== 'ok') throw json;
        dialog.close();
        track('presskit_report', {section: current.section});
        current.button.textContent = root.dataset.reportSent;
        current.button.disabled = true;
      }).catch(function () {
        error.textContent = root.dataset.failed;
        error.hidden = false;
      });
    });
  });
})();
