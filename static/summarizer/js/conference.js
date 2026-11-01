(function () {
  'use strict';
  // Marks and notes of a conference program saved without leaving the page, and the digest profile chosen as
  // interests. Without this script the forms post and the page comes back at the talk.
  function send(form, submitter) {
    var data = new FormData(form);
    if (submitter && submitter.name) data.set(submitter.name, submitter.value);
    return fetch(form.action || location.href, {method: 'POST', body: data, credentials: 'same-origin',
      headers: {'X-Requested-With': 'fetch'}}).then(function (response) {
      if (!response.ok) throw new Error(response.status);
      return response.json();
    });
  }
  // every view of the program put in the cache the service worker reads without a network
  var VIEWS = ['', '?view=now', '?view=day', '?view=priority', '?view=people', '?view=all'];
  function keep() {
    var base = location.pathname;
    return caches.open('conference-v1').then(function (cache) {
      return Promise.all(VIEWS.map(function (view) {
        return fetch(base + view, {credentials: 'same-origin'}).then(function (response) {
          if (response.ok) return cache.put(base + view, response);
        });
      }));
    });
  }
  function init() {
    document.querySelectorAll('[data-conf-mark]').forEach(function (form) {
      form.addEventListener('submit', function (event) {
        event.preventDefault();
        var button = event.submitter;
        send(form, button).then(function (answer) {
          form.querySelectorAll('button[name=state]').forEach(function (b) {
            b.setAttribute('aria-pressed', String(b.value === answer.state));
          });
          var card = form.closest('.conf-talk');
          if (card) card.classList.toggle('conf-skipped', answer.state === 'skip');
        }).catch(function () {
          var state = document.createElement('input');
          state.type = 'hidden'; state.name = 'state'; state.value = button ? button.value : '';
          form.appendChild(state);
          form.submit();
        });
      });
    });
    document.querySelectorAll('[data-conf-note]').forEach(function (form) {
      form.addEventListener('submit', function (event) {
        event.preventDefault();
        var saved = form.querySelector('[data-saved]');
        send(form).then(function () {
          if (saved) { saved.hidden = false; setTimeout(function () { saved.hidden = true; }, 2000); }
        }).catch(function () { form.submit(); });
      });
    });
    // the pages of a program kept on the device, to read them without a network
    var offline = document.querySelector('[data-conf-offline]');
    if (offline && 'serviceWorker' in navigator) {
      navigator.serviceWorker.register(offline.dataset.confOffline).catch(function () {});
      var note = document.querySelector('[data-offline-note]');
      var show = function () { if (note) note.hidden = navigator.onLine; };
      window.addEventListener('online', show);
      window.addEventListener('offline', show);
      show();
      document.querySelectorAll('[data-conf-keep]').forEach(function (button) {
        button.hidden = !('caches' in window);
        button.addEventListener('click', function () {
          button.disabled = true;
          keep().then(function () { button.textContent = button.dataset.kept; })
            .catch(function () { button.disabled = false; });
        });
      });
      // while the conference goes on, its views are kept on the device once an hour without asking
      if (offline.hasAttribute('data-conf-auto') && 'caches' in window && navigator.onLine) {
        var stamp = 'conf-kept:' + location.pathname, last = 0;
        try { last = +localStorage.getItem(stamp) || 0; } catch (e) {}
        if (Date.now() - last > 3600000) {
          keep().then(function () { try { localStorage.setItem(stamp, String(Date.now())); } catch (e) {} }).catch(function () {});
        }
      }
    }
    var picker = document.querySelector('[data-digest-profile]');
    var interests = document.querySelector('[data-interests]');
    if (picker && interests) {
      picker.addEventListener('change', function () {
        interests.value = picker.selectedOptions[0].dataset.profile || interests.value;
      });
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
