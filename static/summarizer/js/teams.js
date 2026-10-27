(function () {
  'use strict';
  // The tabs and filters of a topic map, and the competing teams sent to the scoop watch. Without this script
  // every section shows, unfiltered.
  function init() {
    var toolbar = document.querySelector('[data-teams-toolbar]');
    if (!toolbar) return;
    toolbar.hidden = false;
    var tabs = toolbar.querySelectorAll('[data-tab]');
    var panels = document.querySelectorAll('[data-panel]');
    var filters = toolbar.querySelectorAll('[data-filter]');
    var count = toolbar.querySelector('[data-filter-count]');
    var current = 'people';

    function show(name) {
      current = name;
      tabs.forEach(function (tab) {tab.setAttribute('aria-selected', String(tab.dataset.tab === name));});
      panels.forEach(function (panel) {panel.hidden = panel.dataset.panel !== name;});
      apply();
    }
    function value(name) {
      var field = toolbar.querySelector('[data-filter="' + name + '"]');
      return !field ? '' : field.type === 'checkbox' ? field.checked : field.value;
    }
    function apply() {
      var country = value('country'), approach = value('approach'), lead = value('lead'), mine = value('mine');
      var shown = 0, total = 0;
      document.querySelectorAll('[data-panel="' + current + '"] [data-item]').forEach(function (item) {
        var visible = (!country || item.dataset.country === country) &&
          (!approach || item.dataset.approaches.split(' ').indexOf(approach) >= 0) &&
          (!lead || Number(item.dataset.lead) > 0) && (!mine || !item.dataset.relation);
        item.hidden = !visible;
        total += 1;
        shown += visible ? 1 : 0;
      });
      document.querySelectorAll('[data-approach-card]').forEach(function (card) {
        card.hidden = !!approach && card.dataset.approachCard !== approach;
      });
      count.textContent = total && shown !== total ? shown + ' / ' + total : '';
    }
    tabs.forEach(function (tab) {tab.addEventListener('click', function () {show(tab.dataset.tab);});});
    filters.forEach(function (field) {field.addEventListener('change', apply);});
    show('people');

    var form = document.querySelector('[data-competitor-form]');
    document.querySelectorAll('[data-competitor]').forEach(function (button) {
      button.addEventListener('click', async function () {
        var status = button.parentNode.querySelector('[data-competitor-status]');
        var body = new FormData();
        body.append('csrfmiddlewaretoken', form.querySelector('[name=csrfmiddlewaretoken]').value);
        body.append('pk', form.querySelector('[name=pk]').value);
        body.append('action', 'competitor');
        body.append('name', button.dataset.competitor);
        button.disabled = true;
        try {
          var response = await fetch(form.dataset.url, {method: 'POST', body: body, credentials: 'same-origin',
                                                         headers: {'X-Requested-With': 'fetch'}});
          if (!response.ok) throw new Error('competitor');
          status.textContent = (await response.json()).message || '';
        } catch (error) {
          status.textContent = form.dataset.failed;
          button.disabled = false;
        }
      });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
