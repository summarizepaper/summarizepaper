/* The map of the tree page: the papers around a paper by date (x) and closeness to it (y), lines between the
   papers close to each other, and a panel with the selected paper. The list of templates/summarizer/neighbours.html
   stays below it: what the map shows, readable without a pointer. */
(function () {
  var box = document.getElementById('nb-map');
  var source = document.getElementById('nb-map-data');
  if (!box || !source || !window.d3) return;
  var t = box.dataset;
  var data = JSON.parse(source.textContent);
  var panel = document.getElementById('nb-panel');
  var list = document.querySelector('section.neighbours');
  var base = '/' + (document.documentElement.lang || 'en');
  var COLORS = {close: '#4f46e5', prior: '#d97706', followup: '#059669'};
  var LABELLED = 12;  // the closest ones get a name on the map when it does not cover another, the others on hover or focus
  var focused = null;

  function when(text) {
    if (!text) return null;
    var parts = text.split('-');
    return new Date(Date.UTC(+parts[0], parts[1] ? +parts[1] - 1 : 6, parts[2] ? +parts[2] : 1));
  }
  function surname(authors) {
    var first = (authors || '').split(',')[0].replace(' et al.', '').trim();
    return first.split(' ').pop();
  }
  function el(tag, text, className) {
    var node = document.createElement(tag);
    if (text) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function checkboxes(ref) {
    return list ? Array.prototype.slice.call(list.querySelectorAll('.nb-check')).filter(function (c) { return c.value === ref; }) : [];
  }

  var nodes = data.nodes.map(function (n, i) { return Object.assign({index: i, time: when(n.date)}, n); });
  var dated = nodes.filter(function (n) { return n.time; });
  var mainTime = when(data.main.date);
  var ranked = dated.slice().sort(function (a, b) { return b.similarity - a.similarity; });

  function showPanel(n) {
    panel.textContent = '';
    var tags = el('div', '', 'nb-tags');
    tags.appendChild(el('span', n.level_label, 'nb-level nb-level-' + n.level));
    if (n.relation_label) tags.appendChild(el('span', n.relation_label, 'nb-relation'));
    if (n.cited) tags.appendChild(el('span', t.cited, 'nb-link'));
    if (n.cites) tags.appendChild(el('span', t.cites, 'nb-link'));
    panel.appendChild(tags);
    var title = el('h3');
    var link = el('a', n.title);
    link.href = base + '/paper/' + n.ref + '/';
    title.appendChild(link);
    panel.appendChild(title);
    panel.appendChild(el('div', [n.authors, (n.date || '').slice(0, 4), n.venue].filter(Boolean).join(' · '), 'nb-sub'));
    if (n.why) panel.appendChild(el('p', n.why, 'nb-why'));
    if (n.abstract) panel.appendChild(el('p', n.abstract + (n.abstract.length >= 700 ? '…' : '')));
    var buttons = el('div', '', 'nb-panel-actions');
    var open = el('a', n.summarized ? t.open : t.summarize, 'btn btn-sm');
    open.href = link.href;
    buttons.appendChild(open);
    var boxes = checkboxes(n.ref);
    if (boxes.length) {
      var pick = el('button', boxes[0].checked ? t.unselect : t.select, 'btn btn-sm btn-ghost');
      pick.type = 'button';
      pick.addEventListener('click', function () {
        boxes[0].checked = !boxes[0].checked;
        boxes[0].dispatchEvent(new Event('change'));
        pick.textContent = boxes[0].checked ? t.unselect : t.select;
      });
      buttons.appendChild(pick);
    }
    panel.appendChild(buttons);
    panel.hidden = false;
  }

  function draw() {
    d3.select(box).select('svg').remove();
    if (!dated.length) return;
    var width = box.clientWidth;
    var height = Math.max(340, Math.min(560, Math.round(width * 0.62)));
    var m = {top: 28, right: 20, bottom: 34, left: 46};
    var times = dated.map(function (n) { return n.time; }).concat(mainTime ? [mainTime] : []);
    var lo = d3.min(times), hi = d3.max(times);
    var pad = Math.max(90 * 864e5, (hi - lo) * 0.04);
    var x = d3.scaleUtc().domain([new Date(+lo - pad), new Date(+hi + pad)]).range([m.left, width - m.right]);
    var sims = dated.map(function (n) { return n.similarity; });
    var y = d3.scaleLinear().domain([d3.min(sims) - 0.02, d3.max(sims) + 0.02]).range([height - m.bottom, m.top]);
    var svg = d3.select(box).insert('svg', '.nb-map-legend')
      .attr('viewBox', '0 0 ' + width + ' ' + height).attr('role', 'group').attr('aria-label', t.closeness);

    svg.append('g').attr('class', 'axis').attr('transform', 'translate(0,' + (height - m.bottom) + ')')
      .call(d3.axisBottom(x).ticks(Math.max(2, Math.floor(width / 110))));
    svg.append('g').attr('class', 'axis').attr('transform', 'translate(' + m.left + ',0)')
      .call(d3.axisLeft(y).ticks(5).tickFormat(function (v) { return Math.round(v * 100) + '%'; }));
    svg.append('text').attr('class', 'main-label').attr('x', m.left + 4).attr('y', m.top - 12).attr('fill', 'currentColor')
      .style('font-size', '12px').text('↑ ' + t.closeness);
    if (mainTime) {
      svg.append('line').attr('class', 'main-line').attr('x1', x(mainTime)).attr('x2', x(mainTime))
        .attr('y1', m.top).attr('y2', height - m.bottom);
      svg.append('text').attr('class', 'main-label').attr('x', x(mainTime)).attr('y', height - m.bottom - 6)
        .attr('text-anchor', x(mainTime) > width / 2 ? 'end' : 'start').attr('dx', x(mainTime) > width / 2 ? -4 : 4).text(t.thisPaper);
    }

    // keep the points from covering each other, as close as possible to their place
    dated.forEach(function (n) { n.x = n.tx = x(n.time); n.y = n.ty = y(n.similarity); });
    var sim = d3.forceSimulation(dated)
      .force('x', d3.forceX(function (n) { return n.tx; }).strength(0.6))
      .force('y', d3.forceY(function (n) { return n.ty; }).strength(0.6))
      .force('collide', d3.forceCollide(9)).stop();
    for (var i = 0; i < 120; i++) sim.tick();

    var byIndex = {};
    dated.forEach(function (n) { byIndex[n.index] = n; });
    var edges = svg.append('g').selectAll('line')
      .data(data.edges.filter(function (e) { return byIndex[e[0]] && byIndex[e[1]]; }))
      .enter().append('line').attr('class', 'edge')
      .attr('x1', function (e) { return byIndex[e[0]].x; }).attr('y1', function (e) { return byIndex[e[0]].y; })
      .attr('x2', function (e) { return byIndex[e[1]].x; }).attr('y2', function (e) { return byIndex[e[1]].y; });

    var points = svg.append('g').selectAll('g').data(dated).enter().append('g')
      .attr('class', 'node').attr('transform', function (n) { return 'translate(' + n.x + ',' + n.y + ')'; })
      .attr('tabindex', 0).attr('role', 'button')
      .attr('aria-label', function (n) { return n.title + ' (' + n.level_label + ')'; });
    points.append('circle').attr('r', function (n) { return n.level === 'very_close' ? 8 : 6.5; })
      .attr('fill', function (n) { return COLORS[n.group] || COLORS.close; });
    var labels = points.append('text').attr('x', 10).attr('y', 4)
      .attr('text-anchor', function (n) { return n.x > width - 120 ? 'end' : 'start'; })
      .attr('dx', function (n) { return n.x > width - 120 ? -20 : 0; })
      .text(function (n) { return surname(n.authors) + ' ' + (n.date || '').slice(0, 4); })
      .style('display', 'none');
    points.append('title').text(function (n) { return n.title; });

    // the closest first, each name only where it covers no name already placed
    var room = width < 600 ? LABELLED / 2 : LABELLED, named = 0;
    var placed = dated.map(function (n) { return {x0: n.x - 8, y0: n.y - 8, x1: n.x + 8, y1: n.y + 8, node: n}; });
    dated.forEach(function (n) { n.labelled = false; });
    labels.each(function (n) { n.label = this; });
    ranked.forEach(function (n) {
      if (named >= room) return;
      n.label.style.display = null;
      var box = n.label.getBBox(), x0 = n.x + box.x, y0 = n.y + box.y;
      var free = placed.every(function (b) {
        return b.node === n || x0 > b.x1 || x0 + box.width < b.x0 || y0 > b.y1 || y0 + box.height < b.y0;
      });
      if (free) {
        n.labelled = true;
        named += 1;
        placed.push({x0: x0 - 2, y0: y0 - 1, x1: x0 + box.width + 2, y1: y0 + box.height + 1});
      }
    });
    labels.style('display', function (n) { return n.labelled ? null : 'none'; });

    function light(n, on) {
      edges.classed('is-lit', function (e) { return on && (e[0] === n.index || e[1] === n.index); });
      labels.style('display', function (o) { return o.labelled || (on && o === n) || o === focused ? null : 'none'; });
    }
    function choose(n) {
      focused = n;
      points.classed('is-focused', function (o) { return o === n; });
      light(n, true);
      showPanel(n);
      if (list) list.querySelectorAll('.nb-item').forEach(function (li) { li.classList.toggle('is-focused', li.dataset.ref === n.ref); });
    }
    points.on('mouseenter focus', function (event, n) { light(n, true); })
      .on('mouseleave blur', function () { light(focused, !!focused); })
      .on('click', function (event, n) { choose(n); })
      .on('keydown', function (event, n) {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(n); }
      });
    if (focused) choose(focused);
  }

  var undated = nodes.length - dated.length;
  if (undated) box.appendChild(el('p', t.undated.replace('{n}', undated), 'nb-map-legend'));
  draw();
  var resizing;
  window.addEventListener('resize', function () { clearTimeout(resizing); resizing = setTimeout(draw, 150); });
  // the sentences arrive after the page: the panel shows them too
  if (list) list.addEventListener('neighbours:why', function (event) {
    var relations = JSON.parse(list.dataset.relations || '{}');
    nodes.forEach(function (n) {
      var reason = event.detail[n.ref];
      if (!reason) return;
      n.why = reason.why || n.why;
      n.relation_label = relations[reason.relation] || n.relation_label;
    });
    if (focused) showPanel(focused);
  });
})();
