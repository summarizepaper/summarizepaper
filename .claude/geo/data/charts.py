"""Generate the two guide charts as inline SVG figures (no blank lines, so Markdown passes them through)."""
import sys

BLUE, ORANGE = 'var(--viz-1)', 'var(--viz-2)'


def fmt(n):
    return '{:,}'.format(n)


def columns():
    data = [(2015, 105280), (2016, 113380), (2017, 123523), (2018, 140616), (2019, 155866),
            (2020, 178329), (2021, 181630), (2022, 185692), (2023, 208493), (2024, 244031),
            (2025, 284486), (2026, 293057)]
    W, H = 760, 340
    left, right, top, bottom = 56, 16, 28, 36
    pw, ph = W - left - right, H - top - bottom
    ymax = 300000
    band = pw / len(data)
    bw = min(24, band * 0.6)
    y = lambda v: top + ph - v / ymax * ph
    out = ['<figure class="viz">',
           '<figcaption><strong>New arXiv submissions per year, 2015–2026</strong><span>2026 counted to 18 October. Source: arXiv monthly submissions.</span></figcaption>',
           '<div class="viz-scroll"><svg viewBox="0 0 %d %d" role="img" aria-label="Column chart of new arXiv submissions per year from 2015 (105,280) to 2025 (284,486), with 2026 already at 293,057 by 18 October.">' % (W, H)]
    for t in range(0, ymax + 1, 100000):
        out.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (left, W - right, y(t), y(t)))
        out.append('<text class="tick" x="%d" y="%.1f" text-anchor="end" dy="4">%s</text>' % (left - 8, y(t), '0' if t == 0 else '%dK' % (t // 1000)))
    for i, (yr, v) in enumerate(data):
        cx = left + band * i + band / 2
        x0, top_y, base = cx - bw / 2, y(v), y(0)
        r = 4
        # square at the baseline, 4px rounded data-end
        d = 'M%.1f,%.1f V%.1f Q%.1f,%.1f %.1f,%.1f H%.1f Q%.1f,%.1f %.1f,%.1f V%.1f Z' % (
            x0, base, top_y + r, x0, top_y, x0 + r, top_y, x0 + bw - r, x0 + bw, top_y, x0 + bw, top_y + r, base)
        cls = 'bar partial' if yr == 2026 else 'bar'
        tip = '%d: %s submissions%s' % (yr, fmt(v), ' (to 18 October)' if yr == 2026 else '')
        out.append('<g class="hit" data-tip="%s"><rect x="%.1f" y="%d" width="%.1f" height="%d" fill="transparent"/><path class="%s" d="%s"/></g>' % (
            tip, cx - band / 2, top, band, ph, cls, d))
        out.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (cx, H - 12, ("'%02d" % (yr % 100))))
        if yr in (2015, 2025, 2026):
            out.append('<text class="val" x="%.1f" y="%.1f" text-anchor="middle">%s</text>' % (cx, top_y - 8, '%dK' % round(v / 1000)))
    out.append('<line class="axis" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (left, W - right, y(0), y(0)))
    out += ['</svg></div>', '</figure>']
    return '\n'.join(out)


def dumbbell():
    rows = [('cs.CL', 'Computation and language', 15.9, 5.7), ('cs.CV', 'Computer vision', 11.8, 9.5),
            ('cs.LG', 'Machine learning', 11.7, 8.6), ('cs.AI', 'Artificial intelligence', 5.3, 3.1),
            ('astro-ph.GA', 'Galaxies', 3.4, 1.7), ('cs.CY', 'Computers and society', 2.8, 0.8),
            ('astro-ph.EP', 'Planetary astrophysics', 2.2, 0.8), ('econ.GN', 'General economics', 1.1, 0.3),
            ('quant-ph', 'Quantum physics', 2.2, 4.2), ('math', 'All mathematics', 2.9, 16.1)]
    W = 760
    left, right, top = 270, 50, 34
    rh = 30
    H = top + rh * len(rows) + 34
    pw = W - left - right
    xmax = 18
    x = lambda v: left + v / xmax * pw
    out = ['<figure class="viz">',
           '<figcaption><strong>Share of papers summarized on SummarizePaper vs share of arXiv submissions</strong><span>Selected primary categories. SummarizePaper: 9,995 papers, Feb 2023–Oct 2026. arXiv: Mar 2023–Sep 2026.</span></figcaption>',
           '<div class="viz-legend"><span><i style="background:%s"></i>SummarizePaper</span><span><i style="background:%s"></i>arXiv</span></div>' % (BLUE, ORANGE),
           '<div class="viz-scroll"><svg viewBox="0 0 %d %d" role="img" aria-label="Dot chart comparing each category\'s share of papers summarized on SummarizePaper with its share of arXiv submissions. cs.CL is 15.9%% on SummarizePaper against 5.7%% on arXiv; mathematics is 2.9%% against 16.1%%.">' % (W, H)]
    for t in range(0, xmax + 1, 3):
        out.append('<line class="grid" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>' % (x(t), x(t), top - 10, H - 30))
        out.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%d%%</text>' % (x(t), H - 12, t))
    for i, (cat, topic, ours, arx) in enumerate(rows):
        cy = top + rh * i + rh / 2
        out.append('<text class="row" x="%d" y="%.1f" dy="4" text-anchor="end"><tspan class="cat">%s</tspan> <tspan class="topic">%s</tspan></text>' % (left - 14, cy, cat, topic))
        tip = '%s: %.1f%% of papers summarized vs %.1f%% of arXiv' % (cat, ours, arx)
        out.append('<g class="hit" data-tip="%s"><rect x="%d" y="%.1f" width="%d" height="%d" fill="transparent"/>' % (tip, 0, cy - rh / 2, W, rh))
        out.append('<line class="link" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/>' % (x(ours), x(arx), cy, cy))
        out.append('<circle class="dot" cx="%.1f" cy="%.1f" r="6" fill="%s"/>' % (x(arx), cy, ORANGE))
        out.append('<circle class="dot" cx="%.1f" cy="%.1f" r="6" fill="%s"/></g>' % (x(ours), cy, BLUE))
        if cat in ('cs.CL', 'math'):
            hi, lo = max(ours, arx), min(ours, arx)
            out.append('<text class="val" x="%.1f" y="%.1f" dy="4">%.1f%%</text>' % (x(hi) + 12, cy, hi))
            out.append('<text class="val" x="%.1f" y="%.1f" dy="4" text-anchor="end">%.1f%%</text>' % (x(lo) - 12, cy, lo))
    out += ['</svg></div>', '</figure>']
    return '\n'.join(out)


if __name__ == '__main__':
    print({'columns': columns, 'dumbbell': dumbbell}[sys.argv[1]]())
