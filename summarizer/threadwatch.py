"""Log, every few minutes, how many threads the web process has and where they wait.

The web dyno ran out of threads ("can't start new thread") after hours of
traffic: each line here groups the threads by the code they are stuck in, so
the logs show what piles up. Started once from asgi.py.
"""
import collections
import logging
import sys
import threading
import time
import traceback

logger = logging.getLogger(__name__)

EVERY = 300
ALERT = 150  # threads; about 20 normally, and the dyno failed at a few hundred


def where(frame):
    """Where a thread waits: its innermost frame, and the innermost frame of our own code when there is one."""
    stack = traceback.extract_stack(frame)
    ours = [f for f in stack if '/summarizer/' in f.filename or '/arxivsummarizer/' in f.filename]
    places = [stack[-1]] + ([ours[-1]] if ours and ours[-1] is not stack[-1] else [])
    return ' < '.join('%s:%s %s' % (f.filename.rsplit('/', 2)[-1], f.lineno, f.name) for f in places)


def report():
    frames = sys._current_frames()
    counts = collections.Counter(where(frames[t.ident]) for t in threading.enumerate() if t.ident in frames)
    total = threading.active_count()
    places = '; '.join('%d x %s' % (n, place) for place, n in counts.most_common(8))
    if total >= ALERT:
        logger.error('threadwatch: %d threads, the web process is piling them up; %s', total, places)  # Sentry
    else:
        logger.warning('threadwatch: %d threads; %s', total, places)


def loop():
    while True:
        time.sleep(EVERY)
        try:
            report()
        except Exception:
            logger.exception('threadwatch failed')


def start():
    threading.Thread(target=loop, name='threadwatch', daemon=True).start()
