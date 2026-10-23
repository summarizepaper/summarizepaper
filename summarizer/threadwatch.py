"""Watch the web process: its threads and its database pool, logged every few minutes, and a restart when stuck.

The web dyno used to block once a day: worker threads kept database
connections they never gave back (see sources.in_thread), the pool ran dry,
every request then waited for a connection while holding a thread, and the
process ran out of threads ("can't start new thread"); it served 500s until
the next restart. Each log line groups the threads by where they wait and
gives the pool's use. When the process stays stuck (too many threads, or a
pool with requests waiting and no free connection) for STUCK_MINUTES, it
alerts Sentry and stops itself: Heroku starts a fresh one in seconds.
"""
import collections
import logging
import os
import signal
import sys
import threading
import time
import traceback

from django.db import connections

logger = logging.getLogger(__name__)

CHECK = 60  # seconds
REPORT_EVERY = 5  # checks: a log line every 5 minutes
ALERT = 150  # threads; about 20 normally
STUCK_THREADS = 200
STUCK_MINUTES = 10


def where(frame):
    """Where a thread waits: its innermost frame, and the innermost frame of our own code when there is one."""
    stack = traceback.extract_stack(frame)
    ours = [f for f in stack if '/summarizer/' in f.filename or '/arxivsummarizer/' in f.filename]
    places = [stack[-1]] + ([ours[-1]] if ours and ours[-1] is not stack[-1] else [])
    return ' < '.join('%s:%s %s' % (f.filename.rsplit('/', 2)[-1], f.lineno, f.name) for f in places)


def pool_stats():
    """(connections in use, pool size, max size, requests waiting), or None without a pool (tests, sqlite)."""
    pool = getattr(connections['default'], 'pool', None)
    if pool is None:
        return None
    stats = pool.get_stats()
    size = stats.get('pool_size', 0)
    return size - stats.get('pool_available', 0), size, pool.max_size, stats.get('requests_waiting', 0)


def report(level=logging.WARNING, note=''):
    frames = sys._current_frames()
    counts = collections.Counter(where(frames[t.ident]) for t in threading.enumerate() if t.ident in frames)
    pool = pool_stats()
    logger.log(level, 'threadwatch: %d threads; pool %s%s; %s', threading.active_count(),
               '%d/%d in use (max %d), %d waiting' % pool if pool else 'none', note,
               '; '.join('%d x %s' % (n, place) for place, n in counts.most_common(8)))


def stuck():
    """True when the process cannot serve: too many threads, or requests waiting for a pool with no free connection."""
    if threading.active_count() >= STUCK_THREADS:
        return True
    pool = pool_stats()
    return bool(pool) and pool[0] >= pool[2] and pool[3] > 0


def restart():
    """Stop the process: daphne exits on SIGTERM, and Heroku starts a new dyno process."""
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(30)
    os._exit(1)  # SIGTERM did not end it


def tick(state):
    """One check; state is {'checks', 'stuck_for'}, kept between checks."""
    state['checks'] += 1
    state['stuck_for'] = state['stuck_for'] + 1 if stuck() else 0
    if state['stuck_for'] * CHECK >= STUCK_MINUTES * 60:
        report(logging.ERROR, ', stuck for %d minutes: restarting' % STUCK_MINUTES)  # Sentry
        restart()
    elif threading.active_count() >= ALERT:
        report(logging.ERROR, ', the web process is piling threads up')  # Sentry
    elif state['checks'] % REPORT_EVERY == 0:
        report()


def loop():
    state = {'checks': 0, 'stuck_for': 0}
    while True:
        time.sleep(CHECK)
        try:
            tick(state)
        except Exception:
            logger.exception('threadwatch failed')


def start():
    threading.Thread(target=loop, name='threadwatch', daemon=True).start()
