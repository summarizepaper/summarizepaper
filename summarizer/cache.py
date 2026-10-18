import threading

from django.core.cache.backends.redis import RedisCache, RedisCacheClient

#Under ASGI, django.core.cache.caches is context-local: every request builds a new
#RedisCache, hence a new connection pool, and close() is a no-op -> connections pile up
#until the plan's client limit ("max number of clients reached"). Share pools per process.
_pools = {}
_pools_lock = threading.Lock()


class SharedPoolRedisCacheClient(RedisCacheClient):
    def _get_connection_pool(self, write):
        url = self._servers[self._get_connection_pool_index(write)]
        with _pools_lock:
            if url not in _pools:
                _pools[url] = self._pool_class.from_url(url, **self._pool_options)
            return _pools[url]


class SharedPoolRedisCache(RedisCache):
    def __init__(self, server, params):
        super().__init__(server, params)
        self._class = SharedPoolRedisCacheClient
