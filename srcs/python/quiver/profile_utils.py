import time


class StopWatch(object):
    def __init__(self, name=''):
        self._name = name if name else 'total'
        self._t0 = time.time()
        self._last = self._t0

    def tick(self, name):
        t = time.time()
        d = t - self._last
        self._last = t
        print('%s took %.3fs' % (name, d))

    def __del__(self):
        t = time.time()
        d = t - self._t0
        print('%s took %.3fs' % (self._name, d))
