import time


class Record(object):
    def __init__(self, name=''):
        self._last = time.time()
        self._acc = 0.0
        self._name = name if name else 'total'

    def on(self):
        self._last = time.time()

    def off(self):
        self._acc += time.time() - self._last

    def show(self):
        print('%s took %.3fs' % (self._name, self._acc))


class StopWatch(object):
    def __init__(self, name=''):
        self._name = name if name else 'total'
        self._t0 = time.time()
        self._last = self._t0
        self._rec = {}

    def tick(self, name):
        t = time.time()
        d = t - self._last
        self._last = t
        print('%s took %.3fs' % (name, d))

    def turn_on(self, name):
        if name not in self._rec:
            self._rec[name] = Record(name)
        self._rec[name].on()

    def turn_off(self, name):
        self._rec[name].off()

    def __del__(self):
        for rec in self._rec:
            self._rec[rec].show()
        t = time.time()
        d = t - self._t0
        print('%s took %.3fs' % (self._name, d))
