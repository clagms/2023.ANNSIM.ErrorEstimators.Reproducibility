import time
import unittest


class TimedTest(unittest.TestCase):
    def setUp(self):
        self.start_time = time.time()

    def tearDown(self):
        t = time.time() - self.start_time
        print('%s: %.3f' % (self.id(), t))

