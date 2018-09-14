"""
Copyright (c) Zacker
Hawk I
"""

import visdom
import numpy as np

class Visualizer(visdom.Visdom):
    def __init__(self, env='default', **kwargs):
        visdom.Visdom.__init__(self, env=env, use_incoming_socket=False, **kwargs)
        self.index = {}

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.line(Y=np.array([y]), X=np.array([x]),
                  win=name,
                  opts=dict(title=name),
                  update=None if x == 0 else 'append',
                  **kwargs
                  )
        self.index[name] = x + 1