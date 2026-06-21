import os, time
import pandas as pd
from drugex.training.monitors import FileMonitor

class TimedFileMonitor(FileMonitor):
    """
    FileMonitor that stops training after `time_budget` seconds
    counted from the moment the first molecules are written
    """
    def __init__(self, path, time_budget, **kwargs):
        super().__init__(path, **kwargs)
        self.time_budget = time_budget
        self.t0 = None              
        self._stop = False

    #overwrite the molecule-saving hook 
    def saveMolecules(self, df):
        super().saveMolecules(df)

        #start timer
        if self.t0 is None:
            self.t0 = time.time()

        #if stop
        if time.time() - self.t0 > self.time_budget:
            self._stop = True                 
            raise TimeoutError(               
                f"Time budget ({self.time_budget}s) exhausted; "
                "early-stopping RL loop."
            )

    # helper the main script can query after fit() returns
    def timed_out(self):
        return self._stop