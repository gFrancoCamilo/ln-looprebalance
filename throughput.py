import networkx as nx
import time
import threading
from topology import *

"""
Defines a TimeWindow class that is used to compute the throughput that
passes through a node in the network. The idea is to use this information
to perform a demand-based rebalance.
"""
class TimeWindow:
    def __init__(self, event: threading.Event, max_size: int = 5) -> None:
        
        """Defines the current size of the time window"""
        self.size = max_size

        """Defines the maximum size of the time window"""
        self.max_size = max_size

        """Stores the values of each payment that has arrived at a certain node"""
        self.payment_values = []

        """Stores the throughput in satoshis/s in the current time window"""
        self.throughput = 0

        """Runs the window routine as a thread"""
        thread = threading.Thread(target = self.window_routine, args=(event,))
        thread.start()
    
    def window_routine (self, event):
        while True:

            """Check if payment arrived"""
            self.throughput = self.compute_throughput()

            """Check again next window"""
            time.sleep(self.size)

            """Check for stop"""
            if event.is_set():
                break
    
    """Computes throughput of the node being analyzed"""
    def compute_throughput (self) -> int:
        if len(self.payment_values) == 0:
            return 0

        accumulator = 0
        for payment in self.payment_values:
            accumulator += payment

        self.payment_values.clear()
        return round(accumulator/self.size)