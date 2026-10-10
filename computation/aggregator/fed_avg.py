
from computation.interface.base_aggregator import BaseAggregator


class FedAvg(BaseAggregator):
    def __init__(self):
        self.avg_weights = None
        self.num_client = 0
        self.n = None

    def aggregate(self, weight):
        if weight is None or len(weight) == 0:
            raise ValueError("Invalid weight given")

        if self.num_client == 0:
            self.avg_weights = [
                w.detach().clone() for w in weight
            ]
            self.n = len(weight)
            self.num_client = 1
            return

        if len(weight) != self.n:
            raise ValueError("Mismatch in number of parameters")

        for avg, w in zip(self.avg_weights, weight):
            if avg.shape != w.shape:
                raise ValueError("Shape mismatch between client parameters")

        for avg, w in zip(self.avg_weights, weight):
            avg.add_(w)

        self.num_client += 1

    def finalize(self):
        if self.num_client == 0:
            raise ValueError("No weights have been averaged")

        for w in self.avg_weights:
            w.div_(self.num_client)

        return self.avg_weights
    
    def reset(self):
        self.avg_weights = None
        self.num_client = 0
        self.n = None 
        
        
