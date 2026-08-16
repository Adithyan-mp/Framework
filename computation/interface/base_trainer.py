from abc import ABC,abstractmethod

class BaseTrainer(ABC):
    """
        PROPERTIES:
            Model,Dataset,Optimizer and loss function
        TASK:
            Perform local training 
        RETURN:
            Loss  
    """
    
    @abstractmethod
    def train(self,local_epoch:int) -> float :
        pass