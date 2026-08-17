from abc import ABC,abstractmethod

class BaseDataLoader(ABC):
    @abstractmethod
    def __iter__(self):
        pass
    
    @abstractmethod
    def __len__(self):
        pass