from abc import ABC ,abstractmethod

class BaseScheduler(ABC):
    """
        PROPERTIES:
            NONE
        TASK:
            Maintain the entire work flow
        RETURN:
            None
    """
    
    @abstractmethod
    def run():
        pass