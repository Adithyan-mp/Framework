from computation.interface.base_trainer import BaseTrainer
from computation.interface.base_model import BaseModel


class LocalTrainer(BaseTrainer):
    def __init__(self,dataloader,loss,optim):
        self.loss_fn = loss
        self.dataloader=dataloader
        self.optim = optim
        
    def train(self, local_epoch, model:BaseModel):
        model.train()
        
        for i in range(local_epoch):
            total_loss = 0.0
            batch = 0
            for x,y in self.dataloader:
                self.optim.zero_grad()
                pred = model(x)
                loss = self.loss_fn(pred,y)
                loss.backward()
                self.optim.step()
                
                total_loss +=loss.item()
                batch+=1
            if batch == 0:
                raise ValueError(f"Dataloader is empty")
            avg_loss = total_loss/batch
                
            print(f"local_epoch {i+1} avg_loss : {avg_loss:.4f}")
                
        