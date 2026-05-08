import torch
import torch.nn as nn
from torch import optim
import wandb
import tqdm

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, feat):
        raise NotImplementedError()
    
    def featurize(self, batch):
        raise NotImplementedError()
    
    def compute_and_log_metrics(self, pred, feat):
        raise NotImplementedError()
    
    def run_train(self, train_loader, val_loader, test_loader):
        cfg = self.cfg
        torch.manual_seed(cfg.params.seed)
        optimizer = optim.Adam(self.parameters(), cfg.params.lr)
        
        with wandb.init(project=cfg.exp.project, name=cfg.exp.run_name, config=dict(cfg.params)) as run:
            best_model_state = None
            for epoch in range(cfg.params.epochs):
                self.train()
                for step, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc='train batches'):
                    feat = self.featurize(batch)
                    out = self.forward(feat)
                    optimizer.zero_grad()
                    out['loss'].backward()
                    optimizer.step()
                    
                    self.compute_and_log_metrics(out['pred'], feat)
                    
                self.eval()
                with torch.no_grad():
                    for _, batch in tqdm.tqdm(enumerate(val_loader), total=len(val_loader), desc='val batches'):
                        feat = self.featurize(batch)
                        out = self.forward(feat)
                        
                        self.compute_and_log_metrics(out['pred'], feat)
            
            if best_model_state is not None:
                # TODO: save best model
                pass
            
            self.eval()
            with torch.no_grad():
                for _, batch in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc='test batches'):
                    feat = self.featurize(batch)
                    out = self.forward(feat)
                    
                    self.compute_and_log_metrics(out['pred'], feat)