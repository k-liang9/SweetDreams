import copy
import torch
import torch.nn as nn
from torch import optim
import wandb
import tqdm

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, x, target):
        raise NotImplementedError()
    
    def featurize(self, batch):
        raise NotImplementedError()
    
    def compute_metrics(self, split, out, x, target):
        raise NotImplementedError()
    
    def _prepare_metrics_for_log(self, metrics):
        log = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.detach()
                if value.numel() == 1:
                    value = value.item()
            log[key] = value
        return log
    
    def _aggregate_metrics(self, metrics_list):
        if not metrics_list:
            return {}
        
        aggregated = {}
        keys = metrics_list[0].keys()
        for key in keys:
            values = [metrics[key] for metrics in metrics_list if key in metrics]
            if not values:
                continue
            
            first = values[0]
            if isinstance(first, (int, float)):
                aggregated[key] = sum(values) / len(values)
            elif torch.is_tensor(first) and first.numel() == 1:
                aggregated[key] = torch.stack([value.detach() for value in values]).mean()
            else:
                aggregated[key] = first
        
        return aggregated
        
    def run_train(self, train_loader, val_loader, test_loader):
        cfg = self.cfg
        torch.manual_seed(cfg.params.seed)
        optimizer = optim.Adam(self.parameters(), cfg.params.lr)
        
        with wandb.init(project=cfg.exp.project, name=cfg.exp.run_name, config=dict(cfg.params)) as run:
            best_model_state = None
            tot_steps = 0
            for epoch in range(cfg.params.epochs):
                self.train()
                for step, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc='train batches'):
                    x, target = self.featurize(batch)
                    out = self.forward(x, target)
                    optimizer.zero_grad()
                    out['loss'].backward()
                    optimizer.step()
                    
                    tot_steps += 1
                    metrics = self.compute_metrics('train', out, x, target)
                    run.log(self._prepare_metrics_for_log(metrics), step=tot_steps)
                    
                self.eval()
                with torch.no_grad():
                    val_metrics = []
                    for _, batch in tqdm.tqdm(enumerate(val_loader), total=len(val_loader), desc='val batches'):
                        x, target = self.featurize(batch)
                        out = self.forward(x, target)
                        val_metrics.append(self.compute_metrics('val', out, x, target))
                    
                    metrics = self._aggregate_metrics(val_metrics)
                    run.log(self._prepare_metrics_for_log(metrics), step=tot_steps)
                    
                    val_loss = metrics.get('val/loss')
                    if torch.is_tensor(val_loss):
                        val_loss = val_loss.item()
                    if val_loss is not None and (best_model_state is None or val_loss < best_model_state['val_loss']):
                        best_model_state = {
                            'epoch': epoch,
                            'val_loss': val_loss,
                            'model_state_dict': copy.deepcopy(self.state_dict()),
                            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        }
            
            if best_model_state is not None:
                checkpoint_path = f'{run.dir}/best_model.pt'
                torch.save(best_model_state, checkpoint_path)
                wandb.save(checkpoint_path)
            
            self.eval()
            with torch.no_grad():
                test_metrics = []
                for _, batch in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc='test batches'):
                    x, target = self.featurize(batch)
                    out = self.forward(x, target)
                    test_metrics.append(self.compute_metrics('test', out, x, target))
                
                metrics = self._aggregate_metrics(test_metrics)
                run.log(self._prepare_metrics_for_log(metrics), step=tot_steps)
