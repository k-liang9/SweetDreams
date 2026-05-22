
class LayerCache:
    def __init__(self, cfg):
        pass

class KVCache:
    def __init__(self, cfg):
        self.n_heads = cfg
        self.max_seq = compute_max_seq(cfg)
        self.dim = cfg.model.d_model
        self.B = cfg.generate.batch_size
        self.k_cache = 0
        self.v_cache = 0
        
    def allocate(self, B, max_seq, dtype, device):
        pass
    
    def append(self, layer_idx, k_new, v_new):
        pass
    
    def read(self, layer_idx):
        pass
    
    def reset(self):
        pass