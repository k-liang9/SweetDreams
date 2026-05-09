stale
```
atari-world-model/
│
├── README.md
├── requirements.txt
├── setup.py                  # makes the package pip-installable
│
├── configs/
│   ├── vqvae.yaml            # tokenizer hyperparams
│   ├── transformer.yaml      # world model hyperparams
│   └── benchmark.yaml        # kernel benchmark configs
│
├── data/
│   ├── collect.py            # Gymnasium rollout collection script
│   └── dataset.py            # PyTorch Dataset class for (frame, action) sequences
│
├── src/
│   ├── __init__.py
│   ├── tokenizer/
│   │   └── vqvae.py          # encoder, decoder, codebook
│   ├── world_model/
│   │   ├── transformer.py    # GPT-style architecture
│   │   ├── embeddings.py     # token + action + positional embeddings
│   │   └── generate.py       # imagination / autoregressive sampling
│   └── kernels/
│       ├── fused_cross_entropy.py   # Triton kernel
│       └── benchmark.py             # benchmarking harness
│
├── train/
│   ├── train_vqvae.py        # stage 1 training script
│   ├── train_world_model.py  # stage 2 training script
│   └── utils.py              # logging, checkpointing, lr scheduling
│
├── eval/
│   ├── visualize.py          # generate rollout GIFs
│   ├── fid.py                # reconstruction quality metrics
│   └── codebook_util.py      # codebook utilization diagnostics
│
└── notebooks/
    ├── explore_data.ipynb    # sanity check frames + actions
    └── benchmark_results.ipynb  # plot kernel speedup figures
```