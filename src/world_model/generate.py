# Rollout / inference for the world model.
#
# Mental model
# ------------
# The transformer does NOT produce frames; it produces frame-token logits
# (see world_model.py: frame_head -> num_frame_tokens). The VQ-VAE decoder is
# what turns tokens back into pixels.
#
# It's not "one token per frame" -- it's cfg.model.tokens_per_frame = 16 tokens
# (from the 4x4 latent grid) per frame, sampled autoregressively within a
# frame, then an action is appended, then the next 16 tokens, etc.
#
# The interleaved sequence layout (from embeddings.py) is:
#     f0, a0, f1, a1, f2, a2, ..., f_{T-1}
# where each f_t is N tokens (z_t_0 ... z_t_{N-1}) and a_t is one action token.
# a_t sits AFTER f_t. Semantically: "observed f_t, took a_t, now predict f_{t+1}".
# This matches the dataset's (frame_t, action_t) pairing and matches the
# training target in losses.py: the position holding a_{t-1} predicts the
# first token of f_t.
#
# "Actions from the dataset" is the right call for a deterministic rollout --
# that's the standard setup so imagined frames can be compared against
# ground-truth frames at the same actions. (One could also drive it with a
# policy or random actions; the dataset version is what shows IRIS-style
# imagination quality.)
#
# Sliding window: compute_max_seq_len caps context at
#     seq_len * tokens_per_frame + (seq_len - 1)
# Long rollouts must trim the prefix to fit -- see the comment in embeddings.py
# ("rollout uses a sliding window of the same size").
#
#
# What this file needs to contain
# -------------------------------
# A @torch.no_grad() rollout function with these steps:
#
# 1. Load weights:
#    - VQ-VAE checkpoint (same pattern as train_world_model.load_tokenizer).
#    - Trained world-model checkpoint.
#    Both .eval(), on `device`.
#
# 2. Get a prompt batch:
#    - Pull (frames, actions, _) from AtariEpisodeDataset.
#    - Split into a prompt (first k frames) and a future (rollout_steps frames).
#    - Keep the future frames around as ground truth for side-by-side viewing.
#
# 3. Tokenize the prompt:
#       prompt_tokens = tokenizer.encode(prompt_frames).flatten(2)   # (B, k, N)
#
# 4. Autoregressive loop for each of `rollout_steps` new frames t:
#
#    Prompt state at the start of step t:
#      - frame_tokens shape (B, t, N)   -- t known frames so far
#      - actions      shape (B, t - 1)  -- one fewer action than frames
#    (For the very first generated frame, t = k, so we have the k prompt frames
#    and k-1 prompt actions.)
#
#    a. Append actions[t - 1] from the dataset (the action that was actually
#       taken at frame_{t-1}). This is the action that, in the interleaved
#       sequence, sits between f_{t-1} and f_t.
#         actions shape becomes (B, t).
#
#    b. Allocate a new (B, 1, N) slot for frame t. Then for each token
#       position n = 0 .. N-1:
#
#         - Build the interleaved sequence so far. Cleanest option: feed
#           frame_tokens=(B, t+1, N) and actions=(B, t) through
#           world_model.embeddings, then slice the resulting embedded
#           sequence to end right before the unfilled token. Easier
#           alternative: skip the embeddings module here and build the
#           sequence manually by indexing frame_token_embedding and
#           action_embedding directly, then add positional embeddings.
#
#         - Run transformer(x) + frame_head, take the LAST logit only.
#         - Apply temperature, top_k (from cfg.generate), torch.multinomial
#           -> sampled token id.
#         - Fill that position in the new frame.
#
#    c. Sliding window: if total interleaved length would exceed
#       max_seq_len, drop the oldest frame + its trailing action from
#       frame_tokens / actions before the next iteration.
#
# 5. Decode:
#       all_tokens shape (B, k + rollout_steps, 4, 4)
#       frames = tokenizer.decode_from_indices(all_tokens)   # (B, T, 3, 64, 64)
#
# 6. Log:
#    - Stack imagined vs. ground-truth side-by-side and log as wandb.Video
#      (uint8, shape (T, C, H, W*2), fps ~15).
#    - Optionally also save a local GIF with imageio.
#
#
# Things to watch out for
# -----------------------
# - No KV cache yet. transformer.py has "TODO: add kv cache" on lines 42 and 77.
#   A naive rollout re-runs the full sequence every token; for
#   rollout_steps=32 * 16 tokens = 512 forward passes per sample it's fine on
#   a small model, just slow. Don't add KV caching as part of this file;
#   it's its own task.
#
# - Causal mask size. The registered self.mask is sized to max_seq_len. As
#   long as the window is slid correctly it won't trip the size check in
#   transformer.SelfAttention.forward.
#
# - First action shape contract. embeddings.forward requires actions to be
#   (B, T - 1), so with t+1 frames in the sequence pass t actions. The action
#   at index t-1 is the one that conditions generation of frame t.
#
# - Determinism for debugging. Add a --greedy / temperature=0 path (argmax)
#   so the model can be sanity-checked without sampling noise.
