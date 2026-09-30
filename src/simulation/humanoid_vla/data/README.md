# Data — what preprocessing is, and which parts are yours

**Preprocessing = every transformation between what the recorder wrote to disk
and the tensors the model sees.** If it changes the numbers, it's
preprocessing. If it only looks at them and reports, it isn't — that's
validation, and it lives in `validate.py`.

## Every preprocessing technique in a VLA pipeline

There are twelve. You are responsible for two.

### On the action / state vectors

| # | Technique | What it does | Who |
|---|---|---|---|
| 1 | **Gripper dimensionality reduction** | 8 joints → 7; the `joint7l`/`joint8l` mimic pair becomes one number | **you** (`gripper.py`) |
| 2 | Normalization | rescale each joint to [-1, 1] using q01/q99 | LeRobot |
| 3 | Dimension padding | your 7 dims → π0.5's `max_state_dim = 32` | LeRobot |
| 4 | Action chunking | stack the next 50 actions as the target | LeRobot (`delta_timestamps`) |
| 5 | End-of-episode padding + mask | repeat the last action, emit `action_is_pad` | LeRobot |
| 6 | Absolute → delta actions | predict change instead of target | optional, probably never |

### On the images

| # | Technique | What it does | Who |
|---|---|---|---|
| 7 | Video decode | MP4 → frames | LeRobot |
| 8 | Resize | 480×640 → 224×224 for the vision encoder | LeRobot processor |
| 9 | Encoder normalization | the backbone's own mean/std | LeRobot processor |
| 10 | Augmentation | colour jitter, small crops — never a flip | LeRobot (`image_transforms=`) |

### On the instruction text

| # | Technique | What it does | Who |
|---|---|---|---|
| 11 | Tokenization | text → token ids | LeRobot processor |
| 12 | **Writing a real instruction** | per-episode text naming object and destination | **you** (at record time) |

That's the whole landscape. Ten of twelve are already built; you configure them,
you don't write them.

## So: your two preprocessing techniques

1. **Collapse the gripper** (#1). Do it in `src/il` at record time so datasets
   are born 7-dim, and there is no preprocessing step at all. `gripper.py` still
   exists because deployment needs `expand()` to go back to 8 for
   `joint_command`, and because your existing 8-dim recordings need `collapse()`
   until you re-record.
2. **Write real task strings** (#12). The recorder default is the constant
   `"humanoid teleop demonstration"`. A constant string is not language — the
   model learns to ignore the text input entirely. Fix it in
   `src/il/config/record_defaults.yaml`. This one cannot be fixed later.

## Not preprocessing, but do it anyway

- **Episode filtering** — drop failed demos. This is data *curation*: you're
  choosing which episodes exist, not transforming any of them. `pick_place_gen`
  already gates on success; teleop episodes you score as you go.
- **Validation** (`validate.py`) — four checks that transform nothing and tell
  you whether the data means what you think. The replay check is worth more than
  the other three combined.

## Files here

    gripper.py    technique #1. The only transform that is yours.
    validate.py   four checks. Not preprocessing.

## Read these, in this order

- `lerobot/datasets/lerobot_dataset.py` — `delta_timestamps`, `episodes`,
  `image_transforms`. Covers techniques 4, 5, 7, 10.
- `lerobot/datasets/compute_stats.py` — technique 2. `DEFAULT_QUANTILES =
  [0.01, 0.10, 0.50, 0.90, 0.99]`.
- `lerobot/policies/pi05/configuration_pi05.py` — `chunk_size`,
  `max_state_dim`, `normalization_mapping`. Techniques 2–4 are config here.
- `lerobot/policies/act/modeling_act.py` — ~500 lines, the clearest view of what
  a policy does with a batch.

## The one trap worth knowing before you start

`observation.state` and `action` are both absolute joint angles in radians, one
25 Hz step apart, so they are nearly identical numbers. The easiest thing a
policy can learn is to copy proprioception to its output — scoring a great loss
while ignoring both cameras. Chunking blunts it (copying the current pose
explains `action_t`, not `action_{t+49}`). Confirm with the ablation: run eval
once with proprioception zeroed. If success barely drops, vision was never
being used.
