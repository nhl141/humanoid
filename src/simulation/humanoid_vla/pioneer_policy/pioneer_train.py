import math, random, numpy as np, torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import os

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors

torch.manual_seed(42)
random.seed(42)

MODEL = "lerobot/pi05_base"
REPO_ID = "humanoid/pioneer_vla_pick_place"
ROOT = os.path.expanduser("~/humanoid/src/simulation/humanoid_vla/datasets/pioneer_vla_pick_place")
BATCH_SIZE = 8       # samples per forward pass (limited by GPU memory)
ACCUM_STEPS = 4      # 8 x 4 = 32 samples per weight update
TOTAL_STEPS = 1000   # number of weight updates
WARMUP_STEPS = 100   # ramp LR linear and slowly to not wreck weights
VAL_EVERY = 200

"""
Required when model requires sequential context lilke Diffusion Policy / ACT which expects chunks (not MLP)
Changes the tensor data that is sent to nn; when training data reads frame 500, it takes
delta_timestamp and adds/subtracts frames to get past/future frames of wrist camera, actions
Gets these new frames + old frame, combines into a single tensor  and feeds into nn
"""
delta_timestamps = {
    #gets the history of the current frame and past 2 frames to infer velocity and motion
    "observation.images.wrist_left": [-0.2, -0.1, 0.0],
    #gets current state at current timestep
    "observation.state": [0.0],
    #gets current action and 2 future action assuming 30 FPS
    #gets 2 futurecctions, future action is called action horizon -> depends on policy type. 
    #drawbacks to large action horizon: error compounds if predicting 50 steps in the future
    "action": [0.0, 0.033, 0.066]
}

# dataset info of training data
meta = LeRobotDatasetMetadata(REPO_ID, 
                              root=ROOT)
cfg = PreTrainedConfig.from_pretrained(MODEL) # gets pretrained config of pi0.5
cfg.pretrained_path = MODEL 
cfg.device = "cuda" 

policy = make_policy(cfg, ds_meta=meta)  # matches the policy's inputs to this dataset

#shuffle and split eps
episodes = list(range(meta.total_episodes)) # gets total # of ep
random.shuffle(episodes) 
val_eps, train_eps = episodes[:1], episodes[1:] # "slicing syntax", val gets x eps, train gets the rest

#creates global index mapping of frames across all episodes, concacenates videos together
delta_timestamps = resolve_delta_timestamps(policy.config, meta)
train_ds = LeRobotDataset(REPO_ID, 
                          episodes=train_eps, 
                          delta_timestamps=delta_timestamps,
                          root=ROOT)
val_ds = LeRobotDataset(REPO_ID, 
                        episodes=val_eps, 
                        delta_timestamps=delta_timestamps,
                        root=ROOT)

"""
DataLoader turns training data into batches, used in the training loop (for batches in data_loader)
shuffle randomizes frames -> this is okay for vla, data does not need to be in order because vla 
associates state with proper action, vla will learn to associate which state is next to which

Why shuffle = True:
    - Each sample contains its own future actions via action chunking
    - Without shuffling, consecutive frames are nearly identical. A batch would be 8 very similar copies, and when
    updating weights, the gradient accumulated is heavily weighted towards that image. 
    - Shuffling mixes frames from different episodes, positions, etc so each batch update
    reflects the whole dataset instead of indiviudal frames

While drop_last = True
    - lsat batch of an epoch can be smaller than BATCH_SIZE (leftover samples)
    - Without dropping, 1 image could be weighed as much as a full batch, creaeting inconsistency 

sample = dict containg tensors at a single frame
"""

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)

"""
PROCESSORS
mismatch b/w robots, humans produce vs what machines, models expect. Robots produce raw sensor data (cv images)
and joint position that needs normalization, batching before models can process. Language must be tokenized and
coordinate systems need standardization for robots

Model outputs often need normalization and conversion to real world scales.
Cross domain translation adds complexity; training data from one setup cannot adapt to different hardware. 
Processors serve as universal translators that bridge these gapos, ensuring data flows from 
sensors to models to actuators. Processors handle preprocessing, postprocessing to convert raw env data
into model ready inputs & vice versa

EnvTransition: universal data container: dictioniary that represents complete robot environment interactions:
    - observation: sensor data (images, state, proprioception)
    - action: to execute or was executed
    - reward: rl signal
    - done/truncated: episode boundary indicators
    - info: arbitrary metadata

- dataset.meta.stats: when using LeRobotDataset, creates folder on drive called meta/ containg stats.json
files track stats like mean, standard deviation, dataset.meta.stats pull from this folder

"""
preprocess, postprocess = make_pre_post_processors(policy.config, dataset_stats=train_ds.meta.stats)

#freeze backbone
TRAIN_PREFIXES = (
    "model.paligemma_with_expert.gemma_expert.",  # action expert transformer
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.time_mlp_in.",
    "model.time_mlp_out.",
)

trainable_params = []
for name, p in policy.named_parameters():
    if name.startswith(TRAIN_PREFIXES):
        p.requires_grad = True
        trainable_params.append(p)
    else:
        p.requires_grad = False

optimizer = AdamW(trainable_params, lr=policy.config.optimizer_lr)

#define scheduler for learning rate, cosine decay
#learning rate defines how fast you decend down the loss curve, too high can overshoot / make loss oscillate
def lr_factor(step):
    #linear warmup for settling weights
    if step < WARMUP_STEPS:
        return (step+1) / WARMUP_STEPS                        # 0 -> 1
    #cosine decay after warmuo
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * progress))       # 1 -> 0
scheduler = LambdaLR(optimizer, lr_factor)

#validation function
@torch.no_grad()
def validate():
    policy.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if i == 10:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(preprocess(batch))
        losses.append(loss.item())
    policy.train()   # easy to forget
    return sum(losses) / len(losses)

policy.train()
step = 0        # weight updates
running_loss = 0
micro = 0       # batches seen
done = False

while not done:
    for batch in train_loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(preprocess(batch))

        # gradients add up over ACCUM_STEPS batches, so divide to keep them the right size
        (loss / ACCUM_STEPS).backward()
        running_loss += loss.item() / ACCUM_STEPS
        micro += 1
        if micro % ACCUM_STEPS != 0:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        if step % 10 == 0:
            print(f"step {step} | loss { running_loss/10:.4f} | grad {grad_norm:.2f} | lr {scheduler.get_last_lr()[0]:.2e}")
            running_loss = 0
        if step % VAL_EVERY == 0:
            print(f"step {step} | val_loss {validate():.4f}")
        if step >= TOTAL_STEPS:
            done = True
            break

# ---------- 9. save ----------
policy.save_pretrained("checkpoint")
preprocess.save_pretrained("checkpoint")
postprocess.save_pretrained("checkpoint")


