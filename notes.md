# Training emp_small for EMPD

```
python train.py \
    data_root=TODO \
    model=emp_small batch_size=96 \
    monitor=val_minFDE6 model.target.decoder=detr
```

# Running eval.py for EMPD

Go to `emp/` directory, and then run:

For EMPD:
```
python3 eval.py \
    data_root=TODO \
    batch_size=32 \
    checkpoint=TODO
```


python3 eval.py \
    data_rootTODO \
    batch_size=32 \
    checkpoint=TODO


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      Validate metric      ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       val/reg_loss        │    0.21391896903514862    │
│          val_MR           │    0.18467052280902863    │
│     val_brier-minFDE6     │     2.004626989364624     │
│        val_minADE1        │    1.7357546091079712     │
│        val_minADE6        │    0.7110224366188049     │
│        val_minFDE1        │     4.332010269165039     │
│        val_minFDE6        │    1.3871904611587524     │
└───────────────────────────┴───────────────────────────┘

For EMPM:
- On `src/model/trainer_forecast.py`, adjust the constructor parameter to `decoder: str = "mlp"`
- Then run:
```
python3 eval.py \
    data_root=TODO \
    batch_size=32 \
    checkpoint=TODO \
```

results:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      Validate metric      ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       val/reg_loss        │    0.22536510229110718    │
│          val_MR           │    0.19092966616153717    │
│     val_brier-minFDE6     │    2.0978446006774902     │
│        val_minADE1        │    1.7946380376815796     │
│        val_minADE6        │    0.7303645014762878     │
│        val_minFDE1        │     4.514750957489014     │
│        val_minFDE6        │     1.456929087638855     │
└───────────────────────────┴───────────────────────────┘
```

# notes on a scenario

One scenario has:
- metadata:
    - scenario ID
    - city name
- track data/dynamic actors, where each agent/actor has a track:
    - Track ID (unique ID for each actor)
    - object type: vehicle, bus, pedestrian, motorcyclist, cyclist
    - 2D positions: Sequence of (x, y) positions at each timestamp
    - Heading (yaw): orientation angle at each timestamp
    - Velocity: Usually (vx, vy) or speed + heading
    - Observed mask: flags indicating whether actor is visible at each time step
- focal agent (only one agent):
    - fully observed for entire scenario duration (11 seconds)
- scored actors


# param count:

emp_small: 338,225
emp: 3,158,001

net.actor_type_embed torch.Size([4, 128]) 512 True
net.lane_type_embed torch.Size([1, 1, 128]) 128 True
net.h_proj.weight torch.Size([128, 5]) 640 True
net.h_proj.bias torch.Size([128]) 128 True
net.h_embed.0.norm1.weight torch.Size([128]) 128 True
net.h_embed.0.norm1.bias torch.Size([128]) 128 True
net.h_embed.0.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.h_embed.0.attn.in_proj_bias torch.Size([384]) 384 True
net.h_embed.0.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.h_embed.0.attn.out_proj.bias torch.Size([128]) 128 True
net.h_embed.0.norm2.weight torch.Size([128]) 128 True
net.h_embed.0.norm2.bias torch.Size([128]) 128 True
net.h_embed.0.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.h_embed.0.mlp.fc1.bias torch.Size([512]) 512 True
net.h_embed.0.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.h_embed.0.mlp.fc2.bias torch.Size([128]) 128 True
net.h_embed.1.norm1.weight torch.Size([128]) 128 True
net.h_embed.1.norm1.bias torch.Size([128]) 128 True
net.h_embed.1.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.h_embed.1.attn.in_proj_bias torch.Size([384]) 384 True
net.h_embed.1.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.h_embed.1.attn.out_proj.bias torch.Size([128]) 128 True
net.h_embed.1.norm2.weight torch.Size([128]) 128 True
net.h_embed.1.norm2.bias torch.Size([128]) 128 True
net.h_embed.1.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.h_embed.1.mlp.fc1.bias torch.Size([512]) 512 True
net.h_embed.1.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.h_embed.1.mlp.fc2.bias torch.Size([128]) 128 True
net.h_embed.2.norm1.weight torch.Size([128]) 128 True
net.h_embed.2.norm1.bias torch.Size([128]) 128 True
net.h_embed.2.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.h_embed.2.attn.in_proj_bias torch.Size([384]) 384 True
net.h_embed.2.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.h_embed.2.attn.out_proj.bias torch.Size([128]) 128 True
net.h_embed.2.norm2.weight torch.Size([128]) 128 True
net.h_embed.2.norm2.bias torch.Size([128]) 128 True
net.h_embed.2.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.h_embed.2.mlp.fc1.bias torch.Size([512]) 512 True
net.h_embed.2.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.h_embed.2.mlp.fc2.bias torch.Size([128]) 128 True
net.h_embed.3.norm1.weight torch.Size([128]) 128 True
net.h_embed.3.norm1.bias torch.Size([128]) 128 True
net.h_embed.3.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.h_embed.3.attn.in_proj_bias torch.Size([384]) 384 True
net.h_embed.3.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.h_embed.3.attn.out_proj.bias torch.Size([128]) 128 True
net.h_embed.3.norm2.weight torch.Size([128]) 128 True
net.h_embed.3.norm2.bias torch.Size([128]) 128 True
net.h_embed.3.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.h_embed.3.mlp.fc1.bias torch.Size([512]) 512 True
net.h_embed.3.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.h_embed.3.mlp.fc2.bias torch.Size([128]) 128 True
net.lane_embed.first_conv.0.weight torch.Size([128, 3, 1]) 384 True
net.lane_embed.first_conv.0.bias torch.Size([128]) 128 True
net.lane_embed.first_conv.1.weight torch.Size([128]) 128 True
net.lane_embed.first_conv.1.bias torch.Size([128]) 128 True
net.lane_embed.first_conv.3.weight torch.Size([256, 128, 1]) 32768 True
net.lane_embed.first_conv.3.bias torch.Size([256]) 256 True
net.lane_embed.second_conv.0.weight torch.Size([256, 512, 1]) 131072 True
net.lane_embed.second_conv.0.bias torch.Size([256]) 256 True
net.lane_embed.second_conv.1.weight torch.Size([256]) 256 True
net.lane_embed.second_conv.1.bias torch.Size([256]) 256 True
net.lane_embed.second_conv.3.weight torch.Size([128, 256, 1]) 32768 True
net.lane_embed.second_conv.3.bias torch.Size([128]) 128 True
net.pos_embed.0.weight torch.Size([128, 4]) 512 True
net.pos_embed.0.bias torch.Size([128]) 128 True
net.pos_embed.2.weight torch.Size([128, 128]) 16384 True
net.pos_embed.2.bias torch.Size([128]) 128 True
net.blocks.0.norm1.weight torch.Size([128]) 128 True
net.blocks.0.norm1.bias torch.Size([128]) 128 True
net.blocks.0.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.blocks.0.attn.in_proj_bias torch.Size([384]) 384 True
net.blocks.0.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.blocks.0.attn.out_proj.bias torch.Size([128]) 128 True
net.blocks.0.norm2.weight torch.Size([128]) 128 True
net.blocks.0.norm2.bias torch.Size([128]) 128 True
net.blocks.0.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.blocks.0.mlp.fc1.bias torch.Size([512]) 512 True
net.blocks.0.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.blocks.0.mlp.fc2.bias torch.Size([128]) 128 True
net.blocks.1.norm1.weight torch.Size([128]) 128 True
net.blocks.1.norm1.bias torch.Size([128]) 128 True
net.blocks.1.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.blocks.1.attn.in_proj_bias torch.Size([384]) 384 True
net.blocks.1.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.blocks.1.attn.out_proj.bias torch.Size([128]) 128 True
net.blocks.1.norm2.weight torch.Size([128]) 128 True
net.blocks.1.norm2.bias torch.Size([128]) 128 True
net.blocks.1.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.blocks.1.mlp.fc1.bias torch.Size([512]) 512 True
net.blocks.1.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.blocks.1.mlp.fc2.bias torch.Size([128]) 128 True
net.blocks.2.norm1.weight torch.Size([128]) 128 True
net.blocks.2.norm1.bias torch.Size([128]) 128 True
net.blocks.2.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.blocks.2.attn.in_proj_bias torch.Size([384]) 384 True
net.blocks.2.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.blocks.2.attn.out_proj.bias torch.Size([128]) 128 True
net.blocks.2.norm2.weight torch.Size([128]) 128 True
net.blocks.2.norm2.bias torch.Size([128]) 128 True
net.blocks.2.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.blocks.2.mlp.fc1.bias torch.Size([512]) 512 True
net.blocks.2.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.blocks.2.mlp.fc2.bias torch.Size([128]) 128 True
net.blocks.3.norm1.weight torch.Size([128]) 128 True
net.blocks.3.norm1.bias torch.Size([128]) 128 True
net.blocks.3.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.blocks.3.attn.in_proj_bias torch.Size([384]) 384 True
net.blocks.3.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.blocks.3.attn.out_proj.bias torch.Size([128]) 128 True
net.blocks.3.norm2.weight torch.Size([128]) 128 True
net.blocks.3.norm2.bias torch.Size([128]) 128 True
net.blocks.3.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.blocks.3.mlp.fc1.bias torch.Size([512]) 512 True
net.blocks.3.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.blocks.3.mlp.fc2.bias torch.Size([128]) 128 True
net.norm.weight torch.Size([128]) 128 True
net.norm.bias torch.Size([128]) 128 True
net.decoder.lane_blks.0.normkv.weight torch.Size([128]) 128 True
net.decoder.lane_blks.0.normkv.bias torch.Size([128]) 128 True
net.decoder.lane_blks.0.normk.weight torch.Size([128]) 128 True
net.decoder.lane_blks.0.normk.bias torch.Size([128]) 128 True
net.decoder.lane_blks.0.norm1.weight torch.Size([128]) 128 True
net.decoder.lane_blks.0.norm1.bias torch.Size([128]) 128 True
net.decoder.lane_blks.0.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.decoder.lane_blks.0.attn.in_proj_bias torch.Size([384]) 384 True
net.decoder.lane_blks.0.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.decoder.lane_blks.0.attn.out_proj.bias torch.Size([128]) 128 True
net.decoder.lane_blks.0.norm2.weight torch.Size([128]) 128 True
net.decoder.lane_blks.0.norm2.bias torch.Size([128]) 128 True
net.decoder.lane_blks.0.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.decoder.lane_blks.0.mlp.fc1.bias torch.Size([512]) 512 True
net.decoder.lane_blks.0.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.decoder.lane_blks.0.mlp.fc2.bias torch.Size([128]) 128 True
net.decoder.lane_blks.1.normkv.weight torch.Size([128]) 128 True
net.decoder.lane_blks.1.normkv.bias torch.Size([128]) 128 True
net.decoder.lane_blks.1.normk.weight torch.Size([128]) 128 True
net.decoder.lane_blks.1.normk.bias torch.Size([128]) 128 True
net.decoder.lane_blks.1.norm1.weight torch.Size([128]) 128 True
net.decoder.lane_blks.1.norm1.bias torch.Size([128]) 128 True
net.decoder.lane_blks.1.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.decoder.lane_blks.1.attn.in_proj_bias torch.Size([384]) 384 True
net.decoder.lane_blks.1.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.decoder.lane_blks.1.attn.out_proj.bias torch.Size([128]) 128 True
net.decoder.lane_blks.1.norm2.weight torch.Size([128]) 128 True
net.decoder.lane_blks.1.norm2.bias torch.Size([128]) 128 True
net.decoder.lane_blks.1.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.decoder.lane_blks.1.mlp.fc1.bias torch.Size([512]) 512 True
net.decoder.lane_blks.1.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.decoder.lane_blks.1.mlp.fc2.bias torch.Size([128]) 128 True
net.decoder.lane_blks.2.normkv.weight torch.Size([128]) 128 True
net.decoder.lane_blks.2.normkv.bias torch.Size([128]) 128 True
net.decoder.lane_blks.2.normk.weight torch.Size([128]) 128 True
net.decoder.lane_blks.2.normk.bias torch.Size([128]) 128 True
net.decoder.lane_blks.2.norm1.weight torch.Size([128]) 128 True
net.decoder.lane_blks.2.norm1.bias torch.Size([128]) 128 True
net.decoder.lane_blks.2.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.decoder.lane_blks.2.attn.in_proj_bias torch.Size([384]) 384 True
net.decoder.lane_blks.2.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.decoder.lane_blks.2.attn.out_proj.bias torch.Size([128]) 128 True
net.decoder.lane_blks.2.norm2.weight torch.Size([128]) 128 True
net.decoder.lane_blks.2.norm2.bias torch.Size([128]) 128 True
net.decoder.lane_blks.2.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.decoder.lane_blks.2.mlp.fc1.bias torch.Size([512]) 512 True
net.decoder.lane_blks.2.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.decoder.lane_blks.2.mlp.fc2.bias torch.Size([128]) 128 True
net.decoder.agent_blks.0.normkv.weight torch.Size([128]) 128 True
net.decoder.agent_blks.0.normkv.bias torch.Size([128]) 128 True
net.decoder.agent_blks.0.normk.weight torch.Size([128]) 128 True
net.decoder.agent_blks.0.normk.bias torch.Size([128]) 128 True
net.decoder.agent_blks.0.norm1.weight torch.Size([128]) 128 True
net.decoder.agent_blks.0.norm1.bias torch.Size([128]) 128 True
net.decoder.agent_blks.0.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.decoder.agent_blks.0.attn.in_proj_bias torch.Size([384]) 384 True
net.decoder.agent_blks.0.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.decoder.agent_blks.0.attn.out_proj.bias torch.Size([128]) 128 True
net.decoder.agent_blks.0.norm2.weight torch.Size([128]) 128 True
net.decoder.agent_blks.0.norm2.bias torch.Size([128]) 128 True
net.decoder.agent_blks.0.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.decoder.agent_blks.0.mlp.fc1.bias torch.Size([512]) 512 True
net.decoder.agent_blks.0.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.decoder.agent_blks.0.mlp.fc2.bias torch.Size([128]) 128 True
net.decoder.agent_blks.1.normkv.weight torch.Size([128]) 128 True
net.decoder.agent_blks.1.normkv.bias torch.Size([128]) 128 True
net.decoder.agent_blks.1.normk.weight torch.Size([128]) 128 True
net.decoder.agent_blks.1.normk.bias torch.Size([128]) 128 True
net.decoder.agent_blks.1.norm1.weight torch.Size([128]) 128 True
net.decoder.agent_blks.1.norm1.bias torch.Size([128]) 128 True
net.decoder.agent_blks.1.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.decoder.agent_blks.1.attn.in_proj_bias torch.Size([384]) 384 True
net.decoder.agent_blks.1.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.decoder.agent_blks.1.attn.out_proj.bias torch.Size([128]) 128 True
net.decoder.agent_blks.1.norm2.weight torch.Size([128]) 128 True
net.decoder.agent_blks.1.norm2.bias torch.Size([128]) 128 True
net.decoder.agent_blks.1.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.decoder.agent_blks.1.mlp.fc1.bias torch.Size([512]) 512 True
net.decoder.agent_blks.1.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.decoder.agent_blks.1.mlp.fc2.bias torch.Size([128]) 128 True
net.decoder.agent_blks.2.normkv.weight torch.Size([128]) 128 True
net.decoder.agent_blks.2.normkv.bias torch.Size([128]) 128 True
net.decoder.agent_blks.2.normk.weight torch.Size([128]) 128 True
net.decoder.agent_blks.2.normk.bias torch.Size([128]) 128 True
net.decoder.agent_blks.2.norm1.weight torch.Size([128]) 128 True
net.decoder.agent_blks.2.norm1.bias torch.Size([128]) 128 True
net.decoder.agent_blks.2.attn.in_proj_weight torch.Size([384, 128]) 49152 True
net.decoder.agent_blks.2.attn.in_proj_bias torch.Size([384]) 384 True
net.decoder.agent_blks.2.attn.out_proj.weight torch.Size([128, 128]) 16384 True
net.decoder.agent_blks.2.attn.out_proj.bias torch.Size([128]) 128 True
net.decoder.agent_blks.2.norm2.weight torch.Size([128]) 128 True
net.decoder.agent_blks.2.norm2.bias torch.Size([128]) 128 True
net.decoder.agent_blks.2.mlp.fc1.weight torch.Size([512, 128]) 65536 True
net.decoder.agent_blks.2.mlp.fc1.bias torch.Size([512]) 512 True
net.decoder.agent_blks.2.mlp.fc2.weight torch.Size([128, 512]) 65536 True
net.decoder.agent_blks.2.mlp.fc2.bias torch.Size([128]) 128 True
net.decoder.pi.0.weight torch.Size([256, 128]) 32768 True
net.decoder.pi.0.bias torch.Size([256]) 256 True
net.decoder.pi.2.weight torch.Size([1, 256]) 256 True
net.decoder.pi.2.bias torch.Size([1]) 1 True
net.decoder.loc.0.weight torch.Size([256, 128]) 32768 True
net.decoder.loc.0.bias torch.Size([256]) 256 True
net.decoder.loc.2.weight torch.Size([120, 256]) 30720 True
net.decoder.loc.2.bias torch.Size([120]) 120 True
net.decoder.mode_embed.weight torch.Size([6, 128]) 768 True
net.dense_predictor.0.weight torch.Size([256, 128]) 32768 True
net.dense_predictor.0.bias torch.Size([256]) 256 True
net.dense_predictor.2.weight torch.Size([120, 256]) 30720 True
net.dense_predictor.2.bias torch.Size([120]) 120 True