from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn.functional as F

from src.model.trainer_forecast import Trainer



class RKDTrainer(Trainer):

    def __init__(self, *args, 
                 teacher_model: torch.nn.Module,
                 rkd_training: bool = True,
                 rkd_on: str = "x_agent",
                 lambda_distance: float = 5.0,
                 lambda_angle: float = 50.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.eval()  # keep teacher in eval / frozen
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)
        self.rkd_training = rkd_training
        self.rkd_on = rkd_on
        self.lambda_distance = lambda_distance
        self.lambda_angle = lambda_angle

    def masked_mean(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, L, D]   (L = N+M)
        mask  : [B, L]      (1 = keep, 0 = drop)
        returns: [B, D]
        """
        mask = mask.unsqueeze(-1).type_as(tokens)          # [B,L,1]
        summed = (tokens * mask).sum(dim=1)                # [B, D]
        count  = mask.sum(dim=1).clamp(min=1.0)            # avoid div-by-0
        return summed / count                              # [B, D] mean over valid tokens

    def get_scene_vec(self, data, out):
        """
        Get a [B, D] 'scene vector' by averaging the output of the encoder over all valid tokens.
        """
        x_encoder = out["x_encoder"]                               # [B, N+M, D]
        agent_valid = ~data["x_key_padding_mask"]                  # [B, N] agent mask
        lane_valid  = ~data["lane_key_padding_mask"]               # [B, M] lane mask
        scene_valid = torch.cat([agent_valid, lane_valid], dim=1)  # [B, N+M] joint mask
        scene_vec = self.masked_mean(x_encoder, scene_valid)       # [B, D] scene vector
        return scene_vec

    def get_distances(self, vec):
        """
        Return the upper triangular pairwise distances between every pair of vectors in the batch.
        (Additional normalization by factor mu as in the RKD paper.)
        """

        B = vec.size(0)
        diffs = vec.unsqueeze(1) - vec.unsqueeze(0)                   # [B,B,D]
        dists = diffs.pow(2).sum(dim=-1)                              # [B,B]
        pairwise = torch.triu(dists, diagonal=1)                      # [B,B]
        n_pairs = (B * (B - 1)) / 2
        mu = pairwise.sum() / n_pairs
        return pairwise/(mu + 1e-6)
    
    def get_angles(self, vec):
        """
        Return the upper triangular (in the 3D cubic sense) angles between every triplet of vectors in the batch.
        """
        B, D = vec.shape

        # build broadcasted index grids
        idx = torch.arange(B, device=vec.device)
        I, J, K = torch.meshgrid(idx, idx, idx, indexing="ij")  # each is [B,B,B] such that I[a,b,c]=a, J[a,b,c]=b, K[a,b,c]=c

        # pick out the vectors for each triplet
        ti = vec[I]   # [B,B,B,D] such that ti[a,b,c,:]=vec[a]
        tj = vec[J]
        tk = vec[K]

        eps = 1e-6
        eij = F.normalize(ti - tj, p=2, dim=-1, eps=eps)
        ekj = F.normalize(tk - tj, p=2, dim=-1, eps=eps)

        # cosine of the angle
        angles = (eij * ekj).sum(dim=-1)  # [B,B,B]; normalization unnecessary since eij and ejk already normalized

        # now mask out anything not in the "upper triangle" in the cubic sense => only keep i < j < k
        mask = (I < J) & (J < K)  # boolean [B,B,B]
        angles = angles * mask    # zero out all the rest

        return angles
    
    def get_vec(self, data, out):
        """
        Get the vector to be used for distance/angle loss. 
        "x_encoder":  [B, N+M, D] the scene vector where (N,M)=(#agents, #lanes) and there is average pooling across the objects
        "x_agent":    [B, D] the agent vector
        """
        if self.rkd_on == "x_agent":
            return out["x_agent"]
        elif self.rkd_on == "x_encoder":
            return self.get_scene_vec(data, out)
        else:
            raise ValueError(f"Unknown rkd_on value: {self.rkd_on}. Must be 'x_agent' or 'x_encoder'.")

    def cal_loss(self, data, batch_idx=0):

        out = self(data) # forward pass through the student model

        # Task loss components
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        task_loss = 0
        B = y_hat.shape[0]
        B_range = range(B)

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(-1)

        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[B_range, best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)

        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
        task_loss += agent_reg_loss + agent_cls_loss
        
        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.history_steps:]
        others_reg_loss = F.smooth_l1_loss(y_hat_others[others_reg_mask], y_others[others_reg_mask])
        task_loss += others_reg_loss

        losses = {
            "loss": task_loss,
            "task_loss": task_loss.item(),
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            "others_reg_loss": others_reg_loss.item(),
        }

        if self.rkd_training:
            # RKD loss components
            out_teacher = self.teacher_model(data)  # forward pass through the teacher model
            vec = self.get_vec(data, out)                                 # [B, D]
            vec_teacher = self.get_vec(data, out_teacher)                 # [B, D']
            rkd_loss = 0
            # distance component
            pairwise_distances = self.get_distances(vec)                  # [B, B]
            pairwise_distances_teacher = self.get_distances(vec_teacher)  # [B, B]
            distance_loss = F.huber_loss(pairwise_distances, pairwise_distances_teacher, reduction="mean")
            rkd_loss += self.lambda_distance * distance_loss
            # angle component
            angles = self.get_angles(vec)                                 # [B, B, B]
            angles_teacher = self.get_angles(vec_teacher)                 # [B, B, B]
            angle_loss = F.huber_loss(angles, angles_teacher, reduction="mean")
            rkd_loss += self.lambda_angle * angle_loss

            loss = task_loss + rkd_loss

            losses.update({
                "loss": loss,
                "rkd_loss": rkd_loss.item(),
                "distance_loss": distance_loss.item(),
                "angle_loss": angle_loss.item(),
                "task_loss": task_loss.item(),
                "reg_loss": agent_reg_loss.item(),
                "cls_loss": agent_cls_loss.item(),
                "others_reg_loss": others_reg_loss.item(),
            })
        
        return losses


    def training_step(self, data: dict[str, Any], batch_idx: int):
        
        losses = self.cal_loss(data, batch_idx)

        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        if batch_idx == 0:

            print_str = (
                f"Epoch {self.current_epoch:2d} ▶",
                f"loss={losses['loss']:.4f},",
                f"task_loss={losses['task_loss']:.4f},",
            )

            if self.rkd_training:
                print_str += (
                    f"rkd_loss={losses['rkd_loss']:.4f}",
                    f"distance_loss={losses['distance_loss']:.4f}",
                    f"angle_loss={losses['angle_loss']:.4f}",
                )
            
            print(print_str)

        return losses["loss"]

    def validation_step(self, data, batch_idx):

        losses = self.cal_loss(data, batch_idx=-1)
        out = self(data)
        metrics = self.val_metrics(out, data["y"][:, 0])

        self.log(
            "val/reg_loss",
            losses["reg_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        for k in self.val_scores.keys(): self.val_scores[k].append(metrics[k].item())

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
    
    ## for debugging purposes, uncomment to print gradient norms
    # def on_after_backward(self):
    #     total_norm = 0.0
    #     for p in self.parameters():
    #         if p.grad is not None:
    #             param_norm = p.grad.data.norm(2)
    #             total_norm += param_norm.item() ** 2
    #     total_norm = total_norm ** 0.5
    #     print(f"Gradient norm: {total_norm}")
