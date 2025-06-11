import torch
import torch.nn as nn
import torchdiffeq

from .ot_cfm import ExactOptimalTransportConditionalFlowMatcher
from .decoder import Decoder


class CFMFlow(nn.Module):
    def __init__(self):
        super(CFMFlow, self).__init__()
        self.follow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=1e-4)
        self.decoder = Decoder()

    # def forward(self, mu, speaker_embedding, target, mask=None):
    #     x1 = target
    #     x0 = torch.randn_like(x1)
    #     t, xt, ut = self.follow_matcher.sample_location_and_conditional_flow(x0, x1)
    #     decoded_output = self.decoder(mu, speaker_embedding, t, xt, mask)
    #     x = decoded_output["output"]
    #     # if ut.shape[1] > x.shape[1]:
    #     #     ut = ut[:, :x.shape[1], :]

    #     if decoded_output["mask"] is None:
    #         cfm_loss = F.mse_loss(x, ut, reduction="sum") / (ut.shape[0] * ut.shape[1] * ut.shape[2])
    #     else:
    #         cfm_loss = F.mse_loss(x, ut, reduction="sum") / (torch.sum(decoded_output["mask"]) * ut.shape[1] + 1e-6)
    #     return {
    #         "output": decoded_output["output"],
    #         "cfm_loss": cfm_loss,
    #         "mask": decoded_output["mask"],
    #     }

    def forward(
        self,
        mu,
        speaker_embedding,
        n_timesteps=5,
        mask=None,
        temperature=1.0,
        solver="euler",  # or "rk4"
        return_all=False
    ):
        """
        Args:
            mu: Tensor, (B, C, T)
            speaker_embedding: Tensor, (B, D)
            n_timesteps: number of ODE steps
            mask: Tensor, (B, 1, T)
            temperature: noise scale
            solver: ODE solver method
            return_all: return all trajectory steps
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0.0, 1.0, n_timesteps + 1, device=mu.device)

        traj = torchdiffeq.odeint(
            lambda t_, x_: self.decoder(mu, speaker_embedding, t_, x_, mask)["output"],
            z,
            t_span,
            atol=1e-4,
            rtol=1e-4,
            method=solver,
        )
        return traj[-1] if not return_all else traj