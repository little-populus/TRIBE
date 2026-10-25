import math
import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
from ..utils.bn_layers import BalancedRobustBN2dV5, BalancedRobustBN2dEMA, BalancedRobustBN1dV5
from ..utils.utils import set_named_submodule, get_named_submodule
from ..utils.custom_transforms import get_tta_transforms


class TRIBE(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(TRIBE, self).__init__(cfg, model, optimizer)

        self.aux_model = self.build_ema(self.model)
        for (name1, param1), (name2, param2) in zip(self.model.named_parameters(), self.aux_model.named_parameters()):
            set_named_submodule(self.aux_model, name2, param1)

        self.source_model = self.build_ema(self.model)
        self.transform = get_tta_transforms(cfg)
        # cumulative pseudo-label counts for online class-prior estimation (TLA)
        self.register_buffer("prior_count", torch.zeros(cfg.CORRUPTION.NUM_CLASS))
        # shift-aware statistics re-warming
        self.bn_layers = [m for m in self.modules() if isinstance(m, (BalancedRobustBN2dV5, BalancedRobustBN1dV5))]
        if cfg.ADAPTER.TRIBE.SHIFT_WARMUP:
            # modules() order: student (self.model), teacher (aux_model), anchor.
            # The sensor must sit on the teacher path because the eval-mode
            # inference that produces predictions runs through aux_model.
            n_per_net = len(self.bn_layers) // 3
            self.shift_layer = self.bn_layers[n_per_net]
            self.shift_layer.track_shift = True
            self.prev_batch_mean = None
            self.batch_counter = 0
            self.warmup_left = 0
            for bn in self.bn_layers:
                bn.base_momentum = bn.momentum

    def _maybe_rewarm(self):
        """Detect a domain switch from the distance between consecutive batch
        feature means (captured by the sensor BN layer on the teacher path) and
        temporarily boost the BN momentum so stale statistics are forgotten
        quickly; a permanently large momentum is unstable (cf. TRIBE Table 16),
        so the boost is only applied for a short window after each switch."""
        cfg = self.cfg.ADAPTER.TRIBE
        cur = self.shift_layer.last_batch_mean
        if self.prev_batch_mean is not None:
            dist = (cur - self.prev_batch_mean).norm().item() / (self.prev_batch_mean.norm().item() + 1e-8)
            self.batch_counter += 1
            if self.warmup_left > 0:
                self.warmup_left -= 1
                if self.warmup_left == 0:
                    for bn in self.bn_layers:
                        bn.momentum = bn.base_momentum
            elif dist > cfg.SHIFT_DELTA:
                self.warmup_left = cfg.SHIFT_WARMUP_T
                for bn in self.bn_layers:
                    bn.momentum = bn.base_momentum * cfg.SHIFT_BOOST
                print(f"[shift-warmup] batch {self.batch_counter}: dist={dist:.3f} -> boost x{cfg.SHIFT_BOOST} for {cfg.SHIFT_WARMUP_T} batches", flush=True)
        self.prev_batch_mean = cur

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        with torch.no_grad():
            self.aux_model.eval()
            ema_out = self.aux_model(batch_data)
            adj_out = self.adjust_output(ema_out)

        if self.cfg.ADAPTER.TRIBE.SHIFT_WARMUP:
            self._maybe_rewarm()

        # "train" mode: the de-biased output only feeds the training signal
        # (pseudo-labels -> ST loss + BN statistics); the returned prediction
        # stays micro-optimal, so class-prior correction cannot hurt the
        # instance-wise metric by flipping correct head predictions.
        if self.cfg.ADAPTER.TRIBE.TLA and self.cfg.ADAPTER.TRIBE.TLA_MODE == "train":
            self.update_model(model, optimizer, batch_data, adj_out)
            return ema_out

        if self.cfg.ADAPTER.TRIBE.ENSEMBLE:
            # tri-net ensemble: mix the adapted teacher with the (frozen-weight)
            # anchor view; the anchor is most valuable right after domain
            # switches while the teacher statistics are still re-warming
            with torch.no_grad():
                self.source_model.eval()
                anchor_out = self.source_model(batch_data)
            w = self.cfg.ADAPTER.TRIBE.ENSEMBLE_W
            out = (1.0 - w) * ema_out + w * anchor_out
            self.update_model(model, optimizer, batch_data, adj_out)
            return out

        self.update_model(model, optimizer, batch_data, adj_out)

        return adj_out

    def adjust_output(self, logits):
        """Test-time logit adjustment (TLA): de-bias teacher predictions with an
        online estimate of the test class prior (Saerens et al. 2002; Menon et al. 2021).

        The source model assumes a uniform prior; under a globally imbalanced stream the
        logits are systematically biased towards head classes. Subtracting
        tau * log(K * pi_hat) both corrects the returned prediction and yields
        de-biased pseudo-labels for self-training and BN statistics assignment.
        """
        if not self.cfg.ADAPTER.TRIBE.TLA:
            return logits
        adj, self.prior_count = self.tla_adjust(
            logits, self.prior_count,
            self.cfg.ADAPTER.TRIBE.TLA_TAU,
            self.cfg.ADAPTER.TRIBE.TLA_ALPHA,
            self.cfg.ADAPTER.TRIBE.TLA_WARMUP,
        )
        return adj

    @staticmethod
    def tla_adjust(logits: torch.Tensor, prior_count: torch.Tensor, tau: float, alpha: float, warmup: int):
        """Pure function form of test-time logit adjustment (unit-testable).

        logits: (B, K) raw teacher outputs; prior_count: (K,) cumulative pseudo-label counts.
        Returns (adjusted_logits, updated_prior_count). During the first ``warmup``
        pseudo-labels the estimate is too noisy to correct with, but counts are still collected.
        """
        num_class = logits.size(1)
        pred = logits.argmax(dim=-1)
        count = torch.bincount(pred, minlength=num_class).to(logits.dtype)
        if prior_count.sum() < warmup:
            return logits, prior_count + count
        prior = (prior_count + alpha) / (prior_count.sum() + alpha * num_class)
        adjusted = logits - tau * torch.log(num_class * prior)
        adj_pred = adjusted.argmax(dim=-1)
        adj_count = torch.bincount(adj_pred, minlength=num_class).to(logits.dtype)
        return adjusted, prior_count + adj_count
    

    def update_model(self, model, optimizer, batch_data, logit):

        p_l = logit.argmax(dim=1)

        soft_w = None
        if self.cfg.ADAPTER.TRIBE.SOFT_BN:
            tau = self.cfg.ADAPTER.TRIBE.SOFT_TAU
            soft_w = (logit / tau).softmax(dim=-1).detach()

        self.source_model.train()
        self.aux_model.train()
        model.train()
        strong_sup_aug = self.transform(batch_data)

        self.set_bn_label(self.aux_model, p_l, soft_w)
        ema_sup_out = self.aux_model(batch_data)

        self.set_bn_label(model, p_l, soft_w)
        stu_sup_out = model(strong_sup_aug)

        entropy = self.self_softmax_entropy(ema_sup_out)
        entropy_mask = (entropy < self.cfg.ADAPTER.TRIBE.H0 * math.log(self.cfg.CORRUPTION.NUM_CLASS))

        if entropy_mask.sum() == 0:
            # no confident sample in this batch: skip the update instead of
            # back-propagating a NaN loss (behaviour of the original code on
            # an empty mask)
            return

        # with TLA the de-biased pseudo-labels (from the adjusted eval-mode output)
        # also serve as the self-training target; otherwise keep the original target
        if self.cfg.ADAPTER.TRIBE.TLA:
            pseudo_target = p_l
        else:
            pseudo_target = ema_sup_out.argmax(dim=-1)
        per_sup = torch.nn.functional.cross_entropy(stu_sup_out, pseudo_target, reduction='none')

        with torch.no_grad():
            self.set_bn_label(self.source_model, p_l, soft_w)
            source_anchor = self.source_model(batch_data).detach()

        if self.cfg.ADAPTER.TRIBE.ANCHOR_TYPE == "kl":
            per_reg = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(ema_sup_out, dim=-1),
                torch.nn.functional.softmax(source_anchor, dim=-1),
                reduction='none').sum(dim=-1)
        else:
            per_reg = torch.nn.functional.mse_loss(ema_sup_out, source_anchor, reduction='none').mean(dim=-1)

        sample_w = None
        if self.cfg.ADAPTER.TRIBE.CB_LOSS:
            cls = pseudo_target[entropy_mask]
            cnt = torch.bincount(cls, minlength=self.cfg.CORRUPTION.NUM_CLASS).float()
            sample_w = (1.0 / (cnt[cls] + 1e-8)) ** self.cfg.ADAPTER.TRIBE.CB_BETA
            sample_w = sample_w / sample_w.mean()

        if sample_w is None:
            l_sup = per_sup[entropy_mask].mean()
            l_reg = per_reg[entropy_mask].mean()
        else:
            l_sup = (per_sup[entropy_mask] * sample_w).mean()
            l_reg = (per_reg[entropy_mask] * sample_w).mean()

        l = (l_sup + self.cfg.ADAPTER.TRIBE.LAMBDA * l_reg)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        return

    @staticmethod
    def set_bn_label(model, label=None, soft_w=None):
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, BalancedRobustBN1dV5) or isinstance(sub_module, BalancedRobustBN2dV5) or isinstance(sub_module, BalancedRobustBN2dEMA):
                sub_module.label = label
                sub_module.soft_w = soft_w
        return
    
    @staticmethod
    def self_softmax_entropy(x):
        return -(x.softmax(dim=-1) * x.log_softmax(dim=-1)).sum(dim=-1)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model
    
    def configure_model(self, model: nn.Module):

        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm2d) or isinstance(sub_module, nn.BatchNorm1d):
                normlayer_names.append(name)
                
        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = BalancedRobustBN2dV5
                # NewBN = BalancedRobustBN2dEMA
            elif isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = BalancedRobustBN1dV5
            else:
                raise RuntimeError()
            
            momentum_bn = NewBN(bn_layer,
                                self.cfg.CORRUPTION.NUM_CLASS,
                                self.cfg.ADAPTER.TRIBE.ETA,
                                self.cfg.ADAPTER.TRIBE.GAMMA,
                                self.cfg.ADAPTER.TRIBE.ADAPT_MOM,
                                self.cfg.ADAPTER.TRIBE.ADAPT_MOM_C,
                                self.cfg.ADAPTER.TRIBE.MOM_MAX
                                )
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)
        return model
