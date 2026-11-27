"""Stock-flow and observation-informed crowd forecasting on Sim-SH-ENB."""

import torch
from torch import nn
from torch.nn import functional as F

from .backbones import (
    _MegaCRNNet,
    _STAEformerNet,
    _STIDNet,
    init_named,
    normalized,
    summary_tokens,
    time_features,
)

RAW_KEYS = ("history", "history_mask", "sidecar", "calendar", "issue_slot", "weekday")


def legal_raw(raw):
    r = {k: raw[k] for k in RAW_KEYS if k in raw}
    x, m, s = r["history"], r["history_mask"], r["sidecar"]
    if x.ndim != 3 or x.shape[1:] != (24, 34) or m.shape != x.shape:
        raise ValueError("Require exactly [batch,24,34] values and original mask")
    if m.dtype != torch.bool or s.shape != x.shape + (7,):
        raise ValueError("Require bool mask and seven arrival-sidecar fields")
    if not torch.isfinite(x[m]).all() or not torch.isfinite(s[m]).all():
        raise ValueError("Visible values/quality must be finite")
    if m.any():
        visible = s[m]
        if (
            (visible[:, 0] < 0).any()
            or (visible[:, 1] > 120.001).any()
            or (visible[:, 0] > visible[:, 1] + 0.001).any()
            or (visible[:, 3] < -0.001).any()
            or (visible[:, 3] > visible[:, 2] + 0.001).any()
        ):
            raise ValueError("Illegal reference age/upload clock in visible sidecar")
    r["history"] = torch.where(m, x, torch.full_like(x, float("nan")))
    r["sidecar"] = torch.where(m[..., None], s, torch.zeros_like(s))
    return r


class PrefixEncoder(nn.Module):
    def __init__(self, stats, width):
        super().__init__()
        for key in ("mean", "std"):
            self.register_buffer(
                key, torch.as_tensor(stats[key], dtype=torch.float32).clone()
            )
        if (
            (self.std <= 0).any()
            or not torch.isfinite(self.std).all()
            or not torch.isfinite(self.mean).all()
        ):
            raise ValueError("Training standard deviations must be finite and positive")
        self.input = nn.Linear(8, width)
        self.identity = nn.Embedding(34, width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.width = width
        self.register_buffer(
            "kinds", F.one_hot(torch.tensor([0] * 5 + [1] * 5 + [2] * 24), 3).float()
        )

    def forward(self, raw):
        x, m, s = raw["history"], raw["history_mask"], raw["sidecar"]
        z = torch.where(m, (x - self.mean) / self.std, torch.zeros_like(x))
        info = torch.stack(
            (z, m.float(), s[..., 0] / 120, s[..., 3] / 60, s[..., 6]), -1
        )
        kinds = self.kinds[None, None].expand(len(x), 24, -1, -1)
        h = self.input(torch.cat((info, kinds), -1)) + self.identity.weight[None, None]
        h = h.transpose(1, 2).reshape(len(x) * 34, 24, self.width)
        h, _ = self.gru(h)  # no inherited hidden state
        h = h.reshape(len(x), 34, 24, self.width).transpose(1, 2)
        return h


class StockFlowBranch(nn.Module):
    """Roll out latent stocks and transfers and project their features onto channels."""

    def __init__(self, spatial_arrays, width=32):
        super().__init__()
        for name in ("b", "n_support", "input_support", "stock_scale", "flow_capacity"):
            self.register_buffer(
                name, torch.as_tensor(spatial_arrays[name], dtype=torch.float32)
            )
        for name in ("tails", "heads"):
            self.register_buffer(
                name, torch.as_tensor(spatial_arrays[name], dtype=torch.long)
            )

        self.horizon = nn.Embedding(6, width)
        self.initial = nn.Linear(width, 1)
        self.stock = nn.Linear(width + 1, 1)
        self.flow = nn.Linear(2 * width, 1)
        self.readout = nn.Linear(width + 3, width)
        readout_weights = self.input_support.T
        self.register_buffer(
            "channel_pool",
            readout_weights / readout_weights.sum(-1, keepdim=True).clamp_min(1e-8),
        )

    def forward(self, history_features):
        weights = self.input_support / self.input_support.sum(
            -1, keepdim=True
        ).clamp_min(1e-8)
        nodes = torch.einsum("vc,btcw->btvw", weights, history_features)[:, -1]
        stock = F.softplus(self.initial(nodes).squeeze(-1)) * self.stock_scale
        stocks, flows, features = [stock], [], []
        sources = (-self.b).clamp_min(0)
        destinations = self.b.clamp_min(0)
        for step in range(6):
            context = nodes + self.horizon.weight[step]
            edges = torch.cat((context[:, self.tails], context[:, self.heads]), -1)
            proposed = self.flow(edges).squeeze(-1).sigmoid() * self.flow_capacity
            outgoing = torch.einsum("vd,bd->bv", sources, proposed)
            flow = (
                proposed
                * (stock / outgoing.clamp_min(1e-8)).clamp(max=1)[:, self.tails]
            )
            incoming = torch.einsum("vd,bd->bv", destinations, flow)
            outgoing = torch.einsum("vd,bd->bv", sources, flow)

            # Stock advances through a learned update; flow clipping bounds outgoing transfers.
            stock_input = torch.cat(
                (context, (stock / self.stock_scale)[..., None]), -1
            )
            stock = F.softplus(self.stock(stock_input).squeeze(-1)) * self.stock_scale
            stock_flow = (
                torch.stack((stock, incoming, outgoing), -1) / self.stock_scale[:, None]
            )
            readout = self.readout(torch.cat((context, stock_flow), -1))
            features.append(torch.einsum("cv,bvw->bcw", self.channel_pool, readout))
            stocks.append(stock)
            flows.append(flow)

        stocks = torch.stack(stocks, 1)
        aux = {
            "occupancy_prediction": torch.einsum(
                "nv,bhv->bhn", self.n_support, stocks[:, 1:]
            ),
            "latent_stocks": stocks,
            "latent_flows": torch.stack(flows, 1),
        }
        return torch.stack(features, 1), aux


class OccupancyAdapter(nn.Module):
    """Condition regional forecast residuals on history levels and availability."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(89, 100),
            nn.GELU(),
            nn.Linear(100, 100),
            nn.GELU(),
            nn.Linear(100, 30),
        )

    def forward(self, raw, base, mean, std):
        columns = [0, 2, 4, 5, 6, 7, 8, 9, 10, 11]
        summaries = summary_tokens(raw, mean, std)[:, 5:10, columns].flatten(1)
        base_occupancy = (base[..., 5:10] - mean[5:10]) / std[5:10]
        inputs = torch.cat(
            (summaries, time_features(raw, 288), base_occupancy.flatten(1)), -1
        )
        delta = self.layers(inputs).reshape(-1, 6, 5) * std[5:10]
        return torch.cat(
            (base[..., :5], (base[..., 5:10] + delta).clamp_min(0), base[..., 10:]), -1
        )


class Forecaster(nn.Module):
    normalized = normalized

    def __init__(self, backbone, variant, stats, spatial_arrays, seed=0):
        super().__init__()
        self.name = backbone
        if variant not in {"B", "S", "A", "SA"}:
            raise ValueError(f"Unknown variant: {variant}")
        for key in ("mean", "std", "median", "scales"):
            self.register_buffer(
                key, torch.as_tensor(stats[key], dtype=torch.float32).clone()
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if self.name == "stid":
                self.net = _STIDNet(34, 32, 8, 2)
                head_name, base_width, output_width = "output", 32, 6
            elif self.name == "staeformer":
                self.net = _STAEformerNet(
                    dict(feature_dim=16, adaptive_dim=8, heads=4, layers=1, dropout=0.0)
                )
                head_name, base_width, output_width = "head", 24 * 56, 6
            elif self.name == "megacrn":
                self.net = _MegaCRNNet()
                head_name, base_width, output_width = "projection", 24, 1
            else:
                raise ValueError(self.name)
            # Initialize the common trunk and common head before adding branches.
            init_named(self.net, seed)
            old_head = getattr(self.net, head_name)
            spatial = "S" in variant
            self.encoder = self.branch = None
            if spatial:
                self.encoder = PrefixEncoder(stats, 32)
                init_named(self.encoder, seed + 17000)
                self.branch = StockFlowBranch(spatial_arrays)
                init_named(self.branch, seed + 18000)
                extra = 32 if self.name == "megacrn" else 6 * 32
                head = nn.Linear(base_width + extra, output_width)
                init_named(head, seed + 19000)
                with torch.no_grad():
                    head.weight[:, :base_width].copy_(old_head.weight)
                    head.bias.copy_(old_head.bias)
                setattr(self.net, head_name, head)
            self.n_adapter = OccupancyAdapter() if "A" in variant else None
            if self.n_adapter is not None:
                init_named(self.n_adapter, seed + 20000)
                nn.init.zeros_(self.n_adapter.layers[-1].weight)
                nn.init.zeros_(self.n_adapter.layers[-1].bias)

    def forward(self, raw, *, return_aux=False):
        raw = legal_raw(raw)
        x, mask = self.normalized(raw)
        z, aux = None, {}
        if self.branch is not None:
            z, aux = self.branch(self.encoder(raw))
        net, batch = self.net, len(x)
        regularizer = x.new_zeros(())
        if self.name == "stid":
            identity = net.channel_id.weight[None].expand(batch, -1, -1)
            daily = net.time_of_day(raw["issue_slot"])[:, None].expand(-1, 34, -1)
            weekly = net.day_of_week(raw["weekday"])[:, None].expand(-1, 34, -1)
            h = net.blocks(
                F.relu(
                    net.input(
                        torch.cat(
                            (
                                x.transpose(1, 2),
                                mask.float().transpose(1, 2),
                                identity,
                                daily,
                                weekly,
                            ),
                            -1,
                        )
                    )
                )
            )
            if z is not None:
                h = torch.cat((h, z.transpose(1, 2).flatten(2)), -1)
            result = net.output(h).transpose(1, 2)
        elif self.name == "staeformer":
            calendar = raw["calendar"]
            h = torch.cat(
                (
                    net.feature(torch.stack((x, mask.float()), -1)),
                    net.time_of_day(calendar[..., 0])[:, :, None].expand(
                        -1, -1, 34, -1
                    ),
                    net.day_of_week(calendar[..., 1])[:, :, None].expand(
                        -1, -1, 34, -1
                    ),
                    net.adaptive[None].expand(batch, -1, -1, -1),
                ),
                -1,
            )
            for layer in net.temporal:
                h = (
                    layer(h.transpose(1, 2).reshape(batch * 34, 24, 56))
                    .reshape(batch, 34, 24, 56)
                    .transpose(1, 2)
                )
            for layer in net.spatial:
                h = layer(h.reshape(batch * 24, 34, 56)).reshape(batch, 24, 34, 56)
            h = h.transpose(1, 2).flatten(2)
            if z is not None:
                h = torch.cat((h, z.transpose(1, 2).flatten(2)), -1)
            result = net.head(h).transpose(1, 2)
        else:
            calendar = time_features(raw, 288)[:, None].expand(-1, 34, -1)
            features = torch.cat(
                (
                    x[..., None],
                    mask.float()[..., None],
                    calendar[:, None].expand(-1, 24, -1, -1),
                ),
                -1,
            )
            hidden = x.new_zeros(batch, 34, 16)
            for step in range(24):
                graph, _, _, _ = net.memory_graph(hidden)
                hidden = net.encoder(features[:, step], hidden, graph)
            _, query, _, similarity = net.memory_graph(hidden)
            nearest = similarity.topk(2, dim=-1).indices
            positive, negative = net.bank[nearest[..., 0]], net.bank[nearest[..., 1]]
            regularizer = 0.01 * (
                (query - positive).square().mean()
                + F.relu(
                    (query - positive).square().sum(-1)
                    - (query - negative).square().sum(-1)
                    + net.margin
                ).mean()
            )
            previous, outputs = features[:, -1], []
            for step in range(6):
                graph, _, memory, _ = net.memory_graph(hidden)
                hidden = net.decoder(previous, hidden, graph)
                joined = torch.cat((hidden, memory), -1)
                if z is not None:
                    joined = torch.cat((joined, z[:, step]), -1)
                value = net.projection(joined)
                outputs.append(value.squeeze(-1))
                previous = torch.cat((value, torch.ones_like(value), calendar), -1)
            result = torch.stack(outputs, 1)
        point = (result * self.std + self.mean).clamp_min(0)
        if self.n_adapter is not None:
            point = self.n_adapter(raw, point, self.mean, self.std)
        aux["memory_regularizer"] = regularizer
        return (point, aux) if return_aux else point
