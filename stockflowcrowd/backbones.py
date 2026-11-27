"""Forecasting backbone layers adapted from the registered simulation models."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


class _PointBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)
        )

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class _STIDNet(nn.Module):
    def __init__(self, channels, width, identity_width, blocks):
        super().__init__()
        self.channel_id = nn.Embedding(channels, identity_width)
        self.time_of_day = nn.Embedding(288, identity_width)
        self.day_of_week = nn.Embedding(7, identity_width)
        self.input = nn.Linear(48 + 3 * identity_width, width)
        self.blocks = nn.Sequential(*[_PointBlock(width) for _ in range(blocks)])
        self.output = nn.Linear(width, 6)


class _STAEformerNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        feature = int(cfg["feature_dim"])
        adaptive = int(cfg["adaptive_dim"])
        width = 3 * feature + adaptive
        heads = int(cfg["heads"])
        if width % heads:
            raise ValueError("STAEformer embedding width must divide attention heads")
        self.feature = nn.Linear(2, feature)
        self.time_of_day = nn.Embedding(288, feature)
        self.day_of_week = nn.Embedding(7, feature)
        # Joint chronological/node identity shared across all snapshots.
        self.adaptive = nn.Parameter(torch.empty(24, 34, adaptive))
        nn.init.xavier_uniform_(self.adaptive)

        def layer():
            return nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=2 * width,
                dropout=float(cfg["dropout"]),
                batch_first=True,
                norm_first=True,
            )

        self.temporal = nn.ModuleList(layer() for _ in range(int(cfg["layers"])))
        self.spatial = nn.ModuleList(layer() for _ in range(int(cfg["layers"])))
        self.head = nn.Linear(24 * width, 6)


class _GraphGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        combined = input_dim + hidden_dim
        self.gates = nn.Linear(2 * combined, 2 * hidden_dim)
        self.candidate = nn.Linear(2 * combined, hidden_dim)

    @staticmethod
    def _neighbors(values, graph):
        return torch.bmm(graph, values)

    def forward(self, x, hidden, graph):
        joined = torch.cat((x, hidden), dim=-1)
        gates = torch.sigmoid(
            self.gates(
                torch.cat(
                    (joined, self._neighbors(joined, graph)),
                    dim=-1,
                )
            )
        )
        update, reset = gates.chunk(2, dim=-1)
        candidate_input = torch.cat((x, reset * hidden), dim=-1)
        candidate = torch.tanh(
            self.candidate(
                torch.cat(
                    (candidate_input, self._neighbors(candidate_input, graph)),
                    dim=-1,
                )
            )
        )
        return update * hidden + (1 - update) * candidate


class _MegaCRNNet(nn.Module):
    def __init__(self):
        super().__init__()
        hidden, memory, slots = 16, 8, 8
        self.hidden_dim = hidden
        self.margin = 1.0
        self.node_prior = nn.Parameter(torch.empty(34, memory))
        self.bank = nn.Parameter(torch.empty(slots, memory))
        nn.init.xavier_uniform_(self.node_prior)
        nn.init.xavier_uniform_(self.bank)
        self.query = nn.Linear(hidden, memory)
        self.encoder = _GraphGRUCell(11, hidden)
        self.decoder = _GraphGRUCell(11, hidden)
        self.projection = nn.Linear(hidden + memory, 1)

    def memory_graph(self, hidden):
        query = self.query(hidden) + self.node_prior[None]
        similarity = torch.einsum("bnm,km->bnk", query, self.bank)
        relevance = torch.softmax(similarity, dim=-1)
        memory_value = relevance @ self.bank
        # The graph is conditional on current hidden state but its bank and
        # parameters are frozen in eval/predict; no test-time fitting occurs.
        graph = torch.softmax(
            torch.relu(torch.bmm(memory_value, memory_value.transpose(1, 2))), dim=-1
        )
        return graph, query, memory_value, similarity


def init_named(net, seed):
    parameters = dict(net.named_parameters())
    with torch.no_grad():
        for name, p in parameters.items():
            token = "%d:%s:%s" % (seed, name, tuple(p.shape))
            rng = torch.Generator().manual_seed(
                int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")
                % (2**63 - 1)
            )
            if p.numel() == 0:
                continue
            if "norm" in name and p.ndim == 1:
                p.fill_(1 if name.endswith("weight") else 0)
            elif name.endswith(("in_proj_bias", "out_proj.bias")):
                p.zero_()
            elif name.endswith("adaptive") or name.endswith("in_proj_weight"):
                a, b = nn.init._calculate_fan_in_and_fan_out(p)
                p.uniform_(
                    -math.sqrt(6 / (a + b)), math.sqrt(6 / (a + b)), generator=rng
                )
            elif ".time_of_day." in name or ".day_of_week." in name:
                p.copy_(torch.randn(p.shape, generator=rng))
            elif p.ndim == 0:
                p.zero_()
            else:
                weight = (
                    parameters.get(name[:-4] + "weight") if name.endswith("bias") else p
                )
                fan = (
                    nn.init._calculate_fan_in_and_fan_out(weight)[0]
                    if weight is not None and weight.ndim > 1
                    else max(1, p.numel())
                )
                p.uniform_(-1 / math.sqrt(fan), 1 / math.sqrt(fan), generator=rng)


def normalized(self, raw):
    m = raw["history_mask"]
    last = self.median[None].expand(len(m), -1)
    gap = torch.full_like(last, 3)
    values = []
    for step in range(24):
        gap = torch.where(m[:, step], torch.zeros_like(gap), gap + 1)
        last = torch.where(m[:, step], raw["history"][:, step], last)
        values.append(torch.where(gap <= 2, last, self.median))
    return (torch.stack(values, 1) - self.mean) / self.std, m


def summary_tokens(raw, mean, std):
    values, mask, side = raw["history"], raw["history_mask"], raw["sidecar"]
    _, t, _ = values.shape
    index = (
        torch.where(mask, torch.arange(t, device=values.device)[None, :, None], -1)
        .max(1)
        .values
    )
    seen = index >= 0
    index = index.clamp_min(0)
    last = values.gather(1, index[:, None]).squeeze(1)
    last = torch.where(seen, last, mean)
    age = side[..., 0].gather(1, index[:, None]).squeeze(1)
    delay = side[..., 3].gather(1, index[:, None]).squeeze(1)
    age = torch.where(seen, age, 120.0)
    delay = torch.where(seen, delay, 0.0)
    x = torch.where(mask, (values - mean) / std, 0.0)
    parts = [(last - mean) / std, age / 120.0, seen.float(), delay / 60.0]
    for w in (1, 3, 6, 24):
        count = mask[:, -w:].sum(1)
        parts.extend((x[:, -w:].sum(1) / count.clamp_min(1), count / float(w)))
    return torch.stack(parts, -1)


def time_features(raw, slots):
    theta = raw["issue_slot"].float() * (2 * math.pi / slots)
    return torch.cat(
        (
            theta.sin()[:, None],
            theta.cos()[:, None],
            F.one_hot(raw["weekday"], 7).float(),
        ),
        -1,
    )
