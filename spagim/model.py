"""SpaGIM: node-adaptive implicit graph neural networks via bilevel optimization.

The classifier reads the corrected representation W = Z* - U*. Here Z* is the equilibrium of the last implicit layer,
Z* = f(Z*; Z0), and U* solves the lower-level problem (Eq. 3 of the paper)

    U* in argmin_U  1/2 ||r(Z* - U) - B||_F^2 + lam sum_i ||U_i||_2,
    r(V) = f(V; Z0) - V,   B = diag(sigma) r(Z0),   lam = lam_bar ||r(Z0)||_F / sqrt(n).

The node budgets sigma_i = sigmoid(a_phi(d_i)) come from a small network on six label-free node statistics d_i. The
lower problem is solved approximately by FISTA with gradient restarts. The upper level (training cross-entropy) is
differentiated implicitly: equilibria through their adjoint equations, and U* through the fixed point of one
proximal-gradient step T (Algorithm 1). Equilibria and lower solutions are warm-started across calls.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


def fixed_point(f, z, tol, max_iter, check=1):
    """Iterate z <- f(z) from z until the relative change is at most tol, testing every `check` iterations (and at
    the last). Returns z and the number of iterations."""
    for k in range(1, max_iter + 1):
        z_next = f(z)
        converged = (k % check == 0 or k == max_iter) and bool((z_next - z).norm() <= tol * z_next.norm())
        z = z_next
        if converged:
            break
    if not bool(torch.isfinite(z).all()):
        raise RuntimeError("fixed-point iteration diverged (non-finite iterate)")
    return z, k


def shrink(V, tau):
    """The row-wise group soft-threshold (Eq. 5), the proximal operator of tau sum_i ||v_i||_2: rows with
    ||v_i|| <= tau become zero, and the others shrink by tau along their direction. tau = 0 returns V."""
    norm = V.norm(dim=1, keepdim=True)
    return V * (1.0 - tau / norm.clamp_min(tau).clamp_min(1e-30))


# ------------------------------------------------------------------------------------------- equilibrium solvers
class LinearSolve(torch.autograd.Function):
    """Z = gamma S Z B + Y by fixed-point iteration, with S and B symmetric. The backward pass solves the adjoint
    a = gamma S a B + dZ by the same iteration and returns dY = a (implicit gradients, no unrolling)."""

    @staticmethod
    def forward(ctx, Y, B, S, gamma, warm, tol, max_iter, check=1, btol=None):
        Z, _ = fixed_point(lambda Z: gamma * torch.sparse.mm(S, Z) @ B + Y, Y if warm is None else warm, tol, max_iter,
                           check)
        ctx.save_for_backward(Z, B)
        ctx.S, ctx.gamma, ctx.tol, ctx.max_iter = S, gamma, tol if btol is None else btol, max_iter
        ctx.check = check
        return Z

    @staticmethod
    def backward(ctx, dZ):
        Z, B = ctx.saved_tensors
        S, gamma = ctx.S, ctx.gamma
        a, _ = fixed_point(lambda A: gamma * torch.sparse.mm(S, A) @ B + dZ, dZ, ctx.tol, ctx.max_iter, ctx.check)
        dB = gamma * torch.sparse.mm(S, Z).T @ a if ctx.needs_input_grad[1] else None
        return a, dB, None, None, None, None, None, None, None


class ReLUSolve(torch.autograd.Function):
    """Z = ReLU(S Z W^T + Z0) by fixed-point iteration. With M = 1[S Z W^T + Z0 > 0] at the fixed point, the backward
    pass solves the adjoint a = S (M * a) W + dZ by the same iteration and returns dZ0 = M * a and dW = (M * a)^T S Z."""

    @staticmethod
    def forward(ctx, Z0, W, S, warm, tol, max_iter, check=1, btol=None):
        Z, _ = fixed_point(lambda Z: F.relu(torch.sparse.mm(S, Z @ W.T) + Z0), F.relu(Z0) if warm is None else warm,
                           tol, max_iter, check)
        SZ = torch.sparse.mm(S, Z)
        ctx.save_for_backward(SZ, W, (SZ @ W.T + Z0 > 0).to(Z.dtype))
        ctx.S, ctx.tol, ctx.max_iter, ctx.check = S, tol if btol is None else btol, max_iter, check
        return Z

    @staticmethod
    def backward(ctx, dZ):
        SZ, W, M = ctx.saved_tensors
        S = ctx.S
        a, _ = fixed_point(lambda A: torch.sparse.mm(S, (M * A) @ W) + dZ, dZ, ctx.tol, ctx.max_iter, ctx.check)
        Ma = M * a
        return Ma, Ma.T @ SZ if ctx.needs_input_grad[1] else None, None, None, None, None, None, None


class MeanSolve(torch.autograd.Function):
    """Z = ReLU(P Z W^T + Z0) with the neighbor-mean matrix P (not symmetric), by fixed-point iteration. The backward
    pass solves a = P^T (M * a) W + dZ (Proposition 3.6) and returns dZ0 = M * a and dW = (M * a)^T P Z."""

    @staticmethod
    def forward(ctx, Z0, W, P, PT, warm, tol, max_iter, check=1, btol=None):
        Z, _ = fixed_point(lambda Z: F.relu(torch.sparse.mm(P, Z @ W.T) + Z0), F.relu(Z0) if warm is None else warm,
                           tol, max_iter, check)
        PZ = torch.sparse.mm(P, Z)
        ctx.save_for_backward(PZ, W, (PZ @ W.T + Z0 > 0).to(Z.dtype))
        ctx.PT, ctx.tol, ctx.max_iter, ctx.check = PT, tol if btol is None else btol, max_iter, check
        return Z

    @staticmethod
    def backward(ctx, dZ):
        PZ, W, M = ctx.saved_tensors
        PT = ctx.PT
        a, _ = fixed_point(lambda A: torch.sparse.mm(PT, (M * A) @ W) + dZ, dZ, ctx.tol, ctx.max_iter, ctx.check)
        Ma = M * a
        return Ma, Ma.T @ PZ if ctx.needs_input_grad[1] else None, None, None, None, None, None, None, None


def plain_forward(layer, X, S):
    """An implicit layer used inside the stack: its equilibrium on the input X, warm-started from its last one (with
    layer.mode_warm, from its last one in the same mode, train or eval)."""
    if not layer.mode_warm:
        H = layer.equilibrium(layer.inject(X, S), S, warm=layer.warm)
        layer.warm = H.detach()
        return H
    if not hasattr(layer, "warms"):
        layer.warms = {}
    H = layer.equilibrium(layer.inject(X, S), S, warm=layer.warms.get(layer.training))
    layer.warms[layer.training] = H.detach()
    return H


# ------------------------------------------------------------------------------------------------ implicit layers
class MIGNNLayer(nn.Module):
    """f(Z; Z0) = ReLU(A Z W^T + Z0) with the normalized adjacency A (Eq. 1) and the contractive weight
    W = kappa sigmoid(mu) (I - C)(I + C)^{-1} Sigma (Cayley parametrization; every singular value is
    kappa sigmoid(mu) < 1). The injection Z0 is A X Omega^T ("propagated"), X Omega^T ("own"), or
    X Omega^T + A X Omega2^T ("combined")."""

    check, btol, mode_warm = 1, None, False  # solver options, set by SpaGIM

    def __init__(self, in_dim, hidden, kappa=0.95, tol=1e-5, max_iter=300, injection="propagated",
                 signature_share=0.2, mu_init=1.0):
        super().__init__()
        self.kappa, self.tol, self.max_iter = float(kappa), float(tol), int(max_iter)
        if not 0.0 < self.kappa < 1.0:
            raise ValueError("need 0 < kappa < 1")
        if injection not in ("propagated", "own", "combined"):
            raise ValueError('injection must be "propagated", "own" or "combined"')
        self.injection = injection
        self.warm = None  # the last equilibrium, as a warm start
        bound = 1.0 / math.sqrt(hidden)
        self.skew = nn.Parameter(torch.empty(hidden * (hidden - 1) // 2).uniform_(-bound, bound))  # C, below diagonal
        self.mu = nn.Parameter(torch.full((1,), float(mu_init)))
        self.Omega = nn.Parameter(torch.empty(hidden, in_dim).uniform_(-bound, bound))
        if injection == "combined":
            self.Omega2 = nn.Parameter(torch.empty(hidden, in_dim).uniform_(-bound, bound))
        signs = torch.ones(hidden)
        signs[: int(signature_share * hidden)] = -1.0
        self.register_buffer("signature", signs[torch.randperm(hidden)])  # Sigma
        self.register_buffer("lower", torch.tril_indices(hidden, hidden, -1), persistent=False)

    def weight(self):
        """W = kappa sigmoid(mu) (I - C)(I + C)^{-1} Sigma."""
        h = self.signature.numel()
        L = self.skew.new_zeros(h, h).index_put((self.lower[0], self.lower[1]), self.skew)
        C = L - L.T
        eye = torch.eye(h, dtype=C.dtype, device=C.device)
        return self.kappa * torch.sigmoid(self.mu) * torch.linalg.solve(eye + C, eye - C, left=False) * self.signature

    def inject(self, X, S):
        Z0 = X @ self.Omega.T
        if self.injection == "own":
            return Z0
        if self.injection == "combined":
            return Z0 + torch.sparse.mm(S, X @ self.Omega2.T)
        return torch.sparse.mm(S, Z0)

    def f(self, Z, Z0, S):
        return F.relu(torch.sparse.mm(S, Z @ self.weight().T) + Z0)

    def equilibrium(self, Z0, S, warm=None):
        return ReLUSolve.apply(Z0, self.weight(), S, warm, self.tol, self.max_iter, self.check, self.btol)

    def propagation(self):
        """What lower_grad takes as its last argument: W."""
        return self.weight()

    def lower_grad(self, V, Z0, b, S, W):
        """grad h(U) = (I - J_f(V))^T (r(V) - b) at V = Z* - U, with J_f(V)^T E = A (M * E) W and
        M = 1[A V W^T + Z0 > 0]."""
        P = torch.sparse.mm(S, V @ W.T) + Z0
        E = F.relu(P) - V - b
        return E - torch.sparse.mm(S, (P > 0).to(E.dtype) * E) @ W

    @torch.no_grad()
    def lipschitz(self):
        """beta >= ||J_f||_2: ||W||_2 = kappa sigmoid(mu), since ||A||_2 = 1."""
        return self.kappa * float(torch.sigmoid(self.mu))

    def forward(self, X, S):
        return plain_forward(self, X, S)


class MeanLayer(MIGNNLayer):
    """Mean aggregation (Eq. 8): f(Z; Z0) = ReLU(P Z W^T + Z0) with the neighbor mean P = D^-1 Adj and MIGNN's W.
    The injection uses P in place of A. P's rows sum to at most one, so f contracts in max_i ||Z_i||_2
    (Proposition 3.5); ||P||_2 can exceed one, so the lower step uses beta = ||P||_2 kappa sigmoid(mu)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_ops = None  # (S, P, P^T, ||P||_2), computed once per graph

    def means(self, S):
        """(P, P^T, ||P||_2) for the graph of S (S's pattern without the diagonal)."""
        if self.mean_ops is None or self.mean_ops[0] is not S:
            crow, col = S.crow_indices(), S.col_indices()
            rows = torch.repeat_interleave(torch.arange(S.shape[0], device=col.device), crow[1:] - crow[:-1])
            keep = rows != col
            rows, cols = rows[keep], col[keep]
            dtype = S.values().dtype
            degree = torch.zeros(S.shape[0], device=col.device, dtype=dtype).index_add_(
                0, rows, torch.ones(len(rows), device=col.device, dtype=dtype))
            values = 1.0 / degree[rows].clamp_min(1)
            P = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, S.shape).coalesce()
            PT = torch.sparse_coo_tensor(torch.stack([cols, rows]), values, S.shape).coalesce()
            v = torch.ones(S.shape[0], 1, device=col.device, dtype=dtype)
            P, PT = P.to_sparse_csr(), PT.to_sparse_csr()
            norm = 0.0
            for _ in range(500):  # power iteration on P^T P
                w = torch.sparse.mm(PT, torch.sparse.mm(P, v))
                new = float(w.norm()) ** 0.5
                v = w / w.norm()
                if abs(new - norm) <= 1e-7 * new:
                    break
                norm = new
            self.mean_ops = (S, P, PT, 1.0001 * new)
        return self.mean_ops[1:]

    def inject(self, X, S):
        P = self.means(S)[0]
        Z0 = X @ self.Omega.T
        if self.injection == "own":
            return Z0
        if self.injection == "combined":
            return Z0 + torch.sparse.mm(P, X @ self.Omega2.T)
        return torch.sparse.mm(P, Z0)

    def f(self, Z, Z0, S):
        return F.relu(torch.sparse.mm(self.means(S)[0], Z @ self.weight().T) + Z0)

    def equilibrium(self, Z0, S, warm=None):
        P, PT, _ = self.means(S)
        return MeanSolve.apply(Z0, self.weight(), P, PT, warm, self.tol, self.max_iter, self.check, self.btol)

    def lower_grad(self, V, Z0, b, S, W):
        P, PT, _ = self.means(S)
        pre = torch.sparse.mm(P, V @ W.T) + Z0
        E = F.relu(pre) - V - b
        return E - torch.sparse.mm(PT, (pre > 0).to(E.dtype) * E) @ W

    @torch.no_grad()
    def lipschitz(self):
        scale = 1.0 if self.mean_ops is None else self.mean_ops[3]
        return scale * self.kappa * float(torch.sigmoid(self.mu))


class PPRLayer(nn.Module):
    """Personalized PageRank (Appendix B): f(Z; Z0) = gamma A Z + Z0 with Z0 = X Omega^T and fixed gamma, i.e.
    restart probability 1 - gamma. Z* = (I - gamma A)^{-1} X Omega^T is APPNP's propagation scaled by 1 / (1 - gamma).
    The map is affine, so the lower problem is a strongly convex group lasso."""

    check, btol, mode_warm = 1, None, False  # solver options, set by SpaGIM

    def __init__(self, in_dim, hidden, gamma=0.9, tol=1e-5, max_iter=300):
        super().__init__()
        self.gamma, self.tol, self.max_iter = float(gamma), float(tol), int(max_iter)
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("need 0 < gamma < 1")
        self.W = nn.Linear(in_dim, hidden, bias=False)  # Omega
        self.register_buffer("eye", torch.eye(hidden), persistent=False)
        self.warm = None

    def inject(self, X, S):
        return self.W(X)

    def f(self, Z, Z0, S):
        return self.gamma * torch.sparse.mm(S, Z) @ self.eye + Z0

    def equilibrium(self, Z0, S, warm=None):
        return LinearSolve.apply(Z0, self.eye, S, self.gamma, warm, self.tol, self.max_iter, self.check, self.btol)

    def propagation(self):
        return self.eye

    def lower_grad(self, V, Z0, b, S, B):
        """(I - J)^T (r(V) - b) with the self-adjoint J = gamma A (.) B."""
        E = self.gamma * torch.sparse.mm(S, V) @ B + Z0 - V - b
        return E - self.gamma * torch.sparse.mm(S, E) @ B

    @torch.no_grad()
    def lipschitz(self):
        return self.gamma

    def forward(self, X, S):
        return plain_forward(self, X, S)


class SharedScore(nn.Module):
    """The shared-budget ablation: one learned score for every node."""

    def __init__(self, outputs):
        super().__init__()
        self.score = nn.Parameter(torch.zeros(outputs))

    def forward(self, D):
        return self.score.expand(D.shape[0], -1)


# ------------------------------------------------------------------------------------------------------- the model
class SpaGIM(nn.Module):
    """SpaGIM (Section 3). The backbone stacks `layers` implicit layers; the layers before the last are plain
    equilibria (with batch norm and dropout between them if batch_norm), and the departures act on the last.

    predict() leaves diagnostics in self.stats: the share of departing nodes under the reporting cutoff
    ||U_i|| > 0.01 ||Z*||_F / sqrt(n) ("departing"), ||U*||_F / ||Z*||_F ("relative_departure"), the FISTA steps of
    the call ("lower_steps") and the mean budget ("sigma_mean"); train_step adds the adjoint iterations
    ("adjoint_steps")."""

    def __init__(self, in_dim, out_dim, n_nodes,
                 hidden=512,  # width m
                 backbone="mignn",  # the last implicit layer: "mignn" (Eq. 1), "mean" (Eq. 8) or "ppr" (Appendix B)
                 stack_backbone=None,  # the layers before the last (None: the same as `backbone`)
                 layers=1,  # number of stacked implicit layers L
                 injection="propagated",  # "propagated", "own" or "combined" (MIGNN and mean layers)
                 kappa=0.95,  # fixed contraction cap of the MIGNN parametrization
                 mu_init=1.0,  # initial mu; the contraction factor is kappa sigmoid(mu)
                 gamma=0.9,  # PPR propagation weight (restart probability 1 - gamma)
                 width="hidden",  # "hidden", or "classes": the last layer works on class logits (no head)
                 input_map=False,  # a linear input map Omega_in (with dropout) before the stack
                 mlp_layers=0,  # ReLU layers (with dropout) before the stack, as in the decoupled citation variant
                 batch_norm=False,  # batch norm and dropout between stacked equilibria
                 head="linear",  # the classifier c_omega: "linear" or "mlp" (one hidden ReLU layer)
                 dropout=0.5,  # dropout on the classifier input, the input map, the MLP and between equilibria
                 input_dropout=0.0,  # dropout on the input features
                 lam=0.3,  # penalty coefficient lam_bar; the penalty is lam_bar ||r(Z0)||_F / sqrt(n)
                 departure="bilevel",  # "bilevel" (Eq. 3); ablations: "gate", "thresholded_gate"
                 shared_budget=False,  # ablation: one learned budget for every node instead of the allocator
                 allocator_width=16,  # hidden units of the allocator a_phi
                 lower_steps=30,  # FISTA steps per call, at most ...
                 lower_tol=1e-3,  # ... stopping when the relative change of U falls below this
                 adjoint_steps=30,  # iterations of the lower adjoint equation, at most
                 tol=1e-5,  # equilibrium solves stop at this relative change ...
                 max_iter=300,  # ... or after this many iterations
                 check_every=1,  # test equilibrium convergence every this many iterations
                 backward_tol=None,  # tolerance of the equilibrium adjoint solves (None: tol)
                 separate_warm_starts=False):  # separate warm starts for training and evaluation
        super().__init__()
        names = ("mignn", "mean", "ppr")
        if backbone not in names or (stack_backbone is not None and stack_backbone not in names):
            raise ValueError(f"backbone and stack_backbone must be in {names}")
        if width not in ("hidden", "classes") or head not in ("linear", "mlp") or layers < 1 or mlp_layers < 0:
            raise ValueError('need width in ("hidden", "classes"), head in ("linear", "mlp"), layers >= 1 and '
                             'mlp_layers >= 0')
        if departure not in ("bilevel", "gate", "thresholded_gate"):
            raise ValueError('departure must be "bilevel", "gate" or "thresholded_gate"')
        if not (0.0 <= dropout < 1.0 and 0.0 <= input_dropout < 1.0 and lam >= 0):
            raise ValueError("need 0 <= dropout, input_dropout < 1 and lam >= 0")
        if int(check_every) < 1 or (backward_tol is not None and float(backward_tol) <= 0):
            raise ValueError("need check_every >= 1 and backward_tol > 0 (or None)")
        self.dropout, self.input_dropout, self.lam = float(dropout), float(input_dropout), float(lam)
        self.departure, self.shared_budget = departure, bool(shared_budget)
        self.lower_steps, self.lower_tol, self.adjoint_steps = int(lower_steps), float(lower_tol), int(adjoint_steps)

        self.input_map = nn.Linear(in_dim, hidden) if input_map else None
        first = hidden if input_map else in_dim
        self.mlp = nn.ModuleList(nn.Linear(first if l == 0 else hidden, hidden) for l in range(mlp_layers))
        stack_in = hidden if input_map or mlp_layers > 0 else in_dim
        g = hidden if width == "hidden" else out_dim

        def make(width_in, kind):
            if kind == "ppr":
                return PPRLayer(width_in, g, gamma, tol, max_iter)
            cls = MeanLayer if kind == "mean" else MIGNNLayer
            return cls(width_in, g, kappa, tol, max_iter, injection, mu_init=mu_init)

        self.backbone = nn.ModuleList(
            make(stack_in if l == 0 else g, backbone if stack_backbone is None or l == layers - 1 else stack_backbone)
            for l in range(layers))
        self.batch_norms = nn.ModuleList(nn.BatchNorm1d(g) for _ in range(layers - 1)) if batch_norm else None
        for layer in self.backbone:
            layer.check, layer.btol, layer.mode_warm = int(check_every), backward_tol, bool(separate_warm_starts)
        self.classifier = (nn.Identity() if width == "classes" else nn.Linear(hidden, out_dim) if head == "linear"
                           else nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)))
        if self.shared_budget:
            self.allocator = SharedScore(1)
        else:  # a_phi: R^6 -> R, starting at sigma_i = 1/2 for every node
            self.allocator = nn.Sequential(nn.Linear(6, allocator_width), nn.ReLU(), nn.Linear(allocator_width, 1))
            nn.init.normal_(self.allocator[2].weight, std=0.01)
            nn.init.zeros_(self.allocator[2].bias)

        self.register_buffer("U", torch.zeros(n_nodes, g))  # the lower solution, carried between calls
        self.separate_warm_starts = bool(separate_warm_starts)
        self.warm, self.warms, self.U_modes = None, {}, {}
        self.static = None  # (X, the statistics that depend on X and the graph only)
        self.sigma = None
        self.stats = {}

    # ------------------------------------------------------------------------------------------ forward pieces
    def initial(self, X, S):
        """Z0, the injection of the last layer: input dropout, the input map or MLP, then the plain equilibria of the
        layers before the last (Algorithm 1, lines 1-6)."""
        H = F.dropout(X, self.input_dropout, training=self.training) if self.input_dropout > 0 else X
        if self.input_map is not None:
            H = F.dropout(self.input_map(H), self.dropout, training=self.training)
        for layer in self.mlp:
            H = F.dropout(F.relu(layer(H)), self.dropout, training=self.training)
        for l, layer in enumerate(self.backbone[:-1]):
            H = layer(H, S)
            if self.batch_norms is not None:
                H = F.dropout(self.batch_norms[l](H), self.dropout, training=self.training)
        return self.backbone[-1].inject(H, S)

    def equilibrium(self, X, S):
        """Z0, Z*, the target B = diag(sigma) r(Z0) and the penalty lam (Algorithm 1, lines 7-8)."""
        Z0 = self.initial(X, S)
        last = self.backbone[-1]
        if self.separate_warm_starts:
            Z = last.equilibrium(Z0, S, warm=self.warms.get(self.training))
            self.warms[self.training] = Z.detach()
        else:
            Z = last.equilibrium(Z0, S, warm=self.warm)
            self.warm = Z.detach()
        scores = self.allocator(Z0 if self.shared_budget else self.descriptors(X, S, Z0, Z))
        sigma = torch.sigmoid(scores[:, 0])
        R = last.f(Z0, Z0, S) - Z0  # r(Z0)
        self.sigma = sigma
        return Z0, Z, sigma[:, None] * R, self.lam * R.norm() / math.sqrt(Z.shape[0])

    @torch.no_grad()
    def descriptors(self, X, S, Z0, Z):
        """The allocator's six label-free inputs (Eq. 7), standardized over the nodes: log(1 + degree) (the self-loop
        of A is counted), cos(x_i, (A X)_i), ||x_i||, log ||r_i(Z0)||, log ||z*_i - z0_i|| and cos(z*_i, z0_i).
        Logarithms floor their argument at 1e-12, cosines use eps = 1e-8 (zero for a zero vector), and the standard
        deviation is floored at 1e-6. Computed without gradient (the stop-gradient of Eq. 6)."""
        if self.static is None or self.static[0] is not X:
            crow = S.crow_indices()
            degree = (crow[1:] - crow[:-1]).to(X.dtype)
            agree = F.cosine_similarity(X, torch.sparse.mm(S, X), dim=1)
            self.static = (X, torch.stack([degree.log(), agree, X.norm(dim=1)], 1))
        R0 = self.backbone[-1].f(Z0, Z0, S) - Z0
        D = torch.cat([self.static[1], torch.stack([R0.norm(dim=1).clamp_min(1e-12).log(),
                                                    (Z - Z0).norm(dim=1).clamp_min(1e-12).log(),
                                                    F.cosine_similarity(Z, Z0, dim=1)], 1)], 1)
        return (D - D.mean(0)) / D.std(0).clamp_min(1e-6)

    def mode_start(self, Z):
        """The lower solve's warm start: self.U, or with separate warm starts the last solution in this mode."""
        if not self.separate_warm_starts:
            return None
        start = self.U_modes.get(self.training)
        return torch.zeros_like(Z) if start is None else start

    @torch.no_grad()
    def solve_lower(self, Z, Z0, b, S, eta, tau, U=None):
        """FISTA with gradient restarts on h(U) + lam sum_i ||U_i||, step eta and threshold tau = eta lam, from U
        (default: the last solution), for at most lower_steps steps (Algorithm 1, line 9). Returns U and the steps."""
        last, tau = self.backbone[-1], float(tau)
        P = last.propagation()
        U = self.U if U is None else U
        Y, t = U, 1.0
        for k in range(1, self.lower_steps + 1):
            U_next = shrink(Y - eta * last.lower_grad(Z - Y, Z0, b, S, P), tau)
            if float(((Y - U_next) * (U_next - U)).sum()) > 0:  # the momentum points uphill: restart it
                t = 1.0
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            Y = U_next + ((t - 1.0) / t_next) * (U_next - U)
            done = bool((U_next - U).norm() <= self.lower_tol * U_next.norm())
            U, t = U_next, t_next
            if done:
                break
        if not bool(torch.isfinite(U).all()):
            raise RuntimeError("lower solve diverged (non-finite iterate)")
        return U, k

    def attach(self, U, Z, Z0, b, S, eta, tau):
        """One proximal-gradient step T(U) with autograd (Algorithm 1, line 10). Its value is (nearly) U, and a hook
        replaces the incoming gradient g by the solution a of a = (dT/dU)^T a + g, the implicit gradient at the fixed
        point U = T(U)."""
        last = self.backbone[-1]
        P = last.propagation()
        out = shrink(U - eta * last.lower_grad(Z - U, Z0, b, S, P), tau)
        if out.requires_grad:
            U0 = U.detach().clone().requires_grad_()
            T0 = shrink(U0 - eta * last.lower_grad(Z.detach() - U0, Z0.detach(), b.detach(), S, P.detach()),
                        tau.detach() if torch.is_tensor(tau) else tau)

            def adjoint(g):
                a, k = fixed_point(lambda a: torch.autograd.grad(T0, U0, a, retain_graph=True)[0] + g, g,
                                   self.lower_tol, self.adjoint_steps)
                self.stats["adjoint_steps"] = k
                return a

            out.register_hook(adjoint)
        return out

    def thresholded_gate(self, Z, Z0):
        """Ablation: the row soft-threshold of the gate's departure sigma_i (z*_i - z0_i) at
        tau = lam_bar ||Z* - Z0||_F / sqrt(n); no lower problem and no coupling through the propagation."""
        D = Z - Z0
        return shrink(self.sigma[:, None] * D, self.lam * D.norm() / math.sqrt(Z.shape[0]))

    # ------------------------------------------------------------------------------------- training and inference
    def train_step(self, X, S, y, train_mask, optimizer):
        """One upper-level step (one epoch, Algorithm 1). Returns the training cross-entropy."""
        self.train()
        Z0, Z, b, lam = self.equilibrium(X, S)
        if self.departure == "gate":  # ablation: W = (1 - sigma) Z* + sigma Z0
            U = self.sigma[:, None] * (Z - Z0)
        elif self.departure == "thresholded_gate":
            U = self.thresholded_gate(Z, Z0)
        else:
            eta = 1.0 / (1.0 + self.backbone[-1].lipschitz()) ** 2
            U, _ = self.solve_lower(Z.detach(), Z0.detach(), b.detach(), S, eta, eta * lam, U=self.mode_start(Z))
            U = self.attach(U, Z, Z0, b, S, eta, eta * lam)
            self.U.copy_(U.detach())
            if self.separate_warm_starts:
                self.U_modes[True] = U.detach()
        W = Z - U
        logits = self.classifier(F.dropout(W, self.dropout, training=True))
        loss = F.cross_entropy(logits[train_mask], y[train_mask])  # training labels only
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, X, S):
        """Class logits of every node at the current parameters (evaluation mode, no dropout)."""
        self.eval()
        Z0, Z, b, lam = self.equilibrium(X, S)
        if self.departure == "gate":
            U, steps = self.sigma[:, None] * (Z - Z0), 0
        elif self.departure == "thresholded_gate":
            U, steps = self.thresholded_gate(Z, Z0), 0
        else:
            eta = 1.0 / (1.0 + self.backbone[-1].lipschitz()) ** 2
            U, steps = self.solve_lower(Z, Z0, b, S, eta, eta * lam, U=self.mode_start(Z))
            self.U.copy_(U)
            if self.separate_warm_starts:
                self.U_modes[False] = U.detach()
        size = U.norm(dim=1)
        self.stats.update(departing=float((size > 0.01 * Z.norm() / math.sqrt(Z.shape[0])).float().mean()),
                          relative_departure=float(U.norm() / Z.norm()), lower_steps=steps,
                          sigma_mean=float(self.sigma.mean()))
        return self.classifier(Z - U)
