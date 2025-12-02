#include <torch/extension.h>
#include <unordered_map>

// Helper: neighbor sum s1 = sum_{mu}(s_{x+mu}+s_{x-mu})
static torch::Tensor neighbor_sum(const torch::Tensor& s) {
    return torch::roll(s, { -1 }, { 2 }) + torch::roll(s, { 1 }, { 2 }) +
           torch::roll(s, { -1 }, { 3 }) + torch::roll(s, { 1 }, { 3 });
}

static inline torch::Tensor dot3(const torch::Tensor& a, const torch::Tensor& b) {
    return (a * b).sum(1);
}

static inline torch::Tensor roll2(const torch::Tensor& t, int dx, int dy) {
    return torch::roll(torch::roll(t, {dx}, {2}), {dy}, {3});
}

// Cache rolls of a base tensor (typically s) keyed by (dx,dy).
struct RollCache {
    const torch::Tensor& base;
    std::unordered_map<int64_t, torch::Tensor> cache;
    explicit RollCache(const torch::Tensor& b) : base(b) {}
    static int64_t key(int dx, int dy) {
        return (static_cast<int64_t>(dx & 0xffff) << 16) | static_cast<uint16_t>(dy & 0xffff);
    }
    torch::Tensor roll_base(int dx, int dy) {
        auto k = key(dx, dy);
        auto it = cache.find(k);
        if (it != cache.end()) return it->second;
        auto r = roll2(base, dx, dy);
        cache.emplace(k, r);
        return r;
    }
    torch::Tensor roll(const torch::Tensor& t, int dx, int dy) {
        if (t.data_ptr() == base.data_ptr()) {
            return roll_base(dx, dy);
        }
        return roll2(t, dx, dy);
    }
};

// Fast psi0/psi2 helpers reusing a provided s1 = neighbor_sum(s)
static inline torch::Tensor psi0_action_fast(const torch::Tensor& s, const torch::Tensor& s1) {
    return (s * s1).flatten(1).sum(-1);
}

static inline torch::Tensor psi0_grad_fast(const torch::Tensor& s, const torch::Tensor& s1) {
    return -2.0 * torch::cross(s1, s, 1);
}

static inline torch::Tensor psi0_lapl_fast(const torch::Tensor& s, const torch::Tensor& s1) {
    (void)s; // unused
    return -4.0 * psi0_action_fast(s, s1);
}

static inline torch::Tensor psi0_grad_lapl_fast(const torch::Tensor& s, const torch::Tensor& s1) {
    return -4.0 * psi0_grad_fast(s, s1);
}

static inline torch::Tensor psi2_action_fast(const torch::Tensor& s, const torch::Tensor& s1) {
    auto acc = torch::zeros({s.size(0)}, s.options());
    acc = acc + (s * torch::roll(s1, { -1 }, { 2 })).flatten(1).sum(-1);
    acc = acc + (s * torch::roll(s1, { -1 }, { 3 })).flatten(1).sum(-1);
    auto V = torch::full_like(acc, 2.0);
    V = V * static_cast<double>(s.size(2));
    V = V * static_cast<double>(s.size(3));
    acc = acc - V;  // subtract the constant
    return 2.0 * acc;
}

static inline torch::Tensor psi2_grad_fast(const torch::Tensor& s, const torch::Tensor& s1) {
    auto F = torch::roll(s1, { 1 }, { 2 }) + torch::roll(s1, { -1 }, { 2 }) +
             torch::roll(s1, { 1 }, { 3 }) + torch::roll(s1, { -1 }, { 3 });
    return -2.0 * torch::cross(F, s, 1);
}

static inline torch::Tensor psi2_lapl_fast(const torch::Tensor& s, const torch::Tensor& s1) {
    (void)s;
    return -4.0 * psi2_action_fast(s, s1);
}

static inline torch::Tensor psi2_grad_lapl_fast(const torch::Tensor& s, const torch::Tensor& s1) {
    return -4.0 * psi2_grad_fast(s, s1);
}

// Psi21 fast grad using provided s1=sum nn, s2=neighbor_sum(s1)
static inline torch::Tensor psi21_grad_fast(const torch::Tensor& s, const torch::Tensor& s1, const torch::Tensor& s2) {
    auto d1 = dot3(s, s1).unsqueeze(1);
    auto d2 = dot3(s, s2).unsqueeze(1);
    auto V = s2 * d1;
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    torch::Tensor tmp = torch::zeros_like(s);
    for (auto& d: dirs) tmp += roll2(s * d1, d[0], d[1]);
    for (auto& d: dirs) V += roll2(tmp, d[0], d[1]);
    V = V + s1 * d2;
    for (auto& d: dirs) V += roll2(s * d2, d[0], d[1]);
    return torch::cross(s, V, 1);
}

// Psi12d fast using provided s1
static inline torch::Tensor psi12d_action_fast(const torch::Tensor& s, const torch::Tensor& s1) {
    auto M = dot3(s, s1);
    auto Q = dot3(s1, s1);
    return (M * Q).flatten(1).sum(-1);
}

static inline torch::Tensor psi12d_grad_fast(const torch::Tensor& s, const torch::Tensor& s1) {
    auto Q = dot3(s1, s1).unsqueeze(1);
    auto M = dot3(s, s1).unsqueeze(1);
    auto V = s1 * Q;
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    for (auto& d: dirs) {
        V += roll2(Q * s + 2.0 * M * s1, d[0], d[1]);
    }
    return torch::cross(s, V, 1);
}
// Psi0 action: sum_{x} s_x · (sum_{mu} s_{x+mu} + s_{x-mu})
torch::Tensor psi0_action(torch::Tensor s) {
    auto s1 = torch::zeros_like(s);
    s1 = s1 + (torch::roll(s, { -1 }, { 2 }) + torch::roll(s, { 1 }, { 2 }));
    s1 = s1 + (torch::roll(s, { -1 }, { 3 }) + torch::roll(s, { 1 }, { 3 }));
    return (s * s1).flatten(1).sum(-1);
}

// Psi0 grad: -2 * cross(s1, s)
torch::Tensor psi0_grad(torch::Tensor s) {
    auto s1 = torch::zeros_like(s);
    s1 = s1 + (torch::roll(s, { -1 }, { 2 }) + torch::roll(s, { 1 }, { 2 }));
    s1 = s1 + (torch::roll(s, { -1 }, { 3 }) + torch::roll(s, { 1 }, { 3 }));
    return -2.0 * torch::cross(s1, s, 1);
}

// Psi0 laplacian: -4 * action
torch::Tensor psi0_lapl(torch::Tensor s) {
    return -4.0 * psi0_action(s);
}

// Psi0 grad laplacian: -4 * grad
torch::Tensor psi0_grad_lapl(torch::Tensor s) {
    return -4.0 * psi0_grad(s);
}

// Psi2 action
torch::Tensor psi2_action(torch::Tensor s) {
    auto s1 = torch::zeros_like(s);
    s1 = s1 + (torch::roll(s, { -1 }, { 2 }) + torch::roll(s, { 1 }, { 2 }));
    s1 = s1 + (torch::roll(s, { -1 }, { 3 }) + torch::roll(s, { 1 }, { 3 }));
    auto acc = torch::zeros({s.size(0)}, s.options());
    acc = acc + (s * torch::roll(s1, { -1 }, { 2 })).flatten(1).sum(-1);
    acc = acc + (s * torch::roll(s1, { -1 }, { 3 })).flatten(1).sum(-1);

    // Match Python ordering: start from 2.0 and multiply by each spatial extent
    auto V = torch::full_like(acc, 2.0);
    V = V * static_cast<double>(s.size(2));
    V = V * static_cast<double>(s.size(3));
    acc = acc - V;  // subtract the constant
    return 2.0 * acc;
}

// Psi2 grad
torch::Tensor psi2_grad(torch::Tensor s) {
    auto s1 = torch::zeros_like(s);
    s1 = s1 + (torch::roll(s, { -1 }, { 2 }) + torch::roll(s, { 1 }, { 2 }));
    s1 = s1 + (torch::roll(s, { -1 }, { 3 }) + torch::roll(s, { 1 }, { 3 }));
    auto F = torch::roll(s1, { 1 }, { 2 }) + torch::roll(s1, { -1 }, { 2 }) +
             torch::roll(s1, { 1 }, { 3 }) + torch::roll(s1, { -1 }, { 3 });
    return -2.0 * torch::cross(F, s, 1);
}

torch::Tensor psi2_lapl(torch::Tensor s) {
    return -4.0 * psi2_action(s);
}

torch::Tensor psi2_grad_lapl(torch::Tensor s) {
    return -4.0 * psi2_grad(s);
}

// Psi11_l action
torch::Tensor psi11l_action(torch::Tensor s) {
    auto acc = torch::zeros({s.size(0)}, s.options());
    // mu = x
    auto b1 = torch::einsum("bsxy,bsxy->bxy", {s, torch::roll(s, { -1 }, { 2 })});
    acc = acc + 2.0 * torch::einsum("bxy,bxy->b", {b1, b1});
    // mu = y
    auto b2 = torch::einsum("bsxy,bsxy->bxy", {s, torch::roll(s, { -1 }, { 3 })});
    acc = acc + 2.0 * torch::einsum("bxy,bxy->b", {b2, b2});
    // Match Python ordering: start from 4/3 and multiply by each spatial extent
    auto V = torch::full_like(acc, 4.0 / 3.0);
    V = V * static_cast<double>(s.size(2));
    V = V * static_cast<double>(s.size(3));
    return acc - V;  // subtract the constant
}

// Psi11_l grad
torch::Tensor psi11l_grad(torch::Tensor s) {
    auto F = torch::zeros_like(s);
    // mu = x (dim=2)
    {
        auto sp = torch::roll(s, { +1 }, { 2 });
        auto sm = torch::roll(s, { -1 }, { 2 });
        auto bm = (s * sp).sum(1);                  // B,X,Y
        auto bp = torch::roll(bm, { -1 }, { 1 });   // shift -1 in x
        F.addcmul_(sp, bm.unsqueeze(1));
        F.addcmul_(sm, bp.unsqueeze(1));
    }
    // mu = y (dim=3)
    {
        auto sp = torch::roll(s, { +1 }, { 3 });
        auto sm = torch::roll(s, { -1 }, { 3 });
        auto bm = (s * sp).sum(1);                  // B,X,Y
        auto bp = torch::roll(bm, { -1 }, { 2 });   // shift -1 in y
        F.addcmul_(sp, bm.unsqueeze(1));
        F.addcmul_(sm, bp.unsqueeze(1));
    }
    return 4.0 * torch::cross(s, F, 1);
}

torch::Tensor psi11l_lapl(torch::Tensor s) {
    return -12.0 * psi11l_action(s);
}

torch::Tensor psi11l_grad_lapl(torch::Tensor s) {
    return -12.0 * psi11l_grad(s);
}

// ---- Psi11_t helper ----
torch::Tensor psi11t_action(torch::Tensor s) {
    auto s1 = neighbor_sum(s);
    auto M = dot3(s, s1);
    return (M * M).flatten(1).sum(-1);
}

torch::Tensor psi11t_grad(torch::Tensor s) {
    auto s1 = neighbor_sum(s);
    auto M = dot3(s, s1).unsqueeze(1);
    auto sM = s * M;
    auto V = neighbor_sum(sM) + s1 * M;
    return 2.0 * torch::cross(s, V, 1);
}

torch::Tensor psi11t_lapl(torch::Tensor s) {
    // -2*psi11l + 2*psi2 + constant -10*action
    auto A = -2.0 * psi11l_action(s) + 2.0 * psi2_action(s);
    auto Vconst = torch::full_like(A, 40.0 / 3.0);
    Vconst = Vconst * static_cast<double>(s.size(2));
    Vconst = Vconst * static_cast<double>(s.size(3));
    A = A + Vconst;
    return A - 10.0 * psi11t_action(s);
}

torch::Tensor psi11t_grad_lapl(torch::Tensor s) {
    return -2.0 * psi11l_grad(s) + 2.0 * psi2_grad(s) - 10.0 * psi11t_grad(s);
}

// ---- Psi11 ----
torch::Tensor psi11_action(torch::Tensor s) {
    auto A = psi11t_action(s) - psi11l_action(s) - (1.0 / 3.0) * psi2_action(s);
    auto Vconst = torch::full_like(A, 4.0 / 3.0);
    Vconst = Vconst * static_cast<double>(s.size(2));
    Vconst = Vconst * static_cast<double>(s.size(3));
    return A - Vconst;
}

torch::Tensor psi11_grad(torch::Tensor s) {
    return psi11t_grad(s) - psi11l_grad(s) - (1.0 / 3.0) * psi2_grad(s);
}

torch::Tensor psi11_lapl(torch::Tensor s) {
    return psi11t_lapl(s) - psi11l_lapl(s) - (1.0 / 3.0) * psi2_lapl(s);
}

torch::Tensor psi11_grad_lapl(torch::Tensor s) {
    return psi11t_grad_lapl(s) - psi11l_grad_lapl(s) - (1.0 / 3.0) * psi2_grad_lapl(s);
}

// Psi3
torch::Tensor psi3_action(torch::Tensor s) {
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    RollCache rc(s);
    auto acc = torch::zeros({s.size(0)}, s.options());
    for (auto& d1: dirs) for (auto& d2: dirs) for (auto& d3: dirs) {
        auto r = rc.roll(s, d1[0]+d2[0]+d3[0], d1[1]+d2[1]+d3[1]);
        acc += dot3(s, r).flatten(1).sum(-1);
    }
    return acc;
}

torch::Tensor psi3_grad(torch::Tensor s) {
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    RollCache rc(s);
    auto f = torch::zeros_like(s);
    for (auto& d1: dirs) for (auto& d2: dirs) for (auto& d3: dirs) {
        auto r = rc.roll(s, d1[0]+d2[0]+d3[0], d1[1]+d2[1]+d3[1]);
        f -= torch::cross(r, s, 1);
    }
    return 2.0 * f;
}

torch::Tensor psi3_lapl(torch::Tensor s) { return -4.0 * psi3_action(s); }
torch::Tensor psi3_grad_lapl(torch::Tensor s) { return -4.0 * psi3_grad(s); }

// Psi21
torch::Tensor psi21_action(torch::Tensor s) {
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    auto acc = torch::zeros({s.size(0), s.size(2), s.size(3)}, s.options());
    for (auto& d1: dirs) for (auto& d2: dirs) {
        auto d12 = dot3(s, roll2(s, d1[0]+d2[0], d1[1]+d2[1]));
        for (auto& d3: dirs) {
            acc += d12 * dot3(s, roll2(s, d3[0], d3[1]));
        }
    }
    return acc.flatten(1).sum(-1);
}

torch::Tensor psi21_grad(torch::Tensor s) {
    auto s1 = neighbor_sum(s);
    auto s2 = neighbor_sum(s1);  // matches Python (no -4*s term)
    auto d1 = dot3(s, s1).unsqueeze(1);
    auto d2 = dot3(s, s2).unsqueeze(1);
    auto V = s2 * d1;
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    // V += sum_{d} roll(sum_{d'} roll(s*d1,d'), d)
    torch::Tensor tmp = torch::zeros_like(s);
    for (auto& d: dirs) tmp += roll2(s * d1, d[0], d[1]);
    for (auto& d: dirs) V += roll2(tmp, d[0], d[1]);
    // V += s1 * d2 + sum_{d} roll(s*d2, d)
    V = V + s1 * d2;
    for (auto& d: dirs) V += roll2(s * d2, d[0], d[1]);
    return torch::cross(s, V, 1);
}

torch::Tensor psi21_lapl(torch::Tensor s) {
    return -(10 * psi21_action(s) - 2 * psi3_action(s) - 16 * psi0_action(s));
}
torch::Tensor psi21_grad_lapl(torch::Tensor s) {
    return -(10 * psi21_grad(s) - 2 * psi3_grad(s) - 16 * psi0_grad(s));
}

// ---- Psi12d ----
torch::Tensor psi12d_action(torch::Tensor s) {
    auto s1 = neighbor_sum(s);
    auto M = dot3(s, s1);
    auto Q = dot3(s1, s1);
    return (M * Q).flatten(1).sum(-1);
}

torch::Tensor psi12d_grad(torch::Tensor s) {
    auto s1 = neighbor_sum(s);
    auto Q = dot3(s1, s1).unsqueeze(1);
    auto M = dot3(s, s1).unsqueeze(1);
    auto V = s1 * Q;
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    for (auto& d: dirs) {
        V += roll2(Q * s + 2.0 * M * s1, d[0], d[1]);
    }
    return torch::cross(s, V, 1);
}

// ---- Psi12l ----
torch::Tensor psi12l_action(torch::Tensor s) {
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    RollCache rc(s);
    auto acc = torch::zeros({s.size(0), s.size(2), s.size(3)}, s.options());
    for (auto& d: dirs) {
        auto bond = dot3(s, rc.roll(s, d[0], d[1]));
        torch::Tensor inner = torch::zeros_like(bond);
        for (auto& w: dirs) {
            inner += dot3(s, rc.roll(s, d[0]+w[0], d[1]+w[1]));
        }
        acc += bond * inner;
    }
    return acc.flatten(1).sum(-1);
}

torch::Tensor psi12l_grad(torch::Tensor s) {
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    RollCache rc(s);
    auto V = torch::zeros_like(s);
    auto temp = torch::zeros_like(s);
    for (auto& d: dirs) {
        auto sd = rc.roll(s, d[0], d[1]);
        auto bond = dot3(s, sd).unsqueeze(1);
        torch::Tensor inner1 = torch::zeros_like(bond);
        torch::Tensor inner2 = torch::zeros_like(bond);
        torch::Tensor sumroll = torch::zeros_like(sd);
        for (auto& w: dirs) {
            inner1 += dot3(s, rc.roll(s, d[0]+w[0], d[1]+w[1])).unsqueeze(1);
            inner2 += dot3(sd, rc.roll(s, w[0], w[1])).unsqueeze(1);
            sumroll += rc.roll(s, d[0]+w[0], d[1]+w[1]);
        }
        V += sd * inner1 + sd * inner2;
        V += bond * sumroll;
        temp += bond * sd;
    }
    V += neighbor_sum(temp);
    return torch::cross(s, V, 1);
}

// ---- Psi111 ----
torch::Tensor psi111_action(torch::Tensor s) {
    auto s1 = neighbor_sum(s);
    auto M = dot3(s, s1);
    return (M * M * M).flatten(1).sum(-1);
}

torch::Tensor psi111_grad(torch::Tensor s) {
    auto s1 = neighbor_sum(s);
    auto M2 = (dot3(s, s1).unsqueeze(1)).pow(2);
    auto V = s1 * M2;
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    for (auto& d: dirs) V += roll2(s * M2, d[0], d[1]);
    return 3.0 * torch::cross(s, V, 1);
}

// ---- Psi111l ----
torch::Tensor psi111l_action(torch::Tensor s) {
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    auto acc = torch::zeros({s.size(0)}, s.options());
    for (auto& d: dirs) {
        auto b = dot3(s, roll2(s, d[0], d[1]));
        acc += (b * b * b).flatten(1).sum(-1);
    }
    return acc;
}

torch::Tensor psi111l_grad(torch::Tensor s) {
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    RollCache rc(s);
    auto V = torch::zeros_like(s);
    for (auto& d: dirs) {
        auto b = dot3(s, rc.roll(s, d[0], d[1])).unsqueeze(1);
        auto bneg = dot3(s, rc.roll(s, -d[0], -d[1])).unsqueeze(1);
        V += b * b * rc.roll(s, d[0], d[1]);
        V += bneg * bneg * rc.roll(s, -d[0], -d[1]);
    }
    return 3.0 * torch::cross(s, V, 1);
}

// ---- Psi1l1 ----
torch::Tensor psi1l1_action(torch::Tensor s) {
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    std::vector<torch::Tensor> bonds;
    for (auto& d: dirs) bonds.push_back(dot3(s, roll2(s, d[0], d[1])));
    auto K = torch::zeros_like(bonds[0]);
    auto S = torch::zeros_like(bonds[0]);
    for (auto& b: bonds) { K += b * b; S += b; }
    auto acc = K * S;
    return acc.flatten(1).sum(-1);
}

torch::Tensor psi1l1_grad(torch::Tensor s) {
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    RollCache rc(s);
    auto s1 = neighbor_sum(s);
    auto M = dot3(s, s1).unsqueeze(1);
    std::vector<torch::Tensor> bonds;
    for (auto& d: dirs) bonds.push_back(dot3(s, rc.roll(s, d[0], d[1])).unsqueeze(1));
    auto K = torch::zeros_like(bonds[0]);
    for (auto& b: bonds) K += b * b;
    auto V = torch::zeros_like(s);
    auto sumBroll = torch::zeros_like(s);
    for (size_t i = 0; i < 4; ++i) {
        sumBroll += bonds[i] * rc.roll(s, dirs[i][0], dirs[i][1]);
    }
    V += 2.0 * sumBroll * M;
    V += K * s1;
    for (size_t i = 0; i < 4; ++i) {
        auto dx = dirs[i][0], dy = dirs[i][1];
        auto term = (2 * bonds[i] * roll2(M, dx, dy) + roll2(K, dx, dy)) * rc.roll(s, dx, dy);
        V += term;
    }
    return torch::cross(s, V, 1);
}

// ---- Psi111c ----
torch::Tensor psi111c_action(torch::Tensor s) {
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    RollCache rc(s);
    auto acc = torch::zeros({s.size(0)}, s.options());
    for (auto& d1: dirs) {
        auto s1 = rc.roll(s, d1[0], d1[1]); auto b1 = dot3(s, s1);
        for (auto& d2: dirs) {
            auto s2 = rc.roll(s, d1[0]+d2[0], d1[1]+d2[1]); auto b2 = dot3(s1, s2);
            for (auto& d3: dirs) {
                auto s3 = rc.roll(s, d1[0]+d2[0]+d3[0], d1[1]+d2[1]+d3[1]); auto b3 = dot3(s2, s3);
                acc += (b1 * b2 * b3).flatten(1).sum(-1);
            }
        }
    }
    return acc;
}

torch::Tensor psi111c_grad(torch::Tensor s) {
    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    RollCache rc(s);
    auto V = torch::zeros_like(s);
    for (auto& d1: dirs) {
        auto s1 = rc.roll(s, d1[0], d1[1]); auto b0 = dot3(s, s1);
        for (auto& d2: dirs) {
            auto s2 = rc.roll(s, d1[0]+d2[0], d1[1]+d2[1]); auto b1 = dot3(s1, s2);
            for (auto& d3: dirs) {
                auto s3 = rc.roll(s, d1[0]+d2[0]+d3[0], d1[1]+d2[1]+d3[1]); auto b2 = dot3(s2, s3);
                auto p12 = (b1 * b2).unsqueeze(1);
                auto p02 = (b0 * b2).unsqueeze(1);
                auto p01 = (b0 * b1).unsqueeze(1);
                V += s1 * p12;
                V += roll2(s * p12 + s2 * p02, -d1[0], -d1[1]);
                V += roll2(s1 * p02 + s3 * p01, -(d1[0]+d2[0]), -(d1[1]+d2[1]));
                V += roll2(s2 * p01, -(d1[0]+d2[0]+d3[0]), -(d1[1]+d2[1]+d3[1]));
            }
        }
    }
    return torch::cross(s, V, 1);
}

// Laplacians/grad-laplacians using formulas
torch::Tensor psi12d_lapl(torch::Tensor s) {
    return -(8 * psi12d_action(s) - 32 * psi0_action(s) + 4 * psi12l_action(s));
}

torch::Tensor psi12d_grad_lapl(torch::Tensor s) {
    return -(8 * psi12d_grad(s) - 32 * psi0_grad(s) + 4 * psi12l_grad(s));
}

torch::Tensor psi12l_lapl(torch::Tensor s) {
    return -(10 * psi12l_action(s) - 12 * psi0_action(s));
}

torch::Tensor psi12l_grad_lapl(torch::Tensor s) {
    return -(10 * psi12l_grad(s) - 12 * psi0_grad(s));
}

torch::Tensor psi1l1_lapl(torch::Tensor s) {
    return -(20 * psi1l1_action(s) - 4 * psi12l_action(s) + 4 * psi111l_action(s) - 20 * psi0_action(s));
}

torch::Tensor psi1l1_grad_lapl(torch::Tensor s) {
    return -(20 * psi1l1_grad(s) - 4 * psi12l_grad(s) + 4 * psi111l_grad(s) - 20 * psi0_grad(s));
}

torch::Tensor psi111_lapl(torch::Tensor s) {
    return -(18 * psi111_action(s) - 6 * psi12d_action(s) - 24 * psi0_action(s) + 6 * psi1l1_action(s));
}

torch::Tensor psi111_grad_lapl(torch::Tensor s) {
    return -(18 * psi111_grad(s) - 6 * psi12d_grad(s) - 24 * psi0_grad(s) + 6 * psi1l1_grad(s));
}

torch::Tensor psi111c_lapl(torch::Tensor s) {
    return -(16 * psi111c_action(s) - 4 * psi21_action(s) - 16 * psi0_action(s) - 4 * psi12l_action(s) + 8 * psi1l1_action(s));
}

torch::Tensor psi111c_grad_lapl(torch::Tensor s) {
    return -(16 * psi111c_grad(s) - 4 * psi21_grad(s) - 16 * psi0_grad(s) - 4 * psi12l_grad(s) + 8 * psi1l1_grad(s));
}

torch::Tensor psi111l_lapl(torch::Tensor s) {
    return -(24 * psi111l_action(s) - 12 * psi0_action(s));
}

torch::Tensor psi111l_grad_lapl(torch::Tensor s) {
    return -(24 * psi111l_grad(s) - 12 * psi0_grad(s));
}

// ---------- Utility: SflowO2 fused grad (native path) ----------
// Combines all Psi grads with SflowO2 coefficients to reduce Python overhead.
torch::Tensor sflowo2_grad(torch::Tensor s, double beta, double t) {
    const double coef1 = beta / 8.0;
    const double coef2 = (beta * beta) / 8.0;
    const double coef3 = (beta * beta * beta) / 8.0;
    auto s1 = neighbor_sum(s);
    auto s2 = neighbor_sum(s1);
    auto acc = coef1 * psi0_grad_fast(s, s1);
    if (t != 0.0) {
        const double t1 = t;
        acc = acc + t1 * ((1.0 / 3.0) * coef2 * psi2_grad_fast(s, s1) - (1.0 / 5.0) * coef2 * psi11_grad(s) -
                          (1.0 / 6.0) * coef2 * psi11l_grad(s));
        const double t2 = t1 * t1;
        acc = acc + t2 * ((-1489.0 / 3000.0) * coef3 * psi0_grad_fast(s, s1) + (29.0 / 200.0) * coef3 * psi3_grad(s) +
                          (-11.0 / 100.0) * coef3 * psi21_grad_fast(s, s1, s2) +
                          (-1.0 / 30.0) * coef3 * psi12d_grad_fast(s, s1) +
                          (2.0 / 90.0) * coef3 * psi111_grad(s) + (1.0 / 40.0) * coef3 * psi111c_grad(s) +
                          (41.0 / 1500.0) * coef3 * psi12l_grad(s) + (-7.0 / 300.0) * coef3 * psi1l1_grad(s) +
                          (7.0 / 1800.0) * coef3 * psi111l_grad(s));
    }
    return acc;
}

// ---------- Utility: SflowO2 fused action ----------
torch::Tensor sflowo2_action(torch::Tensor s, double beta, double t) {
    const double coef1 = beta / 8.0;
    const double coef2 = (beta * beta) / 8.0;
    const double coef3 = (beta * beta * beta) / 8.0;
    auto s1 = neighbor_sum(s);
    auto s2 = neighbor_sum(s1);
    auto acc = coef1 * psi0_action_fast(s, s1);
    if (t != 0.0) {
        const double t1 = t;
        acc = acc + t1 * ((1.0 / 3.0) * coef2 * psi2_action_fast(s, s1) - (1.0 / 5.0) * coef2 * psi11_action(s) -
                          (1.0 / 6.0) * coef2 * psi11l_action(s));
        const double t2 = t1 * t1;
        acc = acc + t2 * ((-1489.0 / 3000.0) * coef3 * psi0_action_fast(s, s1) + (29.0 / 200.0) * coef3 * psi3_action(s) +
                          (-11.0 / 100.0) * coef3 * psi21_action(s) + (-1.0 / 30.0) * coef3 * psi12d_action_fast(s, s1) +
                          (2.0 / 90.0) * coef3 * psi111_action(s) + (1.0 / 40.0) * coef3 * psi111c_action(s) +
                          (41.0 / 1500.0) * coef3 * psi12l_action(s) + (-7.0 / 300.0) * coef3 * psi1l1_action(s) +
                          (7.0 / 1800.0) * coef3 * psi111l_action(s));
    }
    return acc;
}

// ---------- Utility: SflowO2 fused laplacian and grad_laplacian ----------
torch::Tensor sflowo2_lapl(torch::Tensor s, double beta, double t) {
    const double coef1 = beta / 8.0;
    const double coef2 = (beta * beta) / 8.0;
    const double coef3 = (beta * beta * beta) / 8.0;
    auto s1 = neighbor_sum(s);
    auto s2 = neighbor_sum(s1);
    auto acc = coef1 * psi0_lapl_fast(s, s1);
    if (t != 0.0) {
        const double t1 = t;
        acc = acc + t1 * ((1.0 / 3.0) * coef2 * psi2_lapl_fast(s, s1) - (1.0 / 5.0) * coef2 * psi11_lapl(s) -
                          (1.0 / 6.0) * coef2 * psi11l_lapl(s));
        const double t2 = t1 * t1;
        acc = acc + t2 * ((-1489.0 / 3000.0) * coef3 * psi0_lapl_fast(s, s1) + (29.0 / 200.0) * coef3 * psi3_lapl(s) +
                          (-11.0 / 100.0) * coef3 * psi21_lapl(s) + (-1.0 / 30.0) * coef3 * psi12d_lapl(s) +
                          (2.0 / 90.0) * coef3 * psi111_lapl(s) + (1.0 / 40.0) * coef3 * psi111c_lapl(s) +
                          (41.0 / 1500.0) * coef3 * psi12l_lapl(s) + (-7.0 / 300.0) * coef3 * psi1l1_lapl(s) +
                          (7.0 / 1800.0) * coef3 * psi111l_lapl(s));
    }
    return -acc;
}

torch::Tensor sflowo2_grad_lapl(torch::Tensor s, double beta, double t) {
    const double coef1 = beta / 8.0;
    const double coef2 = (beta * beta) / 8.0;
    const double coef3 = (beta * beta * beta) / 8.0;
    auto s1 = neighbor_sum(s);
    auto s2 = neighbor_sum(s1);
    auto acc = coef1 * psi0_grad_lapl_fast(s, s1);
    if (t != 0.0) {
        const double t1 = t;
        acc = acc + t1 * ((1.0 / 3.0) * coef2 * psi2_grad_lapl_fast(s, s1) - (1.0 / 5.0) * coef2 * psi11_grad_lapl(s) -
                          (1.0 / 6.0) * coef2 * psi11l_grad_lapl(s));
        const double t2 = t1 * t1;
        acc = acc + t2 * ((-1489.0 / 3000.0) * coef3 * psi0_grad_lapl_fast(s, s1) + (29.0 / 200.0) * coef3 * psi3_grad_lapl(s) +
                          (-11.0 / 100.0) * coef3 * psi21_grad_lapl(s) + (-1.0 / 30.0) * coef3 * psi12d_grad_lapl(s) +
                          (2.0 / 90.0) * coef3 * psi111_grad_lapl(s) + (1.0 / 40.0) * coef3 * psi111c_grad_lapl(s) +
                          (41.0 / 1500.0) * coef3 * psi12l_grad_lapl(s) + (-7.0 / 300.0) * coef3 * psi1l1_grad_lapl(s) +
                          (7.0 / 1800.0) * coef3 * psi111l_grad_lapl(s));
    }
    return acc;
}

// ---------- Utility: SflowO1 fused ops ----------
torch::Tensor sflowo1_action(torch::Tensor s, double beta, double t) {
    const double c0 = beta / 8.0;
    const double coef = (beta * beta) / 8.0;
    auto acc = c0 * psi0_action(s);
    if (t != 0.0) {
        const double t1 = t;
        acc = acc + t1 * ((1.0 / 3.0) * coef * psi2_action(s) - (1.0 / 5.0) * coef * psi11_action(s) -
                          (1.0 / 6.0) * coef * psi11l_action(s));
    }
    return acc;
}

torch::Tensor sflowo1_grad(torch::Tensor s, double beta, double t) {
    const double c0 = beta / 8.0;
    const double coef = (beta * beta) / 8.0;
    auto acc = c0 * psi0_grad(s);
    if (t != 0.0) {
        const double t1 = t;
        acc = acc + t1 * ((1.0 / 3.0) * coef * psi2_grad(s) - (1.0 / 5.0) * coef * psi11_grad(s) -
                          (1.0 / 6.0) * coef * psi11l_grad(s));
    }
    return acc;
}

// Note: returns minus-lapl (mlapl)
torch::Tensor sflowo1_lapl(torch::Tensor s, double beta, double t) {
    const double c0 = beta / 8.0;
    const double coef = (beta * beta) / 8.0;
    auto acc = c0 * psi0_lapl(s);
    if (t != 0.0) {
        const double t1 = t;
        acc = acc + t1 * ((1.0 / 3.0) * coef * psi2_lapl(s) - (1.0 / 5.0) * coef * psi11_lapl(s) -
                          (1.0 / 6.0) * coef * psi11l_lapl(s));
    }
    return -acc;  // return minus-laplacian (mlapl)
}

torch::Tensor sflowo1_grad_lapl(torch::Tensor s, double beta, double t) {
    const double c0 = beta / 8.0;
    const double coef = (beta * beta) / 8.0;
    auto acc = c0 * psi0_grad_lapl(s);
    if (t != 0.0) {
        const double t1 = t;
        acc = acc + t1 * ((1.0 / 3.0) * coef * psi2_grad_lapl(s) - (1.0 / 5.0) * coef * psi11_grad_lapl(s) -
                          (1.0 / 6.0) * coef * psi11l_grad_lapl(s));
    }
    return acc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("psi0_action", &psi0_action, "Psi0 action");
    m.def("psi0_grad", &psi0_grad, "Psi0 grad");
    m.def("psi0_lapl", &psi0_lapl, "Psi0 laplacian");
    m.def("psi0_grad_lapl", &psi0_grad_lapl, "Psi0 grad laplacian");
    m.def("psi2_action", &psi2_action, "Psi2 action");
    m.def("psi2_grad", &psi2_grad, "Psi2 grad");
    m.def("psi2_lapl", &psi2_lapl, "Psi2 laplacian");
    m.def("psi2_grad_lapl", &psi2_grad_lapl, "Psi2 grad laplacian");
    m.def("psi11_action", &psi11_action, "Psi11 action");
    m.def("psi11_grad", &psi11_grad, "Psi11 grad");
    m.def("psi11_lapl", &psi11_lapl, "Psi11 laplacian");
    m.def("psi11_grad_lapl", &psi11_grad_lapl, "Psi11 grad laplacian");
    m.def("psi11l_action", &psi11l_action, "Psi11_l action");
    m.def("psi11l_grad", &psi11l_grad, "Psi11_l grad");
    m.def("psi11l_lapl", &psi11l_lapl, "Psi11_l laplacian");
    m.def("psi11l_grad_lapl", &psi11l_grad_lapl, "Psi11_l grad laplacian");
    m.def("psi3_action", &psi3_action, "Psi3 action");
    m.def("psi3_grad", &psi3_grad, "Psi3 grad");
    m.def("psi3_lapl", &psi3_lapl, "Psi3 laplacian");
    m.def("psi3_grad_lapl", &psi3_grad_lapl, "Psi3 grad laplacian");
    m.def("psi21_action", &psi21_action, "Psi21 action");
    m.def("psi21_grad", &psi21_grad, "Psi21 grad");
    m.def("psi21_lapl", &psi21_lapl, "Psi21 laplacian");
    m.def("psi21_grad_lapl", &psi21_grad_lapl, "Psi21 grad laplacian");
    m.def("psi12d_action", &psi12d_action, "Psi12d action");
    m.def("psi12d_grad", &psi12d_grad, "Psi12d grad");
    m.def("psi12d_lapl", &psi12d_lapl, "Psi12d laplacian");
    m.def("psi12d_grad_lapl", &psi12d_grad_lapl, "Psi12d grad laplacian");
    m.def("psi12l_action", &psi12l_action, "Psi12l action");
    m.def("psi12l_grad", &psi12l_grad, "Psi12l grad");
    m.def("psi12l_lapl", &psi12l_lapl, "Psi12l laplacian");
    m.def("psi12l_grad_lapl", &psi12l_grad_lapl, "Psi12l grad laplacian");
    m.def("psi111_action", &psi111_action, "Psi111 action");
    m.def("psi111_grad", &psi111_grad, "Psi111 grad");
    m.def("psi111_lapl", &psi111_lapl, "Psi111 laplacian");
    m.def("psi111_grad_lapl", &psi111_grad_lapl, "Psi111 grad laplacian");
    m.def("psi111c_action", &psi111c_action, "Psi111c action");
    m.def("psi111c_grad", &psi111c_grad, "Psi111c grad");
    m.def("psi111c_lapl", &psi111c_lapl, "Psi111c laplacian");
    m.def("psi111c_grad_lapl", &psi111c_grad_lapl, "Psi111c grad laplacian");
    m.def("psi1l1_action", &psi1l1_action, "Psi1l1 action");
    m.def("psi1l1_grad", &psi1l1_grad, "Psi1l1 grad");
    m.def("psi1l1_lapl", &psi1l1_lapl, "Psi1l1 laplacian");
    m.def("psi1l1_grad_lapl", &psi1l1_grad_lapl, "Psi1l1 grad laplacian");
    m.def("psi111l_action", &psi111l_action, "Psi111l action");
    m.def("psi111l_grad", &psi111l_grad, "Psi111l grad");
    m.def("psi111l_lapl", &psi111l_lapl, "Psi111l laplacian");
    m.def("psi111l_grad_lapl", &psi111l_grad_lapl, "Psi111l grad laplacian");
    m.def("sflowo1_action", &sflowo1_action, "Fused SflowO1 action");
    m.def("sflowo1_grad", &sflowo1_grad, "Fused SflowO1 grad");
    m.def("sflowo1_lapl", &sflowo1_lapl, "Fused SflowO1 minus laplacian");
    m.def("sflowo1_grad_lapl", &sflowo1_grad_lapl, "Fused SflowO1 grad laplacian");
    m.def("sflowo2_action", &sflowo2_action, "Fused SflowO2 action");
    m.def("sflowo2_grad", &sflowo2_grad, "Fused SflowO2 grad");
    m.def("sflowo2_lapl", &sflowo2_lapl, "Fused SflowO2 laplacian");
    m.def("sflowo2_grad_lapl", &sflowo2_grad_lapl, "Fused SflowO2 grad laplacian");
}
