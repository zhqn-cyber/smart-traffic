# -*- coding: utf-8 -*-
"""
Intelligent Traffic Control Simulator — "AI Fundamentals" пәнінің қорытынды жобасы

Орнату және іске қосу:
    pip install streamlit pandas numpy plotly
    streamlit run app.py

Силлабусты қамту:
  * 3-апта (Local Search)   — Hill Climbing қайта бастаулармен жасыл фазаның ұзақтығын таңдайды.
  * 5-апта (RL)             — кестелік Q-learning фазаны "ұстап тұру / ауыстыру" шешімін қабылдайды.
  * 6-7-апта (HMM)          — трафиктің жасырын күйі (бос жол / қарбалас сағат / кептеліс)
                              жылдамдық пен тығыздықтың шулы датчиктері бойынша бағаланады
                              (алға алгоритмі + Viterbi). Баға Q-агент күйінің бөлігі ретінде қолданылады;
                              HMM бойынша орташа жүктеме Hill Climbing жанында көрсетіледі.
"""
import json
import math
import random
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ----------------------------------------------------------------------------
# Модель тұрақтылары
# ----------------------------------------------------------------------------
STATE_NAMES = ["Бос жол", "Қарбалас сағат", "Кептеліс"]
STATE_COLORS = ["#2ca02c", "#ff9f1c", "#d62728"]
LOAD_MULT = np.array([0.6, 1.0, 1.5])          # жасырын күй бойынша қарқындылық көбейткіші
SPEED_MU, SPEED_SD = np.array([42.0, 29.0, 15.0]), np.array([10.0, 9.0, 7.0])      # км/сағ
DENS_MU, DENS_SD = np.array([20.0, 38.0, 62.0]), np.array([10.0, 12.0, 18.0])      # көлік/км
P_OFF = np.array([[0.0, 0.85, 0.15], [0.4, 0.0, 0.6], [0.2, 0.8, 0.0]])          # режим ауысқанда қайда өтеміз

SAT_FLOW = 0.5      # жасыл кезінде бір жолаққа көлік/с (1800 көлік/сағ)
YELLOW = 3.0        # ауыстыру кезіндегі жоғалған уақыт, с
MIN_GREEN = 10.0    # жасылдың ең аз ұзақтығы, с

N_FIX, N_HC, N_Q = "Қолмен (тұрақты цикл)", "Hill Climbing (3-апта)", "Q-learning (5-апта)"
NAMES = [N_FIX, N_HC, N_Q]
SHORT = {N_FIX: "Қолмен", N_HC: "Hill Climbing", N_Q: "Q-learning"}
COLORS = {N_FIX: "#7f7f7f", N_HC: "#1f77b4", N_Q: "#e4572e"}

# Q-learning үшін дискреттеу (бір жолаққа кезек ұзындығы)
EDGES = (1, 3, 6, 10, 16, 25)
NB = len(EDGES) + 1
BUCKET_LABELS = ["<1", "1–3", "3–6", "6–10", "10–16", "16–25", "≥25"]
N_STATES = NB * NB * 2 * 3 * 2
Q_MID = (0.5, 2.0, 4.5, 8.0, 13.0, 20.5, 30.0)   # әр себеттегі бір жолаққа типтік кезек


def init_q(gamma_eff):
    """Ақпараттық инициализация: кезек ұзын болған сайын күтілетін сыйақы нашар.
    (Нөлдік "оптимистік" инициализация «әрқашан ауыстыру» деген азғындалған саясатқа әкелді.)"""
    Q = []
    for s in range(N_STATES):
        rest = (s >> 1) // 3 // 2
        v = -(Q_MID[rest // NB] + Q_MID[rest % NB]) / (1 - gamma_eff)
        Q.append([v, v])
    return Q


@dataclass(frozen=True)
class Config:
    base_rate: float     # көлік/мин бір жолаққа (екі бағыт бойынша орташа)
    ratio: float         # NS : EW жүктеме қатынасы
    lanes: int
    dt: int              # симуляция қадамы, с
    duration_min: int
    dwell_min: float     # трафик режимінің орташа ұзақтығы, мин
    noise: float         # датчиктер шуының көбейткіші
    seed: int

    @property
    def n_steps(self):
        return int(self.duration_min * 60 / self.dt)

    @property
    def rates(self):
        r = self.ratio
        return self.base_rate * 2 * r / (1 + r), self.base_rate * 2 / (1 + r)


# ----------------------------------------------------------------------------
# 6-7-апта: жасырын марковтық модель (HMM)
# ----------------------------------------------------------------------------
def transition_matrix(cfg):
    stay = math.exp(-cfg.dt / (cfg.dwell_min * 60.0))
    return stay * np.eye(3) + (1 - stay) * P_OFF


def stationary(A):
    p = np.ones(3) / 3
    for _ in range(500):
        p = p @ A
    return p / p.sum()


def log_gauss(x, mu, sd):
    return -0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd)


def forward_filter(LL, A, pi):
    """Сүзу: P(s_t | o_1..t) — онлайн қолданылады, тек өткен деректер."""
    L = np.exp(LL - LL.max(axis=1, keepdims=True))
    B = np.empty_like(L)
    b = pi * L[0]
    b /= b.sum()
    B[0] = b
    for t in range(1, len(L)):
        b = (b @ A) * L[t]
        b /= b.sum()
        B[t] = b
    return B


def viterbi(LL, A, pi):
    """Күйлердің ең ықтимал тізбегі (офлайн)."""
    n = len(LL)
    logA = np.log(A + 1e-300)
    delta = np.log(pi + 1e-300) + LL[0]
    psi = np.zeros((n, 3), dtype=int)
    for t in range(1, n):
        cand = delta[:, None] + logA
        psi[t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + LL[t]
    path = np.zeros(n, dtype=int)
    path[-1] = int(delta.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path


@dataclass
class Scenario:
    states: np.ndarray
    speed: np.ndarray
    dens: np.ndarray
    arrivals: np.ndarray
    LL: np.ndarray
    belief: np.ndarray
    est: np.ndarray


def generate_scenario(cfg, rng, hmm=True):
    n = cfg.n_steps
    A = transition_matrix(cfg)
    pi = stationary(A)
    cum = np.cumsum(A, axis=1).tolist()
    u = rng.random(n).tolist()
    s = min(2, int(np.searchsorted(np.cumsum(pi), rng.random())))
    st_list = [0] * n
    for t in range(n):
        st_list[t] = s
        c, x = cum[s], u[t]
        s = 0 if x < c[0] else (1 if x < c[1] else 2)
    states = np.array(st_list)

    sd_s, sd_d = SPEED_SD * cfg.noise, DENS_SD * cfg.noise
    speed = np.maximum(0.0, rng.normal(SPEED_MU[states], sd_s[states]))
    dens = np.maximum(0.0, rng.normal(DENS_MU[states], sd_d[states]))
    lam = LOAD_MULT[states][:, None] * np.array(cfg.rates)[None, :] * cfg.lanes * cfg.dt / 60.0
    arrivals = rng.poisson(lam)

    LL = log_gauss(speed, SPEED_MU, sd_s) + log_gauss(dens, DENS_MU, sd_d)
    if hmm:
        belief = forward_filter(LL, A, pi)
        est = belief.argmax(axis=1)
    else:
        belief, est = np.tile(pi, (n, 1)), np.zeros(n, dtype=int)
    return Scenario(states, speed, dens, arrivals, LL, belief, est)


# ----------------------------------------------------------------------------
# Қиылыс моделі
# ----------------------------------------------------------------------------
class Intersection:
    """Екі фаза: 0 — NS үшін жасыл, 1 — EW үшін жасыл. Кезектер — көліктердің бүтін саны."""

    def __init__(self, lanes, dt):
        self.lanes, self.dt = lanes, dt
        self.cap_rate = lanes * SAT_FLOW
        self.reset()

    def reset(self):
        self.q = [0, 0]
        self.phase = 0
        self.green_elapsed = 0.0
        self.lost = 0.0
        self.carry = 0.0
        self.wait_acc = 0.0   # жиналған көлік-секунд күту
        self.served = 0
        self.arrived = 0

    def step(self, a0, a1, switch):
        dt, q = self.dt, self.q
        if switch and self.lost <= 0 and self.green_elapsed >= MIN_GREEN:
            self.phase ^= 1
            self.green_elapsed = 0.0
            self.lost = YELLOW
            self.carry = 0.0
        q[0] += a0
        q[1] += a1
        self.arrived += a0 + a1
        avail = dt
        if self.lost > 0:
            used = min(self.lost, avail)
            self.lost -= used
            avail -= used
        if avail > 0:
            p = self.phase
            cap = self.cap_rate * avail + self.carry
            n = int(cap)
            if n >= q[p]:
                s, self.carry = q[p], 0.0
            else:
                s, self.carry = n, cap - n
            q[p] -= s
            self.served += s
            self.green_elapsed += avail
        self.wait_acc += (q[0] + q[1]) * dt


def run_policy(cfg, arrivals, decide):
    inter = Intersection(cfg.lanes, cfg.dt)
    rec = {k: [] for k in ("q0", "q1", "phase", "amb", "wait", "arr", "srv")}
    prev_phase = 0
    for t, (a0, a1) in enumerate(arrivals):
        inter.step(a0, a1, decide(inter, t))
        rec["amb"].append(1 if inter.phase != prev_phase else 0)
        prev_phase = inter.phase
        rec["q0"].append(inter.q[0]); rec["q1"].append(inter.q[1])
        rec["phase"].append(inter.phase); rec["wait"].append(inter.wait_acc)
        rec["arr"].append(inter.arrived); rec["srv"].append(inter.served)
    rec = {k: np.array(v) for k, v in rec.items()}
    hours = cfg.n_steps * cfg.dt / 3600.0
    qtot = rec["q0"] + rec["q1"]
    met = dict(avg_wait=inter.wait_acc / max(1, inter.arrived), throughput=inter.served / hours,
               mean_q=float(qtot.mean()), max_q=float(qtot.max()), left=float(qtot[-1]))
    rec["qtot"] = qtot
    return met, rec


def fixed_decider(greens):
    def decide(inter, t):
        return inter.lost <= 0 and inter.green_elapsed >= greens[inter.phase]
    return decide


def eval_fixed(cfg, arr, greens):
    inter = Intersection(cfg.lanes, cfg.dt)
    for a0, a1 in arr:
        inter.step(a0, a1, inter.lost <= 0 and inter.green_elapsed >= greens[inter.phase])
    return inter.wait_acc / max(1, inter.arrived)


# ----------------------------------------------------------------------------
# 3-апта: Hill Climbing (жергілікті іздеу)
# ----------------------------------------------------------------------------
def hill_climb(cfg, load_mult, fixed_green, restarts, seed, step=5, gmin=10, gmax=80, n_train=3):
    """f(g_ns, g_ew) орташа күтуін минимумдайды. 8 көршіге қарай тік көтерілу + кездейсоқ қайта бастаулар.
    Мақсат функциясы режимдері ауысатын бірнеше "тарихи" ағын бойынша орташаланады,
    сондықтан жоспар тек орташа жүктемеге емес, кептеліске де төзімді болады."""
    rng = np.random.default_rng(seed)
    train = [generate_scenario(cfg, rng, hmm=False).arrivals.tolist() for _ in range(n_train)]
    cache = {}

    def f(g):
        if g not in cache:
            cache[g] = float(np.mean([eval_fixed(cfg, a, g) for a in train]))
        return cache[g]

    g0 = int(min(gmax, max(gmin, round(fixed_green / step) * step)))
    starts = [(g0, g0)]
    for _ in range(restarts - 1):
        starts.append(tuple(int(rng.integers(gmin // step, gmax // step + 1)) * step for _ in range(2)))

    moves = [(dx, dy) for dx in (-step, 0, step) for dy in (-step, 0, step) if (dx, dy) != (0, 0)]
    paths = []
    for cur in starts:
        fc = f(cur)
        path = [(cur, fc)]
        for _ in range(100):
            neigh = [(cur[0] + dx, cur[1] + dy) for dx, dy in moves
                     if gmin <= cur[0] + dx <= gmax and gmin <= cur[1] + dy <= gmax]
            best = min(neigh, key=f)
            if f(best) < fc - 1e-9:
                cur, fc = best, f(best)
                path.append((cur, fc))
            else:
                break
        paths.append(path)
    evals = len(cache)
    best_g, best_f = min((p[-1] for p in paths), key=lambda x: x[1])

    grid = list(range(gmin, gmax + 1, 10))
    Z = np.array([[f((gx, gy)) for gx in grid] for gy in grid])
    return dict(best=best_g, best_f=best_f, base_f=f((g0, g0)), base_g=g0, paths=paths,
                evals=evals, grid=grid, Z=Z, load_mult=load_mult)


# ----------------------------------------------------------------------------
# 5-апта: Q-learning
# ----------------------------------------------------------------------------
def encode(q0, q1, phase, level, can, lanes):
    b0, b1 = bisect_right(EDGES, q0 / lanes), bisect_right(EDGES, q1 / lanes)
    return (((b0 * NB + b1) * 2 + phase) * 3 + level) * 2 + can


def train_q(cfg, episodes, alpha, gamma, use_hmm, seed, progress=None):
    """Күй: (NS кезегі, EW кезегі, ағымдағы фаза, HMM трафик деңгейі, ауыстыруға бола ма).
    Әрекеттер: 0 — ұстап тұру, 1 — ауыстыру. Сыйақы: -(бір жолаққа жалпы кезек) әр қадамға."""
    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)
    visits = [0] * N_STATES
    n, lanes = cfg.n_steps, cfg.lanes
    g = gamma ** (cfg.dt / 5.0)            # дисконтты симуляция қадамына бейімдейміз
    Q = init_q(g)
    inter = Intersection(lanes, cfg.dt)
    rew_log, wait_log = [], []
    for ep in range(episodes):
        scen = generate_scenario(cfg, rng, hmm=use_hmm)
        arr = scen.arrivals.tolist()
        est = scen.est.tolist() if use_hmm else [0] * n
        eps = max(0.03, 0.03 ** (ep / (0.8 * episodes)))
        inter.reset()
        s = encode(0, 0, 0, est[0], 0, lanes)
        tot = 0.0
        for t in range(n):
            qs = Q[s]
            if not (s & 1):
                a = 0
            elif rnd.random() < eps:
                a = rnd.getrandbits(1)
            else:
                a = 1 if qs[1] > qs[0] else 0
            inter.step(arr[t][0], arr[t][1], a)
            q = inter.q
            r = -(q[0] + q[1]) / lanes
            can = 1 if (inter.lost <= 0 and inter.green_elapsed >= MIN_GREEN) else 0
            s2 = encode(q[0], q[1], inter.phase, est[t + 1] if t + 1 < n else est[t], can, lanes)
            q2 = Q[s2]
            qs[a] += alpha * (r + g * (q2[0] if q2[0] > q2[1] else q2[1]) - qs[a])
            visits[s] += 1
            s = s2
            tot += r
        rew_log.append(tot / n)
        wait_log.append(inter.wait_acc / max(1, inter.arrived))
        if progress and ep % 10 == 0:
            progress((ep + 1) / episodes)
    return dict(Q=Q, visits=visits, rew=rew_log, wait=wait_log)


def q_decider(Q, est, lanes, use_hmm):
    def decide(inter, t):
        if inter.lost > 0 or inter.green_elapsed < MIN_GREEN:
            return False
        s = encode(inter.q[0], inter.q[1], inter.phase, int(est[t]) if use_hmm else 0, 1, lanes)
        return Q[s][1] > Q[s][0]
    return decide


# ----------------------------------------------------------------------------
# Эксперименттің толық орындалуы
# ----------------------------------------------------------------------------
def run_all(cfg, hp, progress=lambda f, text="": None):
    progress(0.02, "HMM: сценарий жасау және датчиктерді сүзу…")
    scen0 = generate_scenario(cfg, np.random.default_rng(cfg.seed + 1000))
    load_est = float((scen0.belief @ LOAD_MULT).mean())

    progress(0.05, "Hill Climbing: фаза ұзақтығын жергілікті іздеу…")
    hc = hill_climb(cfg, load_est, hp["fixed_green"], hp["restarts"], cfg.seed + 1)

    progress(0.15, "Q-learning: агентті оқыту…")
    qres = train_q(cfg, hp["episodes"], hp["alpha"], hp["gamma"], hp["use_hmm"], cfg.seed + 2,
                   progress=lambda f: progress(0.15 + 0.7 * f, f"Q-learning: агентті оқыту… {int(f * 100)}%"))

    progress(0.88, "Алгоритмдерді сынақ сценарийлерінде бағалау…")
    metrics = {n: [] for n in NAMES}
    hmm_acc = []
    recs = {}
    scen_list = [scen0] + [generate_scenario(cfg, np.random.default_rng(cfg.seed + 1000 + r))
                           for r in range(1, hp["replicas"])]
    A = transition_matrix(cfg)
    pi = stationary(A)
    for r, sc in enumerate(scen_list):
        arr = sc.arrivals.tolist()
        deciders = {
            N_FIX: fixed_decider((hp["fixed_green"],) * 2),
            N_HC: fixed_decider(hc["best"]),
            N_Q: q_decider(qres["Q"], sc.est, cfg.lanes, hp["use_hmm"]),
        }
        for name in NAMES:
            met, rec = run_policy(cfg, arr, deciders[name])
            metrics[name].append(met)
            if r == 0:
                recs[name] = rec
        vit = viterbi(sc.LL, A, pi)
        hmm_acc.append(dict(naive=(sc.LL.argmax(1) == sc.states).mean(),
                            filt=(sc.est == sc.states).mean(), viterbi=(vit == sc.states).mean()))
        if r == 0:
            vit0 = vit
    progress(1.0, "Дайын")
    return dict(cfg=cfg, hp=hp, scen=scen0, vit=vit0, load_est=load_est, hc=hc, q=qres,
                metrics={n: pd.DataFrame(v) for n, v in metrics.items()}, recs=recs,
                hmm_acc=pd.DataFrame(hmm_acc), A=A, pi=pi)


# ----------------------------------------------------------------------------
# Визуализация
# ----------------------------------------------------------------------------
def show(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.12))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def segments(states):
    out, start = [], 0
    for t in range(1, len(states) + 1):
        if t == len(states) or states[t] != states[start]:
            out.append((start, t, int(states[start])))
            start = t
    return out


def tab_compare(res):
    cfg, hp, M, recs, sc = res["cfg"], res["hp"], res["metrics"], res["recs"], res["scen"]
    base_w, base_t = M[N_FIX].avg_wait.mean(), M[N_FIX].throughput.mean()
    st.subheader("Тиімділік көрсеткіштері")
    st.caption(f"{hp['replicas']} тәуелсіз трафик сценарийі бойынша орташа (барлық алгоритмдерге бірдей).")
    cols = st.columns(3)
    for c, n in zip(cols, NAMES):
        w, th = M[n].avg_wait.mean(), M[n].throughput.mean()
        c.markdown(f"**{n}**")
        c.metric("Орташа күту уақыты, с/көлік", f"{w:.1f}",
                 None if n == N_FIX else f"{(w - base_w) / base_w * 100:+.1f}%", delta_color="inverse")
        c.metric("Өткізу қабілеті, көлік/сағ", f"{th:.0f}",
                 None if n == N_FIX else f"{(th - base_t) / base_t * 100:+.1f}%")
    rows = []
    for n in NAMES:
        m = M[n]
        rows.append({"Алгоритм": n,
                     "Орт. күту, с": f"{m.avg_wait.mean():.1f} ± {m.avg_wait.std(ddof=0):.1f}",
                     "Өткізу қабілеті, көлік/сағ": f"{m.throughput.mean():.0f} ± {m.throughput.std(ddof=0):.0f}",
                     "Орт. кезек, көлік": f"{m.mean_q.mean():.1f}",
                     "Макс. кезек, көлік": f"{m.max_q.mean():.0f}",
                     "Соңында қалғаны, көлік": f"{m.left.mean():.0f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    best = min(NAMES, key=lambda n: M[n].avg_wait.mean())
    gain = (base_w - M[best].avg_wait.mean()) / base_w * 100
    if best == N_FIX:
        st.warning("Бұл параметрлерде қолмен басқару ЖИ-алгоритмдерден нашар емес болып шықты (ағын төмен немесе симметриялы). "
                   "Қарқындылықты/жүктеме қатынасын немесе Q-learning эпизодтарының санын арттырыңыз.")
    else:
        st.success(f"Орташа күту бойынша ең жақсысы: **{best}** — қолмен басқарудан {gain:.1f}% аз.")

    t = (np.arange(cfg.n_steps) + 1) * cfg.dt / 60.0
    f1 = go.Figure()
    for a, b, s in segments(sc.states):
        f1.add_vrect(x0=a * cfg.dt / 60, x1=b * cfg.dt / 60, fillcolor=STATE_COLORS[s],
                     opacity=0.08, line_width=0, layer="below")
    for n in NAMES:
        f1.add_trace(go.Scatter(x=t, y=recs[n]["qtot"], name=n, line=dict(color=COLORS[n], width=2)))
    f1.update_layout(title="Жалпы кезек ұзындығы уақыт бойынша (фон: жасырын трафик режимі — "
                           "жасыл/қызғылт сары/қызыл)", xaxis_title="Уақыт, мин", yaxis_title="Кезектегі көлік")
    show(f1)

    f2 = go.Figure()
    for n in NAMES:
        w = np.where(recs[n]["arr"] > 0, recs[n]["wait"] / np.maximum(recs[n]["arr"], 1), np.nan)
        f2.add_trace(go.Scatter(x=t, y=w, name=n, line=dict(color=COLORS[n], width=2)))
    f2.update_layout(title="Бір көліктің жиналған орташа күту уақыты", xaxis_title="Уақыт, мин",
                     yaxis_title="Секунд")
    show(f2)

    f3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                       subplot_titles=("NS кезегі", "EW кезегі"))
    for n in NAMES:
        f3.add_trace(go.Scatter(x=t, y=recs[n]["q0"], name=n, line=dict(color=COLORS[n]), legendgroup=n), row=1, col=1)
        f3.add_trace(go.Scatter(x=t, y=recs[n]["q1"], name=n, line=dict(color=COLORS[n]), legendgroup=n,
                                showlegend=False), row=2, col=1)
    f3.update_layout(title="Бағыттар бойынша кезектер", height=480)
    f3.update_xaxes(title_text="Уақыт, мин", row=2, col=1)
    show(f3)

    hi = max(2, int(cfg.duration_min))
    lo_hi = st.slider("Бағдаршам фазаларын қарау терезесі, мин", 0, hi, (0, min(15, hi)))
    mask = (t >= lo_hi[0]) & (t <= lo_hi[1])
    f4 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=NAMES)
    for i, n in enumerate(NAMES, 1):
        f4.add_trace(go.Scatter(x=t[mask], y=recs[n]["phase"][mask], line_shape="hv", showlegend=False,
                                line=dict(color=COLORS[n])), row=i, col=1)
        f4.update_yaxes(tickvals=[0, 1], ticktext=["NS жасыл", "EW жасыл"], row=i, col=1)
    f4.update_layout(title="Бағдаршам фазалары (бейімделгіш алгоритмдер циклді жүктемеге қарай өзгертеді)", height=520)
    show(f4)

    fb = make_subplots(rows=1, cols=3, subplot_titles=("Орт. күту, с (аз болса — жақсы)",
                                                        "Өткізу қабілеті, көлік/сағ", "Орташа кезек, көлік"))
    for i, key in enumerate(("avg_wait", "throughput", "mean_q"), 1):
        for n in NAMES:
            fb.add_trace(go.Bar(x=[SHORT[n]], y=[M[n][key].mean()], name=n, marker_color=COLORS[n],
                                error_y=dict(type="data", array=[M[n][key].std(ddof=0)]),
                                showlegend=(i == 1)), row=1, col=i)
    show(fb)


def tab_hc(res):
    hc, cfg = res["hc"], res["cfg"]
    st.subheader("3-апта — Hill Climbing (жергілікті іздеу)")
    st.markdown(
        "Іздеу кеңістігі — жасыл жану ұзақтығының жұбы **(g_NS, g_EW)**, қадамы 5 с, диапазоны 10–80 с. "
        "Мақсат функциясы — симуляциядағы көліктің орташа күту уақыты (кішірейтеміз). "
        "Көршілер — 8 жақын нүкте; алгоритм жақсару болғанша ең жақсы көршіге өтеді және жергілікті минимумда "
        "қалып қоймау үшін кездейсоқ нүктелерден қайта басталады. "
        "Мақсат функциясы режимдері ауысатын (қарбалас сағат/кептеліс) 3 тарихи ағын бойынша орташаланады, "
        f"орташа жүктеме бойынша емес (HMM бойынша орташа жүктеме бағасы: ×{hc['load_mult']:.2f}).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Бастау (қолмен цикл)", f"{hc['base_g']} / {hc['base_g']} с")
    c2.metric("Табылғаны (NS / EW)", f"{hc['best'][0]} / {hc['best'][1]} с")
    c3.metric("Мақсат: бастау → табылған", f"{hc['base_f']:.1f} → {hc['best_f']:.1f} с")
    c4.metric("Мақсат функциясын есептеу саны", hc["evals"])
    fig = go.Figure(go.Heatmap(x=hc["grid"], y=hc["grid"], z=hc["Z"], colorscale="Viridis_r",
                               colorbar=dict(title="Орт. күту, с")))
    for i, p in enumerate(hc["paths"]):
        fig.add_trace(go.Scatter(x=[g[0][0] for g in p], y=[g[0][1] for g in p], mode="lines+markers",
                                 name=f"Қайта бастау {i + 1}" + (" (қолмен циклден)" if i == 0 else ""),
                                 line=dict(width=3), marker=dict(size=8, line=dict(color="white", width=1))))
    fig.add_trace(go.Scatter(x=[hc["best"][0]], y=[hc["best"][1]], mode="markers", name="Ең жақсы шешім",
                             marker=dict(symbol="star", size=18, color="white", line=dict(color="black", width=2))))
    fig.update_layout(title="Мақсат функциясының ландшафты және көтерілу траекториялары",
                      xaxis_title="NS жасылы, с", yaxis_title="EW жасылы, с", height=520)
    show(fig)
    f2 = go.Figure()
    for i, p in enumerate(hc["paths"]):
        f2.add_trace(go.Scatter(x=list(range(len(p))), y=[g[1] for g in p], mode="lines+markers",
                                name=f"Қайта бастау {i + 1}"))
    f2.update_layout(title="Жинақталу: мақсат функциясының мәні итерациялар бойынша",
                     xaxis_title="Итерация", yaxis_title="Орт. күту, с")
    show(f2)


def policy_grid(Q, visits, phase, level):
    Z = np.full((NB, NB), np.nan)
    for b0 in range(NB):
        for b1 in range(NB):
            s = (((b0 * NB + b1) * 2 + phase) * 3 + level) * 2 + 1
            if visits[s] > 0:
                Z[b1, b0] = 1.0 if Q[s][1] > Q[s][0] else 0.0
    return Z


def tab_q(res):
    q, hp = res["q"], res["hp"]
    st.subheader("5-апта — Q-learning")
    st.markdown(
        "Агент әр *Δt* секунд сайын **ұстап тұру** немесе **ауыстыру** шешімін таңдайды. "
        "**Күй:** NS және EW кезегінің ұзындығы (бір жолаққа, 7 себет), ағымдағы фаза, "
        "HMM бойынша трафик режимінің бағасы (қосылған болса) және «ең аз жасыл өтті» белгісі. "
        "**Сыйақы:** жалпы кезектің теріс мәні. Жаңарту: "
        "`Q(s,a) ← Q(s,a) + α·[r + γ·maxₐ′ Q(s′,a′) − Q(s,a)]`, ε-ашкөз стратегия, ε біртіндеп кемиді.")
    visited = sum(1 for v in q["visits"] if v > 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Оқыту эпизодтарының саны", hp["episodes"])
    c2.metric("Көрілген күйлер", f"{visited} / {N_STATES}")
    c3.metric("Соңғы 10% эпизодтағы орт. күту",
              f"{np.mean(q['wait'][-max(1, len(q['wait']) // 10):]):.1f} с")
    w = max(1, len(q["wait"]) // 20)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Эпизодтағы орташа күту, с (эпсилон-зерттеумен)",
                                                        "Қадамға орташа сыйақы"))
    for i, key in enumerate(("wait", "rew"), 1):
        s = pd.Series(q[key])
        fig.add_trace(go.Scatter(y=s, mode="lines", line=dict(color="#cccccc"), name="эпизод", showlegend=(i == 1)),
                      row=1, col=i)
        fig.add_trace(go.Scatter(y=s.rolling(w, min_periods=1).mean(), line=dict(color=COLORS[N_Q], width=3),
                                 name="жылжымалы орташа", showlegend=(i == 1)), row=1, col=i)
    fig.update_xaxes(title_text="Эпизод")
    show(fig)
    st.markdown("**Үйренген саясат** («ең аз жасыл өтті» күйі үшін): "
                "қызыл — ауыстыру, жасыл — ұстап тұру, бос — күй кездеспеген.")
    cc1, cc2 = st.columns(2)
    phase = 0 if cc1.radio("Ағымдағы фаза", ["NS жасыл", "EW жасыл"], horizontal=True) == "NS зелёный" else 1
    level = 0
    if hp["use_hmm"]:
        level = STATE_NAMES.index(cc2.selectbox("Трафик режимі (HMM бағасы)", STATE_NAMES, index=1))
    Z = policy_grid(q["Q"], q["visits"], phase, level)
    hm = go.Figure(go.Heatmap(z=Z, x=BUCKET_LABELS, y=BUCKET_LABELS, zmin=0, zmax=1,
                              colorscale=[[0, "#2ca02c"], [1, "#d62728"]],
                              colorbar=dict(tickvals=[0, 1], ticktext=["ұстап тұру", "ауыстыру"])))
    hm.update_layout(xaxis_title="NS кезегі (көлік/жолақ)", yaxis_title="EW кезегі (көлік/жолақ)", height=450)
    show(hm)


def tab_hmm(res):
    cfg, sc, acc = res["cfg"], res["scen"], res["hmm_acc"]
    t = (np.arange(cfg.n_steps) + 1) * cfg.dt / 60.0
    st.subheader("6-7-апта — жасырын марковтық модель (HMM)")
    st.markdown(
        "Трафиктің нақты режимі (**бос жол / қарбалас сағат / кептеліс**) жасырын. Тек жылдамдық пен "
        "тығыздықтың шулы датчиктері көрінеді. HMM = ауысу матрицасы A + датчиктердің гаусстық модельдері. "
        "**Сүзгі (алға алгоритмі)** нақты уақытта P(күй | t дейінгі барлық бақылаулар) береді, "
        "**Viterbi** — кейінірек ең ықтимал күйлер тізбегін береді. Салыстыру үшін — тек ағымдағы "
        "көрсеткішке қарайтын аңғал классификатор.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Аңғал (уақыт моделінсіз)", f"{acc.naive.mean() * 100:.1f}%")
    c2.metric("HMM-сүзгі (онлайн)", f"{acc.filt.mean() * 100:.1f}%", f"{(acc.filt.mean() - acc.naive.mean()) * 100:+.1f} п.т.")
    c3.metric("Viterbi (офлайн)", f"{acc.viterbi.mean() * 100:.1f}%", f"{(acc.viterbi.mean() - acc.naive.mean()) * 100:+.1f} п.т.")
    st.caption(f"Режимді анықтау дәлдігі, {len(acc)} сценарий бойынша орташа. "
               f"Жүктеменің бағаланған орташа көбейткіші: ×{res['load_est']:.2f}.")

    f = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                      subplot_titles=("Ағын жылдамдығы, км/сағ", "Тығыздық, көлік/км"))
    f.add_trace(go.Scatter(x=t, y=sc.speed, name="датчик", line=dict(color="#9ecae1", width=1)), row=1, col=1)
    f.add_trace(go.Scatter(x=t, y=SPEED_MU[sc.states], name="нақты орташа", line=dict(color="black", dash="dash")), row=1, col=1)
    f.add_trace(go.Scatter(x=t, y=sc.dens, name="датчик", line=dict(color="#9ecae1", width=1), showlegend=False), row=2, col=1)
    f.add_trace(go.Scatter(x=t, y=DENS_MU[sc.states], name="нақты орташа", line=dict(color="black", dash="dash"), showlegend=False), row=2, col=1)
    f.update_layout(title="Датчиктердің шулы көрсеткіштері", height=460)
    f.update_xaxes(title_text="Уақыт, мин", row=2, col=1)
    show(f)

    g = go.Figure()
    g.add_trace(go.Scatter(x=t, y=sc.states, name="Шындық", line_shape="hv", line=dict(color="black", width=4)))
    g.add_trace(go.Scatter(x=t, y=sc.LL.argmax(1) + 0.0, name="Аңғал", line_shape="hv", mode="lines",
                           line=dict(color="#bbbbbb", width=1)))
    g.add_trace(go.Scatter(x=t, y=sc.est, name="HMM-сүзгі", line_shape="hv", line=dict(color="#e4572e", width=2)))
    g.add_trace(go.Scatter(x=t, y=res["vit"], name="Viterbi", line_shape="hv", line=dict(color="#1f77b4", width=2, dash="dot")))
    g.update_yaxes(tickvals=[0, 1, 2], ticktext=STATE_NAMES)
    g.update_layout(title="Жасырын күйдің нақты және бағаланған мәні", xaxis_title="Уақыт, мин", height=380)
    show(g)

    p = go.Figure()
    for k in range(3):
        p.add_trace(go.Scatter(x=t, y=sc.belief[:, k], name=STATE_NAMES[k], stackgroup="one",
                               line=dict(width=0.5, color=STATE_COLORS[k])))
    p.update_layout(title="Күйлердің апостериорлық ықтималдықтары (belief)", xaxis_title="Уақыт, мин",
                    yaxis=dict(range=[0, 1], title="P"))
    show(p)

    c1, c2 = st.columns(2)
    c1.markdown("**A ауысу матрицасы (бір қадамға)**")
    c1.dataframe(pd.DataFrame(res["A"], index=STATE_NAMES, columns=STATE_NAMES).round(4))
    c2.markdown("**Датчиктер моделі (μ ± σ, шумды ескере отырып)**")
    c2.dataframe(pd.DataFrame({"Жылдамдық, км/сағ": [f"{m:.0f} ± {s * cfg.noise:.1f}" for m, s in zip(SPEED_MU, SPEED_SD)],
                               "Тығыздық, көлік/км": [f"{m:.0f} ± {s * cfg.noise:.1f}" for m, s in zip(DENS_MU, DENS_SD)],
                               "Ағын көбейткіші": LOAD_MULT}, index=STATE_NAMES))


ROAD_HTML = r'''
<style>
*{box-sizing:border-box}
body{margin:0;font-family:Arial,Helvetica,sans-serif;background:#0f1626;color:#e8ebf2}
#bar{display:flex;gap:12px;align-items:center;padding:10px 14px;flex-wrap:wrap}
button,select{background:#24304d;color:#e8ebf2;border:1px solid #3a4a73;border-radius:8px;padding:7px 12px;font-size:14px;cursor:pointer}
#slider{flex:1;min-width:180px}
#clock{font-variant-numeric:tabular-nums;min-width:120px;font-size:14px}
#banner{padding:4px 14px 10px;display:flex;gap:28px;font-size:15px;flex-wrap:wrap}
#wrap{display:flex;gap:10px;padding:0 10px 6px;flex-wrap:wrap;justify-content:center}
#legend{padding:2px 14px 12px;font-size:13px;opacity:.8}
</style>
<div id="bar">
  <button id="play">&#9208; Тоқтату</button>
  <button id="restart">&#9198; Басынан</button>
  <label>Жылдамдық <select id="speed"><option value="2">1x</option><option value="5" selected>2.5x</option><option value="10">5x</option><option value="20">10x</option></select></label>
  <input id="slider" type="range" min="0" max="1" step="0.01" value="0">
  <span id="clock"></span>
</div>
<div id="banner"><span>Трафиктің нақты режимі (ЖИ-ден жасырын): <b id="tr"></b></span><span>Датчиктер бойынша HMM бағасы: <b id="es"></b></span></div>
<div id="wrap"></div>
<div id="legend">Үш қиылыста көліктер ағыны бірдей. Бағдаршам: жасыл / сары (ауыстыру кезіндегі жоғалған уақыт) / қызыл. Экрандағы әр көлік = 1 автомобиль, «+N» — экранға сыймаған көліктер.</div>
<script>
const D=__DATA__;
const W=400,H=400,C=200,L=D.lanes,RW=L*13+6;
const MAXROWS=Math.max(3,Math.floor((C-RW-8)/14));
const PAL=['#e63946','#f4a261','#2a9d8f','#e9c46a','#8ecae6','#90be6d','#c77dff'];
const LIGHT={green:'#2ecc71',red:'#e74c3c',amber:'#f1c40f'};
const wrap=document.getElementById('wrap');
const ctxs=[];
D.ctrls.forEach(()=>{const e=document.createElement('canvas');e.width=W;e.height=H;e.style.flex='1 1 300px';e.style.maxWidth='420px';e.style.height='auto';e.style.borderRadius='12px';wrap.appendChild(e);ctxs.push(e.getContext('2d'));});
const slider=document.getElementById('slider'),clock=document.getElementById('clock'),playBtn=document.getElementById('play'),
      restartBtn=document.getElementById('restart'),speedSel=document.getElementById('speed'),trEl=document.getElementById('tr'),esEl=document.getElementById('es');
slider.max=D.n-1;
let t=0,playing=true,speed=+speedSel.value,clk=0,last=performance.now(),drag=false;
const cars=D.ctrls.map(()=>[]);

function lightState(c,i,axis){
  const ph=c.ph[i],amb=c.amb[i];
  if(amb) return axis===ph?'red':'amber';
  return axis===ph?'green':'red';
}
function carShape(g,x,y,vert,col){
  const w=vert?10:13,h=vert?13:10;
  g.fillStyle=col;g.beginPath();
  if(g.roundRect){g.roundRect(x-w/2,y-h/2,w,h,3);}else{g.rect(x-w/2,y-h/2,w,h);}
  g.fill();
  g.fillStyle='rgba(255,255,255,0.35)';
  if(vert){g.fillRect(x-w/2+2,y-h/2+3,w-4,3);}else{g.fillRect(x-w/2+3,y-h/2+2,3,h-4);}
}
function armPos(arm,idx){
  const col=idx%L,row=Math.floor(idx/L);
  const off=3+(col+0.5)*13,back=RW+4+(row+0.5)*14;
  if(arm===0)return [C-off,C-back];
  if(arm===1)return [C+off,C+back];
  if(arm===2)return [C-back,C+off];
  return [C+back,C-off];
}
function drawQueue(g,arm,n){
  const shown=Math.min(n,MAXROWS*L);
  for(let idx=0;idx<shown;idx++){const p=armPos(arm,idx);carShape(g,p[0],p[1],arm<2,PAL[(idx*3+arm)%7]);}
  if(n>shown){
    const p=armPos(arm,MAXROWS*L);
    let x=p[0],y=p[1];
    if(arm===0)x=C-RW/2;if(arm===1)x=C+RW/2;if(arm===2)y=C+RW/2;if(arm===3)y=C-RW/2;
    g.fillStyle='#ffd166';g.font='bold 13px Arial';g.textAlign='center';g.fillText('+'+(n-shown),x,y+4);
  }
}
function drawLight(g,x,y,state){
  g.fillStyle='#0b0d12';g.beginPath();g.arc(x,y,10,0,6.2832);g.fill();
  g.shadowColor=LIGHT[state];g.shadowBlur=14;g.fillStyle=LIGHT[state];
  g.beginPath();g.arc(x,y,7,0,6.2832);g.fill();g.shadowBlur=0;
}
function drawMoving(g,k){
  for(const m of cars[k]){
    const p=(clk-m.born)/1100;
    if(p<0||p>1)continue;
    const col=m.col%L,off=3+(col+0.5)*13,d=p*(C+RW);
    let x,y,vert;
    if(m.axis===0){vert=true;if(m.sign>0){x=C-off;y=C-RW+d;}else{x=C+off;y=C+RW-d;}}
    else{vert=false;if(m.sign>0){x=C-RW+d;y=C+off;}else{x=C+RW-d;y=C-off;}}
    carShape(g,x,y,vert,PAL[(m.col*2+m.axis+(m.sign>0?0:3))%7]);
  }
}
function drawScene(k,i){
  const g=ctxs[k],c=D.ctrls[k];
  g.clearRect(0,0,W,H);
  g.fillStyle='#22364a';g.fillRect(0,0,W,H);
  g.fillStyle='#3a3f4b';g.fillRect(C-RW,0,2*RW,H);g.fillRect(0,C-RW,W,2*RW);
  g.fillStyle='#31353f';g.fillRect(C-RW,C-RW,2*RW,2*RW);
  g.strokeStyle='#d9d9d9';g.lineWidth=2;g.setLineDash([8,8]);
  g.beginPath();g.moveTo(C,0);g.lineTo(C,C-RW);g.moveTo(C,C+RW);g.lineTo(C,H);g.moveTo(0,C);g.lineTo(C-RW,C);g.moveTo(C+RW,C);g.lineTo(W,C);g.stroke();
  g.setLineDash([]);
  g.strokeStyle='#ffffff';g.lineWidth=3;
  g.beginPath();g.moveTo(C-RW,C-RW);g.lineTo(C,C-RW);g.moveTo(C,C+RW);g.lineTo(C+RW,C+RW);
  g.moveTo(C-RW,C);g.lineTo(C-RW,C+RW);g.moveTo(C+RW,C-RW);g.lineTo(C+RW,C);g.stroke();
  drawQueue(g,0,Math.ceil(c.q0[i]/2));drawQueue(g,1,Math.floor(c.q0[i]/2));
  drawQueue(g,2,Math.ceil(c.q1[i]/2));drawQueue(g,3,Math.floor(c.q1[i]/2));
  drawMoving(g,k);
  const sNS=lightState(c,i,0),sEW=lightState(c,i,1);
  drawLight(g,C-RW-12,C-RW-12,sNS);drawLight(g,C+RW+12,C+RW+12,sNS);
  drawLight(g,C-RW-12,C+RW+12,sEW);drawLight(g,C+RW+12,C-RW-12,sEW);
  g.textAlign='left';g.font='bold 14px Arial';g.fillStyle=c.color;g.fillText(c.name,10,20);
  g.font='13px Arial';g.fillStyle='#e8ebf2';
  g.fillText('Кезек: NS '+c.q0[i]+' · EW '+c.q1[i],10,H-26);
  const w=c.arr[i]>0?c.wait[i]/c.arr[i]:0;
  g.fillText('Орт. күту: '+w.toFixed(1)+' с · өткені: '+c.srv[i],10,H-8);
}
function fmt(sec){const m=Math.floor(sec/60),s=Math.floor(sec%60);return String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');}
function render(){
  const i=Math.max(0,Math.min(D.n-1,Math.floor(t)));
  D.ctrls.forEach((c,k)=>drawScene(k,i));
  if(!drag)slider.value=t;
  clock.textContent=fmt(i*D.dt)+' / '+fmt((D.n-1)*D.dt);
  const s=D.state[i],e=D.est[i];
  trEl.innerHTML='<span style="color:'+D.stateColors[s]+'">&#9679;</span> '+D.stateNames[s];
  esEl.innerHTML='<span style="color:'+D.stateColors[e]+'">&#9679;</span> '+D.stateNames[e];
}
function spawn(k,i){
  if(i<=0)return;
  const c=D.ctrls[k],served=c.srv[i]-c.srv[i-1];
  if(served<=0)return;
  const axis=c.ph[i],m=Math.min(served,3);
  for(let j=0;j<m;j++){cars[k].push({axis:axis,sign:1,col:j,born:clk+j*170});cars[k].push({axis:axis,sign:-1,col:j+1,born:clk+j*170});}
}
function clearCars(){cars.forEach(a=>a.length=0);}
function setBtn(){playBtn.innerHTML=playing?'&#9208; Тоқтату':'&#9654; Бастау';}
playBtn.onclick=()=>{if(!playing&&t>=D.n-1){t=0;clearCars();}playing=!playing;setBtn();};
restartBtn.onclick=()=>{t=0;clearCars();playing=true;setBtn();};
speedSel.onchange=()=>{speed=+speedSel.value;};
slider.oninput=()=>{t=+slider.value;clearCars();drag=true;render();};
slider.onchange=()=>{drag=false;};
function frame(now){
  const dtm=Math.min(0.1,(now-last)/1000);last=now;
  if(playing){
    clk+=dtm*1000;
    const prev=Math.floor(t);
    t=Math.min(D.n-1,t+speed*dtm);
    const cur=Math.floor(t);
    for(let s=prev+1;s<=cur;s++){if(cur-s<3)D.ctrls.forEach((c,k)=>spawn(k,s));}
    if(t>=D.n-1){playing=false;setBtn();}
    for(let k=0;k<cars.length;k++)cars[k]=cars[k].filter(m=>clk-m.born<1200);
  }
  render();
  requestAnimationFrame(frame);
}
render();setBtn();requestAnimationFrame(frame);
</script>
'''


def build_road_html(res):
    """Қиылыстың HTML/JS-анимациясы: деректер есептелген симуляциядан алынады."""
    cfg, sc = res["cfg"], res["scen"]
    ctrls = []
    for n in NAMES:
        r = res["recs"][n]
        ctrls.append(dict(name=SHORT[n], color=COLORS[n] if n != N_FIX else "#cfcfcf",
                          q0=r["q0"].tolist(), q1=r["q1"].tolist(), ph=r["phase"].tolist(),
                          amb=r["amb"].tolist(), srv=r["srv"].tolist(), arr=r["arr"].tolist(),
                          wait=[round(float(x), 1) for x in r["wait"]]))
    data = dict(dt=cfg.dt, lanes=cfg.lanes, n=cfg.n_steps, state=sc.states.tolist(), est=sc.est.tolist(),
                stateNames=STATE_NAMES, stateColors=STATE_COLORS, ctrls=ctrls)
    return ROAD_HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))


def tab_road(res):
    import streamlit.components.v1 as components
    st.subheader("Қиылыстың визуализациясы")
    st.markdown(
        "Көліктер ағыны **бірдей**, бірақ басқаруы әртүрлі үш қиылыс: сол жақта — қарапайым таймер, "
        "ортада — Hill Climbing, оң жақта — Q-learning. **«Бастау»** батырмасын басыңыз, жылдамдықты өзгертіңіз "
        "немесе уақыт жүгірткісін сүйреңіз. Кезек ұзындығына және ақылды бағдаршамдардың кептелісте қалай "
        "ауысатынына қараңыз.")
    components.html(build_road_html(res), height=700, scrolling=True)


THEORY = """
**Симуляция қалай құрылған.** Екі фазалы қиылыс (NS / EW). Көліктердің келуі пуассондық, қарқындылығы
трафиктің жасырын режиміне байланысты. Жасыл жанғанда әр жолаққа секундына 0.5 көлікке дейін қызмет көрсетіледі;
ауыстыру 3 секунд жоғалған уақытқа түседі, жасылдың ең аз ұзақтығы — 10 секунд. Орташа күту Литтл заңы бойынша
есептеледі (кезекте жиналған көлік-секундтар / келген көліктер саны).

**Үш әдіс бір-бірімен қалай байланысқан.**
1. *HMM* шулы датчиктерді трафик режимінің ықтималдық бағасына айналдырады.
2. *Hill Climbing* режимдердің ауысуына төзімді статикалық бағдаршам жоспарын таңдайды (HMM бойынша орташа жүктеме бақылау үшін көрсетіледі).
3. *Q-learning* режим бағасын күй құрамына қосады және нақты уақытта кезекке бейімделуді үйренеді.

**Прототиптің шектеулері.** Бір қиылыс, екі фаза, бұрылыстар мен жаяу жүргіншілер жоқ; датчиктер берілген
күйде шартты түрде тәуелсіз; Q-learning кестелік (кезектер дискреттеледі); Hill Climbing тұрақты жоспарды
оңтайландырады және модельденген тарихи ағындарда калибрленеді.
"""


def main():
    st.set_page_config(page_title="Intelligent Traffic Control Simulator", page_icon="🚦", layout="wide")
    st.title("🚦 Ақылды бағдаршам симуляторы")
    st.caption("AI Fundamentals · Local Search (3-апта) + Q-learning (5-апта) + HMM (6-7-апта)")

    sb = st.sidebar
    sb.header("⚙️ Параметрлер")
    sb.subheader("Трафик")
    base_rate = sb.slider("Кіріс ағынының қарқындылығы, көлік/мин (бір жолаққа)", 2.0, 15.0, 8.0, 0.5)
    lanes = sb.slider("Бағыттағы жолақ саны", 1, 4, 2)
    ratio = sb.slider("Жүктеме қатынасы NS : EW", 1.0, 3.0, 1.5, 0.1)
    dwell = sb.slider("Трафик режимінің орташа ұзақтығы, мин", 2, 15, 5)
    noise = sb.slider("Датчиктер шуы (σ көбейткіші)", 0.5, 3.0, 1.0, 0.1)
    sb.subheader("Симуляция")
    dt = sb.slider("Симуляция қадамы, с", 2, 10, 5)
    duration = sb.slider("Ұзақтығы, мин", 10, 120, 60, 5)
    replicas = sb.slider("Сынақ сценарийлерінің саны (статистика үшін)", 1, 20, 8)
    seed = sb.number_input("Random seed", 0, 10_000, 42)
    sb.subheader("Қолмен басқару")
    fixed_green = sb.slider("Тұрақты циклдегі жасыл, с", 10, 80, 30, 5)
    sb.subheader("Hill Climbing")
    restarts = sb.slider("Іске қосу саны (қайта бастаумен)", 1, 6, 3)
    sb.subheader("Q-learning")
    episodes = sb.slider("Оқыту эпизодтарының саны", 100, 1500, 400, 50)
    alpha = sb.slider("Оқыту жылдамдығы α", 0.02, 0.5, 0.15, 0.01)
    gamma = sb.slider("Дисконт γ (5 с үшін)", 0.80, 0.99, 0.95, 0.01)
    use_hmm = sb.checkbox("Агент күйінде HMM бағасын пайдалану", value=True)
    run = sb.button("▶️ Симуляцияны іске қосу", type="primary")

    if run:
        cfg = Config(base_rate, ratio, lanes, dt, duration, float(dwell), noise, int(seed))
        hp = dict(fixed_green=fixed_green, restarts=restarts, episodes=episodes, alpha=alpha,
                  gamma=gamma, use_hmm=use_hmm, replicas=replicas)
        bar = st.progress(0.0, text="Іске қосылуда…")
        res = run_all(cfg, hp, lambda f, text="": bar.progress(min(1.0, f), text=text or "Оқыту…"))
        bar.empty()
        st.session_state["res"] = res

    res = st.session_state.get("res")
    if res is None:
        st.info("Бүйір панельде параметрлерді таңдап, **«Симуляцияны іске қосу»** батырмасын басыңыз. "
                "Әдепкі параметрлермен бірінші іске қосу бірнеше секунд алады (Q-агентті оқыту).")
        st.markdown(THEORY)
        return

    c = res["cfg"]
    st.caption(f"Соңғы іске қосу нәтижелері: {c.base_rate} көлік/мин/жолақ · {c.lanes} жолақ · қатынас {c.ratio} · "
               f"қадам {c.dt} с · {c.duration_min} мин · шум ×{c.noise}")
    tabs = st.tabs(["🚗 Жол (көрнекі)", "📊 Салыстыру", "🧗 Hill Climbing (3-апта)",
                    "🤖 Q-learning (5-апта)", "🌫️ HMM (6-7-апта)", "📖 Түсініктемелер"])
    with tabs[0]:
        tab_road(res)
    with tabs[1]:
        tab_compare(res)
    with tabs[2]:
        tab_hc(res)
    with tabs[3]:
        tab_q(res)
    with tabs[4]:
        tab_hmm(res)
    with tabs[5]:
        st.markdown(THEORY)


if __name__ == "__main__":
    main()
