# -*- coding: utf-8 -*-
"""
Intelligent Traffic Control Simulator — финальный проект "AI Fundamentals"

Установка и запуск:
    pip install streamlit pandas numpy plotly
    streamlit run app.py

Покрытие силлабуса:
  * Неделя 3 (Local Search)   — Hill Climbing с перезапусками подбирает длительности зелёной фазы.
  * Неделя 5 (RL)             — табличный Q-learning решает "держать / переключить" фазу.
  * Неделя 6-7 (HMM)          — скрытое состояние трафика (свободно / час пик / пробка)
                                оценивается по зашумлённым датчикам скорости и плотности
                                (алгоритм вперёд + Viterbi). Оценка используется
                                как часть состояния Q-агента; средняя нагрузка по HMM показывается рядом с Hill Climbing.
"""
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
# Константы модели
# ----------------------------------------------------------------------------
STATE_NAMES = ["Свободно", "Час пик", "Пробка"]
STATE_COLORS = ["#2ca02c", "#ff9f1c", "#d62728"]
LOAD_MULT = np.array([0.6, 1.0, 1.5])          # множитель интенсивности по скрытому состоянию
SPEED_MU, SPEED_SD = np.array([42.0, 29.0, 15.0]), np.array([10.0, 9.0, 7.0])      # км/ч
DENS_MU, DENS_SD = np.array([20.0, 38.0, 62.0]), np.array([10.0, 12.0, 18.0])      # авт/км
P_OFF = np.array([[0.0, 0.85, 0.15], [0.4, 0.0, 0.6], [0.2, 0.8, 0.0]])          # куда уходим при смене режима

SAT_FLOW = 0.5      # авт/с на полосу при зелёном (1800 авт/ч)
YELLOW = 3.0        # потерянное время при переключении, с
MIN_GREEN = 10.0    # минимальная длительность зелёного, с

N_FIX, N_HC, N_Q = "Ручное (фикс. цикл)", "Hill Climbing (нед. 3)", "Q-learning (нед. 5)"
NAMES = [N_FIX, N_HC, N_Q]
SHORT = {N_FIX: "Ручное", N_HC: "Hill Climbing", N_Q: "Q-learning"}
COLORS = {N_FIX: "#7f7f7f", N_HC: "#1f77b4", N_Q: "#e4572e"}

# Дискретизация для Q-learning (длина очереди на полосу)
EDGES = (1, 3, 6, 10, 16, 25)
NB = len(EDGES) + 1
BUCKET_LABELS = ["<1", "1–3", "3–6", "6–10", "10–16", "16–25", "≥25"]
N_STATES = NB * NB * 2 * 3 * 2
Q_MID = (0.5, 2.0, 4.5, 8.0, 13.0, 20.5, 30.0)   # типичная очередь на полосу в каждой корзине


def init_q(gamma_eff):
    """Информативная инициализация: чем длиннее очереди, тем хуже ожидаемая награда.
    (Нулевая "оптимистичная" инициализация приводила к вырожденной политике «переключать всегда».)"""
    Q = []
    for s in range(N_STATES):
        rest = (s >> 1) // 3 // 2
        v = -(Q_MID[rest // NB] + Q_MID[rest % NB]) / (1 - gamma_eff)
        Q.append([v, v])
    return Q


@dataclass(frozen=True)
class Config:
    base_rate: float     # авт/мин на полосу (средняя по двум направлениям)
    ratio: float         # перекос нагрузки NS : EW
    lanes: int
    dt: int              # шаг симуляции, с
    duration_min: int
    dwell_min: float     # средняя длительность режима трафика, мин
    noise: float         # множитель шума датчиков
    seed: int

    @property
    def n_steps(self):
        return int(self.duration_min * 60 / self.dt)

    @property
    def rates(self):
        r = self.ratio
        return self.base_rate * 2 * r / (1 + r), self.base_rate * 2 / (1 + r)


# ----------------------------------------------------------------------------
# Неделя 6-7: скрытая марковская модель (HMM)
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
    """Фильтрация: P(s_t | o_1..t) — используется онлайн, только прошлое."""
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
    """Наиболее вероятная последовательность состояний (офлайн)."""
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
# Модель перекрёстка
# ----------------------------------------------------------------------------
class Intersection:
    """Две фазы: 0 — зелёный для NS, 1 — зелёный для EW. Очереди — целые числа машин."""

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
        self.wait_acc = 0.0   # накопленные машино-секунды ожидания
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
    rec = {k: [] for k in ("q0", "q1", "phase", "wait", "arr", "srv")}
    for t, (a0, a1) in enumerate(arrivals):
        inter.step(a0, a1, decide(inter, t))
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
# Неделя 3: Hill Climbing (локальный поиск)
# ----------------------------------------------------------------------------
def hill_climb(cfg, load_mult, fixed_green, restarts, seed, step=5, gmin=10, gmax=80, n_train=3):
    """Минимизирует среднее ожидание f(g_ns, g_ew). Крутой подъём по 8 соседям + случайные рестарты.
    Целевая функция усредняется по нескольким "историческим" потокам с переключением режимов трафика,
    поэтому план должен быть устойчив и к пробкам, а не только к средней нагрузке."""
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
# Неделя 5: Q-learning
# ----------------------------------------------------------------------------
def encode(q0, q1, phase, level, can, lanes):
    b0, b1 = bisect_right(EDGES, q0 / lanes), bisect_right(EDGES, q1 / lanes)
    return (((b0 * NB + b1) * 2 + phase) * 3 + level) * 2 + can


def train_q(cfg, episodes, alpha, gamma, use_hmm, seed, progress=None):
    """Состояние: (очередь NS, очередь EW, текущая фаза, HMM-уровень трафика, можно ли переключать).
    Действия: 0 — держать, 1 — переключить. Награда: -(суммарная очередь на полосу) за шаг."""
    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)
    visits = [0] * N_STATES
    n, lanes = cfg.n_steps, cfg.lanes
    g = gamma ** (cfg.dt / 5.0)            # масштабируем дисконт под шаг симуляции
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
# Полный прогон эксперимента
# ----------------------------------------------------------------------------
def run_all(cfg, hp, progress=lambda f, text="": None):
    progress(0.02, "HMM: генерация сценария и фильтрация датчиков…")
    scen0 = generate_scenario(cfg, np.random.default_rng(cfg.seed + 1000))
    load_est = float((scen0.belief @ LOAD_MULT).mean())

    progress(0.05, "Hill Climbing: локальный поиск длительностей фаз…")
    hc = hill_climb(cfg, load_est, hp["fixed_green"], hp["restarts"], cfg.seed + 1)

    progress(0.15, "Q-learning: обучение агента…")
    qres = train_q(cfg, hp["episodes"], hp["alpha"], hp["gamma"], hp["use_hmm"], cfg.seed + 2,
                   progress=lambda f: progress(0.15 + 0.7 * f, f"Q-learning: обучение агента… {int(f * 100)}%"))

    progress(0.88, "Оценка алгоритмов на тестовых сценариях…")
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
    progress(1.0, "Готово")
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
    st.subheader("Метрики эффективности")
    st.caption(f"Среднее по {hp['replicas']} независимым сценариям трафика (одинаковым для всех алгоритмов).")
    cols = st.columns(3)
    for c, n in zip(cols, NAMES):
        w, th = M[n].avg_wait.mean(), M[n].throughput.mean()
        c.markdown(f"**{n}**")
        c.metric("Среднее время ожидания, с/авт", f"{w:.1f}",
                 None if n == N_FIX else f"{(w - base_w) / base_w * 100:+.1f}%", delta_color="inverse")
        c.metric("Пропускная способность, авт/ч", f"{th:.0f}",
                 None if n == N_FIX else f"{(th - base_t) / base_t * 100:+.1f}%")
    rows = []
    for n in NAMES:
        m = M[n]
        rows.append({"Алгоритм": n,
                     "Ср. ожидание, с": f"{m.avg_wait.mean():.1f} ± {m.avg_wait.std(ddof=0):.1f}",
                     "Пропускная способность, авт/ч": f"{m.throughput.mean():.0f} ± {m.throughput.std(ddof=0):.0f}",
                     "Ср. очередь, авт": f"{m.mean_q.mean():.1f}",
                     "Макс. очередь, авт": f"{m.max_q.mean():.0f}",
                     "Остаток в конце, авт": f"{m.left.mean():.0f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    best = min(NAMES, key=lambda n: M[n].avg_wait.mean())
    gain = (base_w - M[best].avg_wait.mean()) / base_w * 100
    if best == N_FIX:
        st.warning("При этих параметрах ручной цикл оказался не хуже ИИ-алгоритмов (поток низкий или симметричный). "
                   "Увеличьте интенсивность/перекос нагрузки или число эпизодов Q-learning.")
    else:
        st.success(f"Лучший по среднему ожиданию: **{best}** — на {gain:.1f}% меньше, чем у ручного управления.")

    t = (np.arange(cfg.n_steps) + 1) * cfg.dt / 60.0
    f1 = go.Figure()
    for a, b, s in segments(sc.states):
        f1.add_vrect(x0=a * cfg.dt / 60, x1=b * cfg.dt / 60, fillcolor=STATE_COLORS[s],
                     opacity=0.08, line_width=0, layer="below")
    for n in NAMES:
        f1.add_trace(go.Scatter(x=t, y=recs[n]["qtot"], name=n, line=dict(color=COLORS[n], width=2)))
    f1.update_layout(title="Суммарная длина очереди во времени (фон: скрытый режим трафика — "
                           "зелёный/оранжевый/красный)", xaxis_title="Время, мин", yaxis_title="Машин в очереди")
    show(f1)

    f2 = go.Figure()
    for n in NAMES:
        w = np.where(recs[n]["arr"] > 0, recs[n]["wait"] / np.maximum(recs[n]["arr"], 1), np.nan)
        f2.add_trace(go.Scatter(x=t, y=w, name=n, line=dict(color=COLORS[n], width=2)))
    f2.update_layout(title="Накопленное среднее время ожидания одной машины", xaxis_title="Время, мин",
                     yaxis_title="Секунд")
    show(f2)

    f3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                       subplot_titles=("Очередь NS", "Очередь EW"))
    for n in NAMES:
        f3.add_trace(go.Scatter(x=t, y=recs[n]["q0"], name=n, line=dict(color=COLORS[n]), legendgroup=n), row=1, col=1)
        f3.add_trace(go.Scatter(x=t, y=recs[n]["q1"], name=n, line=dict(color=COLORS[n]), legendgroup=n,
                                showlegend=False), row=2, col=1)
    f3.update_layout(title="Очереди по направлениям", height=480)
    f3.update_xaxes(title_text="Время, мин", row=2, col=1)
    show(f3)

    hi = max(2, int(cfg.duration_min))
    lo_hi = st.slider("Окно просмотра фаз светофора, мин", 0, hi, (0, min(15, hi)))
    mask = (t >= lo_hi[0]) & (t <= lo_hi[1])
    f4 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=NAMES)
    for i, n in enumerate(NAMES, 1):
        f4.add_trace(go.Scatter(x=t[mask], y=recs[n]["phase"][mask], line_shape="hv", showlegend=False,
                                line=dict(color=COLORS[n])), row=i, col=1)
        f4.update_yaxes(tickvals=[0, 1], ticktext=["NS зелёный", "EW зелёный"], row=i, col=1)
    f4.update_layout(title="Фазы светофора (адаптивные алгоритмы меняют цикл под нагрузку)", height=520)
    show(f4)

    fb = make_subplots(rows=1, cols=3, subplot_titles=("Ср. ожидание, с (меньше — лучше)",
                                                        "Пропускная способность, авт/ч", "Средняя очередь, авт"))
    for i, key in enumerate(("avg_wait", "throughput", "mean_q"), 1):
        for n in NAMES:
            fb.add_trace(go.Bar(x=[SHORT[n]], y=[M[n][key].mean()], name=n, marker_color=COLORS[n],
                                error_y=dict(type="data", array=[M[n][key].std(ddof=0)]),
                                showlegend=(i == 1)), row=1, col=i)
    show(fb)


def tab_hc(res):
    hc, cfg = res["hc"], res["cfg"]
    st.subheader("Неделя 3 — Hill Climbing (локальный поиск)")
    st.markdown(
        "Пространство поиска — пара длительностей зелёного **(g_NS, g_EW)**, шаг 5 с, диапазон 10–80 с. "
        "Целевая функция — среднее время ожидания машины в симуляции (минимизируем). "
        "Соседи — 8 ближайших точек; алгоритм идёт в лучшего соседа, пока есть улучшение, и "
        "перезапускается из случайных точек, чтобы не застрять в локальном минимуме. "
        "Целевая функция усредняется по 3 историческим потокам с переключением режимов (час пик/пробка), "
        f"а не по средней нагрузке (оценка средней нагрузки по HMM: ×{hc['load_mult']:.2f}).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Старт (ручной цикл)", f"{hc['base_g']} / {hc['base_g']} с")
    c2.metric("Найдено (NS / EW)", f"{hc['best'][0]} / {hc['best'][1]} с")
    c3.metric("Цель: старт → найдено", f"{hc['base_f']:.1f} → {hc['best_f']:.1f} с")
    c4.metric("Вычислений целевой функции", hc["evals"])
    fig = go.Figure(go.Heatmap(x=hc["grid"], y=hc["grid"], z=hc["Z"], colorscale="Viridis_r",
                               colorbar=dict(title="Ср. ожидание, с")))
    for i, p in enumerate(hc["paths"]):
        fig.add_trace(go.Scatter(x=[g[0][0] for g in p], y=[g[0][1] for g in p], mode="lines+markers",
                                 name=f"Рестарт {i + 1}" + (" (из ручного цикла)" if i == 0 else ""),
                                 line=dict(width=3), marker=dict(size=8, line=dict(color="white", width=1))))
    fig.add_trace(go.Scatter(x=[hc["best"][0]], y=[hc["best"][1]], mode="markers", name="Лучшее решение",
                             marker=dict(symbol="star", size=18, color="white", line=dict(color="black", width=2))))
    fig.update_layout(title="Ландшафт целевой функции и траектории подъёма",
                      xaxis_title="Зелёный NS, с", yaxis_title="Зелёный EW, с", height=520)
    show(fig)
    f2 = go.Figure()
    for i, p in enumerate(hc["paths"]):
        f2.add_trace(go.Scatter(x=list(range(len(p))), y=[g[1] for g in p], mode="lines+markers",
                                name=f"Рестарт {i + 1}"))
    f2.update_layout(title="Сходимость: значение целевой функции по итерациям",
                     xaxis_title="Итерация", yaxis_title="Ср. ожидание, с")
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
    st.subheader("Неделя 5 — Q-learning")
    st.markdown(
        "Агент каждые *Δt* секунд выбирает **держать** или **переключить** фазу. "
        "**Состояние:** длина очереди NS и EW (на полосу, 7 корзин), текущая фаза, "
        "оценка режима трафика из HMM (если включено) и флаг «минимальный зелёный прошёл». "
        "**Награда:** минус суммарная очередь. Обновление: "
        "`Q(s,a) ← Q(s,a) + α·[r + γ·maxₐ′ Q(s′,a′) − Q(s,a)]`, ε-жадная стратегия с затуханием ε.")
    visited = sum(1 for v in q["visits"] if v > 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Эпизодов обучения", hp["episodes"])
    c2.metric("Посещено состояний", f"{visited} / {N_STATES}")
    c3.metric("Ср. ожидание в последних 10% эпизодов",
              f"{np.mean(q['wait'][-max(1, len(q['wait']) // 10):]):.1f} с")
    w = max(1, len(q["wait"]) // 20)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Среднее ожидание в эпизоде, с (с эпсилон-исследованием)",
                                                        "Средняя награда за шаг"))
    for i, key in enumerate(("wait", "rew"), 1):
        s = pd.Series(q[key])
        fig.add_trace(go.Scatter(y=s, mode="lines", line=dict(color="#cccccc"), name="эпизод", showlegend=(i == 1)),
                      row=1, col=i)
        fig.add_trace(go.Scatter(y=s.rolling(w, min_periods=1).mean(), line=dict(color=COLORS[N_Q], width=3),
                                 name="скользящее среднее", showlegend=(i == 1)), row=1, col=i)
    fig.update_xaxes(title_text="Эпизод")
    show(fig)
    st.markdown("**Выученная политика** (для состояния «минимальный зелёный прошёл»): "
                "красное — переключить, зелёное — держать, пусто — состояние не встречалось.")
    cc1, cc2 = st.columns(2)
    phase = 0 if cc1.radio("Текущая фаза", ["NS зелёный", "EW зелёный"], horizontal=True) == "NS зелёный" else 1
    level = 0
    if hp["use_hmm"]:
        level = STATE_NAMES.index(cc2.selectbox("Режим трафика (оценка HMM)", STATE_NAMES, index=1))
    Z = policy_grid(q["Q"], q["visits"], phase, level)
    hm = go.Figure(go.Heatmap(z=Z, x=BUCKET_LABELS, y=BUCKET_LABELS, zmin=0, zmax=1,
                              colorscale=[[0, "#2ca02c"], [1, "#d62728"]],
                              colorbar=dict(tickvals=[0, 1], ticktext=["держать", "переключить"])))
    hm.update_layout(xaxis_title="Очередь NS (авт/полосу)", yaxis_title="Очередь EW (авт/полосу)", height=450)
    show(hm)


def tab_hmm(res):
    cfg, sc, acc = res["cfg"], res["scen"], res["hmm_acc"]
    t = (np.arange(cfg.n_steps) + 1) * cfg.dt / 60.0
    st.subheader("Неделя 6-7 — скрытая марковская модель (HMM)")
    st.markdown(
        "Реальный режим трафика (**свободно / час пик / пробка**) скрыт. Видны только шумные датчики скорости и "
        "плотности. HMM = матрица переходов A + гауссовы модели излучения датчиков. "
        "**Фильтр (алгоритм вперёд)** даёт P(состояние | все наблюдения до t) в реальном времени, "
        "**Viterbi** — наиболее вероятную цепочку задним числом. Для сравнения — наивный классификатор, "
        "который смотрит только на текущее показание.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Наивно (без модели времени)", f"{acc.naive.mean() * 100:.1f}%")
    c2.metric("HMM-фильтр (онлайн)", f"{acc.filt.mean() * 100:.1f}%", f"{(acc.filt.mean() - acc.naive.mean()) * 100:+.1f} п.п.")
    c3.metric("Viterbi (офлайн)", f"{acc.viterbi.mean() * 100:.1f}%", f"{(acc.viterbi.mean() - acc.naive.mean()) * 100:+.1f} п.п.")
    st.caption(f"Точность определения режима, среднее по {len(acc)} сценариям. "
               f"Оценённый средний множитель нагрузки: ×{res['load_est']:.2f}.")

    f = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                      subplot_titles=("Скорость потока, км/ч", "Плотность, авт/км"))
    f.add_trace(go.Scatter(x=t, y=sc.speed, name="датчик", line=dict(color="#9ecae1", width=1)), row=1, col=1)
    f.add_trace(go.Scatter(x=t, y=SPEED_MU[sc.states], name="истинное среднее", line=dict(color="black", dash="dash")), row=1, col=1)
    f.add_trace(go.Scatter(x=t, y=sc.dens, name="датчик", line=dict(color="#9ecae1", width=1), showlegend=False), row=2, col=1)
    f.add_trace(go.Scatter(x=t, y=DENS_MU[sc.states], name="истинное среднее", line=dict(color="black", dash="dash"), showlegend=False), row=2, col=1)
    f.update_layout(title="Зашумлённые показания датчиков", height=460)
    f.update_xaxes(title_text="Время, мин", row=2, col=1)
    show(f)

    g = go.Figure()
    g.add_trace(go.Scatter(x=t, y=sc.states, name="Истина", line_shape="hv", line=dict(color="black", width=4)))
    g.add_trace(go.Scatter(x=t, y=sc.LL.argmax(1) + 0.0, name="Наивно", line_shape="hv", mode="lines",
                           line=dict(color="#bbbbbb", width=1)))
    g.add_trace(go.Scatter(x=t, y=sc.est, name="HMM-фильтр", line_shape="hv", line=dict(color="#e4572e", width=2)))
    g.add_trace(go.Scatter(x=t, y=res["vit"], name="Viterbi", line_shape="hv", line=dict(color="#1f77b4", width=2, dash="dot")))
    g.update_yaxes(tickvals=[0, 1, 2], ticktext=STATE_NAMES)
    g.update_layout(title="Истинное и оценённое скрытое состояние", xaxis_title="Время, мин", height=380)
    show(g)

    p = go.Figure()
    for k in range(3):
        p.add_trace(go.Scatter(x=t, y=sc.belief[:, k], name=STATE_NAMES[k], stackgroup="one",
                               line=dict(width=0.5, color=STATE_COLORS[k])))
    p.update_layout(title="Апостериорные вероятности состояний (belief)", xaxis_title="Время, мин",
                    yaxis=dict(range=[0, 1], title="P"))
    show(p)

    c1, c2 = st.columns(2)
    c1.markdown("**Матрица переходов A (за один шаг)**")
    c1.dataframe(pd.DataFrame(res["A"], index=STATE_NAMES, columns=STATE_NAMES).round(4))
    c2.markdown("**Модель датчиков (μ ± σ с учётом шума)**")
    c2.dataframe(pd.DataFrame({"Скорость, км/ч": [f"{m:.0f} ± {s * cfg.noise:.1f}" for m, s in zip(SPEED_MU, SPEED_SD)],
                               "Плотность, авт/км": [f"{m:.0f} ± {s * cfg.noise:.1f}" for m, s in zip(DENS_MU, DENS_SD)],
                               "Множитель потока": LOAD_MULT}, index=STATE_NAMES))


THEORY = """
**Как устроена симуляция.** Перекрёсток с двумя фазами (NS / EW). Прибытие машин — пуассоновское, интенсивность
зависит от скрытого режима трафика. При зелёном обслуживается до 0.5 авт/с на полосу; переключение стоит 3 с
потерянного времени, минимальный зелёный — 10 с. Среднее ожидание считается по закону Литтла
(накопленные машино-секунды в очереди / число прибывших).

**Как три метода связаны между собой.**
1. *HMM* превращает шумные датчики в вероятностную оценку режима трафика.
2. *Hill Climbing* подбирает статический план светофора, устойчивый к смене режимов (оценка средней нагрузки по HMM показывается для контроля).
3. *Q-learning* использует оценку режима как часть состояния и учится реагировать на очереди в реальном времени.

**Ограничения прототипа.** Один перекрёсток, две фазы, без поворотов и пешеходов; датчики условно независимы
при данном состоянии; Q-learning — табличный (дискретизация очередей); Hill Climbing оптимизирует фиксированный план
и калибруется на смоделированных исторических потоках.
"""


def main():
    st.set_page_config(page_title="Intelligent Traffic Control Simulator", page_icon="🚦", layout="wide")
    st.title("🚦 Интеллектуальный симулятор умного светофора")
    st.caption("AI Fundamentals · Local Search (нед. 3) + Q-learning (нед. 5) + HMM (нед. 6-7)")

    sb = st.sidebar
    sb.header("⚙️ Параметры")
    sb.subheader("Трафик")
    base_rate = sb.slider("Интенсивность входящего потока, авт/мин на полосу", 2.0, 15.0, 8.0, 0.5)
    lanes = sb.slider("Количество полос на направление", 1, 4, 2)
    ratio = sb.slider("Перекос нагрузки NS : EW", 1.0, 3.0, 1.5, 0.1)
    dwell = sb.slider("Средняя длительность режима трафика, мин", 2, 15, 5)
    noise = sb.slider("Шум датчиков (множитель σ)", 0.5, 3.0, 1.0, 0.1)
    sb.subheader("Симуляция")
    dt = sb.slider("Шаг симуляции, с", 2, 10, 5)
    duration = sb.slider("Длительность, мин", 10, 120, 60, 5)
    replicas = sb.slider("Число тестовых сценариев (для статистики)", 1, 20, 8)
    seed = sb.number_input("Random seed", 0, 10_000, 42)
    sb.subheader("Ручное управление")
    fixed_green = sb.slider("Зелёный в фиксированном цикле, с", 10, 80, 30, 5)
    sb.subheader("Hill Climbing")
    restarts = sb.slider("Число запусков (с рестартами)", 1, 6, 3)
    sb.subheader("Q-learning")
    episodes = sb.slider("Эпизодов обучения", 100, 1500, 400, 50)
    alpha = sb.slider("Скорость обучения α", 0.02, 0.5, 0.15, 0.01)
    gamma = sb.slider("Дисконт γ (на 5 с)", 0.80, 0.99, 0.95, 0.01)
    use_hmm = sb.checkbox("Использовать оценку HMM в состоянии агента", value=True)
    run = sb.button("▶️ Запустить симуляцию", type="primary")

    if run:
        cfg = Config(base_rate, ratio, lanes, dt, duration, float(dwell), noise, int(seed))
        hp = dict(fixed_green=fixed_green, restarts=restarts, episodes=episodes, alpha=alpha,
                  gamma=gamma, use_hmm=use_hmm, replicas=replicas)
        bar = st.progress(0.0, text="Запуск…")
        res = run_all(cfg, hp, lambda f, text="": bar.progress(min(1.0, f), text=text or "Обучение…"))
        bar.empty()
        st.session_state["res"] = res

    res = st.session_state.get("res")
    if res is None:
        st.info("Задайте параметры в боковой панели и нажмите **«Запустить симуляцию»**. "
                "Первый запуск с параметрами по умолчанию занимает несколько секунд (обучение Q-агента).")
        st.markdown(THEORY)
        return

    c = res["cfg"]
    st.caption(f"Результаты последнего запуска: {c.base_rate} авт/мин/полосу · {c.lanes} полос · перекос {c.ratio} · "
               f"шаг {c.dt} с · {c.duration_min} мин · шум ×{c.noise}")
    tabs = st.tabs(["📊 Сравнение", "🧗 Hill Climbing (нед. 3)", "🤖 Q-learning (нед. 5)",
                    "🌫️ HMM (нед. 6-7)", "📖 Пояснения"])
    with tabs[0]:
        tab_compare(res)
    with tabs[1]:
        tab_hc(res)
    with tabs[2]:
        tab_q(res)
    with tabs[3]:
        tab_hmm(res)
    with tabs[4]:
        st.markdown(THEORY)


if __name__ == "__main__":
    main()
