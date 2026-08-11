"""Metrica SE(2) multi-horizonte, estratificacao por movimento e veredito GO/NO-GO
por bin. CONSOME WindowRollout (pose_sim/pose_real ja alinhadas) — NAO recalcula
dinamica.

Por janela w e horizonte h (amostras):
  e_pos(w,h) = ||(x_sim[h]-x_real[h], y_sim[h]-y_real[h])||_2          [m]
  e_ang(w,h) = |wrap_to_pi(theta_sim[h]-theta_real[h])|                [rad]
  e_se2(w,h) = sqrt(e_pos^2 + (rho*e_ang)^2)   (rho=robot_radius; OPCIONAL)

Independencia: CURVAS usam TODAS as janelas (densas, stride 15); INFERENCIA usa
SO janelas NAO sobrepostas (stride>=H_max) p/ Wilcoxon pareado, MAIS um
moving-block bootstrap (bloco ~H_max) sobre o denso p/ o CI do gap. Os dois sao
reportados; divergencia de sinal/ordem e sinalizada.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata, wilcoxon

from .params import load_twin_params
from .plant import wrap_to_pi
from .io import repo_root

BIN_ORDER = ("trans", "misto", "rot")
BIN_LABEL = {"trans": "trans-dominado", "misto": "misto", "rot": "rot-dominado"}


# =========================================================================
def load_metrics_config(path=None) -> dict:
    p = path or (repo_root() / "config" / "metrics.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_bin(razao: float, bins_cfg: dict) -> str:
    if razao < bins_cfg["trans_max"]:
        return "trans"
    if razao > bins_cfg["rot_min"]:
        return "rot"
    return "misto"


# =========================================================================
@dataclass
class ErrorTable:
    horizons: np.ndarray         # (nh,) amostras
    ts: float
    robot_radius: float
    seg_id: np.ndarray           # (W,)
    seg_name: list               # (W,)
    t0: np.ndarray               # (W,)
    bin: np.ndarray              # (W,) str
    razao: np.ndarray            # (W,)
    indep: np.ndarray            # (W,) bool (nao-sobreposta)
    epos_twin: np.ndarray        # (W, nh)
    eang_twin: np.ndarray        # (W, nh)
    epos_def: np.ndarray         # (W, nh)
    eang_def: np.ndarray         # (W, nh)
    ese2_twin: np.ndarray        # (W, nh) ou None
    ese2_def: np.ndarray         # (W, nh) ou None
    block_windows: int           # tamanho de bloco do MBB (em janelas)

    @property
    def W(self) -> int:
        return self.t0.size

    def mask(self, scope: str) -> np.ndarray:
        return np.ones(self.W, bool) if scope == "all" else (self.bin == scope)


def _window_errors(win, horizons):
    ps, pr = win.pose_sim, win.pose_real            # (H+1,3)
    dx = ps[horizons, 0] - pr[horizons, 0]
    dy = ps[horizons, 1] - pr[horizons, 1]
    epos = np.hypot(dx, dy)
    dth = ps[horizons, 2] - pr[horizons, 2]
    eang = np.abs(np.arctan2(np.sin(dth), np.cos(dth)))   # wrap_to_pi vetorizado
    return epos, eang


def _window_activity(win, robot_radius, ts):
    tr = np.hypot(win.cmd[:, 0], win.cmd[:, 1])      # comando translacional
    rot = np.abs(win.cmd[:, 2]) * robot_radius       # rad/s -> m/s no raio
    ativ_trans = float(tr.sum() * ts)
    ativ_rot = float(rot.sum() * ts)
    denom = ativ_rot + ativ_trans
    razao = ativ_rot / denom if denom > 1e-12 else 0.0
    return ativ_trans, ativ_rot, razao


def build_error_table(twin_rollouts, default_rollouts, *, params=None, mcfg=None) -> ErrorTable:
    """Tabela de erros pareada (twin vs default) sobre os MESMOS t0 (mesmas janelas).
    twin_rollouts/default_rollouts: listas de WindowRollout (1 por segmento), pareadas."""
    params = params or load_twin_params()
    mcfg = mcfg or load_metrics_config()
    horizons = np.asarray(mcfg["horizons"], int)
    use_se2 = bool(mcfg.get("use_se2", False))
    tw = load_twin_params()
    robot_radius = float(_get_robot_radius())

    seg_id, seg_name, t0s, bins, razoes, indep = [], [], [], [], [], []
    EPT, EAT, EPD, EAD = [], [], [], []
    block_windows = 1
    for si, (rt, rd) in enumerate(zip(twin_rollouts, default_rollouts)):
        assert rt.meta["h_max"] == rd.meta["h_max"], "h_max divergente twin/default"
        ts = rt.meta["ts"]
        stride = rt.meta["stride"]
        h_max = rt.meta["h_max"]
        step = max(1, math.ceil(h_max / stride))          # janelas por ~H_max
        block_windows = max(block_windows, max(1, round(mcfg["bootstrap"]["block_samples"] / stride)))
        nw = len(rt.windows)
        assert nw == len(rd.windows), "n janelas divergente twin/default"
        for wi in range(nw):
            wt, wd = rt.windows[wi], rd.windows[wi]
            assert wt.t0 == wd.t0, "t0 divergente (janelas nao pareadas)"
            ept, eat = _window_errors(wt, horizons)
            epd, ead = _window_errors(wd, horizons)
            _, _, razao = _window_activity(wt, robot_radius, ts)
            seg_id.append(si); seg_name.append(rt.meta.get("event"))
            t0s.append(wt.t0); razoes.append(razao)
            bins.append(classify_bin(razao, mcfg["bins"]))
            indep.append(wi % step == 0)                  # nao-sobreposta dentro do segmento
            EPT.append(ept); EAT.append(eat); EPD.append(epd); EAD.append(ead)

    epos_twin = np.array(EPT); eang_twin = np.array(EAT)
    epos_def = np.array(EPD); eang_def = np.array(EAD)
    ese2_twin = ese2_def = None
    if use_se2:
        ese2_twin = np.sqrt(epos_twin**2 + (robot_radius * eang_twin)**2)
        ese2_def = np.sqrt(epos_def**2 + (robot_radius * eang_def)**2)

    return ErrorTable(horizons=horizons, ts=ts, robot_radius=robot_radius,
                      seg_id=np.array(seg_id), seg_name=seg_name, t0=np.array(t0s),
                      bin=np.array(bins), razao=np.array(razoes), indep=np.array(indep, bool),
                      epos_twin=epos_twin, eang_twin=eang_twin, epos_def=epos_def,
                      eang_def=eang_def, ese2_twin=ese2_twin, ese2_def=ese2_def,
                      block_windows=block_windows)


def _get_robot_radius() -> float:
    p = repo_root() / "config" / "twin_params.json"
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return float(cfg.get("robot_radius", 0.0796))


# =========================================================================
# Estratificacao / distribuicao
def bin_distribution(table: ErrorTable) -> dict:
    out = {}
    for b in BIN_ORDER:
        m = table.bin == b
        out[b] = {"total": int(m.sum()), "indep": int((m & table.indep).sum())}
    out["all"] = {"total": table.W, "indep": int(table.indep.sum())}
    return out


def format_bin_distribution(table: ErrorTable) -> str:
    d = bin_distribution(table)
    lines = ["Distribuicao de bins (razao = ativ_rot/(ativ_rot+ativ_trans)):",
             f"  {'bin':14s} {'total':>7s} {'indep':>7s}"]
    for b in BIN_ORDER:
        lines.append(f"  {BIN_LABEL[b]:14s} {d[b]['total']:7d} {d[b]['indep']:7d}")
    lines.append(f"  {'TODOS':14s} {d['all']['total']:7d} {d['all']['indep']:7d}")
    return "\n".join(lines)


# =========================================================================
# Bootstrap moving-block (sobre o conjunto denso)
def _block_pool(rows: np.ndarray, seg_id: np.ndarray, L: int) -> list:
    pool = []
    for s in np.unique(seg_id[rows]):
        idx = rows[seg_id[rows] == s]                  # linhas do segmento (ordem densa)
        ws = idx.size
        if ws <= L:
            pool.append(idx)
        else:
            for start in range(0, ws - L + 1):
                pool.append(idx[start:start + L])
    return pool


def _resample(pool: list, W: int, rng) -> np.ndarray:
    chosen, total = [], 0
    while total < W:
        b = pool[rng.integers(len(pool))]
        chosen.append(b); total += b.size
    return np.concatenate(chosen)[:W]


def _mbb(rows, seg_id, L, W, n_boot, rng, fn) -> np.ndarray:
    pool = _block_pool(rows, seg_id, L)
    return np.array([fn(_resample(pool, W, rng)) for _ in range(n_boot)])


def _ci(dist: np.ndarray, ci: float) -> np.ndarray:
    lo = (1 - ci) / 2 * 100
    return np.percentile(dist, [lo, 100 - lo], axis=0).T   # (nh,2)


# =========================================================================
# Effect size (rank-biserial p/ Wilcoxon pareado, direcao default-twin)
def rank_biserial(default: np.ndarray, twin: np.ndarray) -> float:
    d = default - twin
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    rp = r[d > 0].sum(); rm = r[d < 0].sum()
    tot = rp + rm
    return float((rp - rm) / tot) if tot > 0 else 0.0


def _wilcoxon_less(twin: np.ndarray, default: np.ndarray) -> float:
    """p de twin < default (pareado). Robusto a empates totais."""
    if np.allclose(twin, default):
        return 1.0
    try:
        return float(wilcoxon(twin, default, alternative="less", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


# =========================================================================
def evaluate(twin_rollouts, default_rollouts, *, params=None, mcfg=None) -> dict:
    """Calcula curvas (densas), inferencia (nao-sobreposta + bootstrap) e veredito
    por bin. Retorna um dict com tudo para figuras/relatorio."""
    params = params or load_twin_params()
    mcfg = mcfg or load_metrics_config()
    table = build_error_table(twin_rollouts, default_rollouts, params=params, mcfg=mcfg)
    H = table.horizons
    nb = mcfg["bootstrap"]["n_boot"]; ci = mcfg["bootstrap"]["ci"]
    rng = np.random.default_rng(mcfg["bootstrap"]["seed"])
    L = table.block_windows

    scopes = {}
    for scope in ("all",) + BIN_ORDER:
        rows = np.flatnonzero(table.mask(scope))
        indep_rows = np.flatnonzero(table.mask(scope) & table.indep)
        res = {"n_total": rows.size, "n_indep": indep_rows.size, "horizons": H,
               "horizons_s": H * table.ts}
        if rows.size == 0:
            scopes[scope] = res
            continue
        ept, epd = table.epos_twin, table.epos_def
        eat, ead = table.eang_twin, table.eang_def
        # --- curvas (densas) ---
        res["epos_twin_med"] = np.median(ept[rows], 0)
        res["epos_def_med"] = np.median(epd[rows], 0)
        res["eang_twin_med"] = np.median(eat[rows], 0)
        res["eang_def_med"] = np.median(ead[rows], 0)
        res["gap_epos"] = res["epos_def_med"] - res["epos_twin_med"]
        # --- bandas CI (bootstrap densas) ---
        W = rows.size
        res["epos_twin_ci"] = _ci(_mbb(rows, table.seg_id, L, W, nb, rng,
                                       lambda ix: np.median(ept[ix], 0)), ci)
        res["epos_def_ci"] = _ci(_mbb(rows, table.seg_id, L, W, nb, rng,
                                      lambda ix: np.median(epd[ix], 0)), ci)
        res["eang_twin_ci"] = _ci(_mbb(rows, table.seg_id, L, W, nb, rng,
                                       lambda ix: np.median(eat[ix], 0)), ci)
        res["eang_def_ci"] = _ci(_mbb(rows, table.seg_id, L, W, nb, rng,
                                      lambda ix: np.median(ead[ix], 0)), ci)
        res["gap_epos_ci"] = _ci(_mbb(rows, table.seg_id, L, W, nb, rng,
                                      lambda ix: np.median(epd[ix], 0) - np.median(ept[ix], 0)), ci)
        # --- inferencia (nao-sobreposta) ---
        p = np.ones(H.size); eff = np.zeros(H.size)
        if indep_rows.size >= 2:
            for k in range(H.size):
                p[k] = _wilcoxon_less(ept[indep_rows, k], epd[indep_rows, k])
                eff[k] = rank_biserial(epd[indep_rows, k], ept[indep_rows, k])
        res["p_wilcoxon"] = p
        res["effect"] = eff
        # divergencia: Wilcoxon significativo mas bootstrap CI inclui 0
        sig = p < mcfg["verdict"]["alpha"]
        boot_excl0 = res["gap_epos_ci"][:, 0] > 0
        res["divergence"] = bool(np.any(sig & ~boot_excl0))
        scopes[scope] = res

    verdicts = {b: _verdict_bin(b, scopes[b], params, mcfg) for b in BIN_ORDER}
    return {"table": table, "scopes": scopes, "verdicts": verdicts,
            "bin_distribution": bin_distribution(table), "mcfg": mcfg, "params": params}


def _verdict_bin(bin_name: str, res: dict, params, mcfg) -> dict:
    h = np.asarray(res["horizons"]); h_ref = mcfg["verdict"]["h_ref"]
    n_min = mcfg["verdict"]["n_min_independent"]
    alpha = mcfg["verdict"]["alpha"]; eff_min = mcfg["verdict"]["effect_size_min"]
    out = {"bin": bin_name, "n_indep": res.get("n_indep", 0), "n_total": res.get("n_total", 0)}
    ref_mask = h >= h_ref
    if res.get("n_total", 0) and "gap_epos" in res:
        kref = int(np.flatnonzero(ref_mask)[0])
        out.update(gap_at_href=float(res["gap_epos"][kref]),
                   p_at_href=float(res["p_wilcoxon"][kref]),
                   effect_at_href=float(res["effect"][kref]))

    # GATE DE PLACEHOLDER (omega): misto e rot ficam indeterminados
    if params["w"].placeholder and bin_name in ("misto", "rot"):
        out["verdict"] = "INDETERMINADO (omega placeholder)"
        return out
    # GATE DE POPULACAO
    if out["n_indep"] < n_min:
        out["verdict"] = f"SEM PODER (N={out['n_indep']})"
        return out
    # CRITERIOS GO (sobre horizontes >= h_ref)
    gap = res["gap_epos"][ref_mask]
    p = res["p_wilcoxon"][ref_mask]
    eff = res["effect"][ref_mask]
    boot_lo = res["gap_epos_ci"][ref_mask, 0]
    c_dir = bool(np.all(gap > 0))                                   # twin < default
    c_incr = bool(gap[-1] > gap[0] + 1e-9) and bool(np.all(np.diff(gap) > -1e-3))
    c_sig = bool(np.all(p < alpha))
    c_eff = bool(np.all(eff >= eff_min))
    c_boot = bool(np.all(boot_lo > 0))
    go = c_dir and c_incr and c_sig and c_eff and c_boot
    out["criteria"] = dict(direction=c_dir, increasing=c_incr, wilcoxon=c_sig,
                           effect=c_eff, bootstrap=c_boot)
    out["divergence"] = bool(res.get("divergence", False))
    out["verdict"] = "GO" if go else "NO-GO"
    return out


# =========================================================================
def make_figures(result: dict, outdir, stamp: str) -> list:
    """e_pos e e_ang vs horizonte (twin vs default) com bandas de CI; agregado e
    por bin. Salva PNG + SVG. Retorna caminhos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    files = []
    C_TWIN, C_DEF = (0.10, 0.30, 0.65), (0.40, 0.40, 0.40)
    for scope in ("all",) + BIN_ORDER:
        res = result["scopes"][scope]
        if res.get("n_total", 0) == 0 or "epos_twin_med" not in res:
            continue
        hs = res["horizons_s"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for a, key, ttl, ylab in ((ax[0], "epos", "RMSE posicao", "e_pos mediana (m)"),
                                   (ax[1], "eang", "erro heading", "e_ang mediana (rad)")):
            tw, df = res[f"{key}_twin_med"], res[f"{key}_def_med"]
            tw_ci, df_ci = res[f"{key}_twin_ci"], res[f"{key}_def_ci"]
            a.fill_between(hs, df_ci[:, 0], df_ci[:, 1], color=C_DEF, alpha=0.18)
            a.fill_between(hs, tw_ci[:, 0], tw_ci[:, 1], color=C_TWIN, alpha=0.18)
            a.plot(hs, df, "-o", color=C_DEF, lw=1.8, label="ideal (default)")
            a.plot(hs, tw, "-d", color=C_TWIN, lw=2.0, label="twin (FOPDT)")
            a.set_xlabel("horizonte h (s)"); a.set_ylabel(ylab); a.set_title(ttl)
            a.grid(alpha=0.3); a.legend(loc="upper left")
        vb = result["verdicts"].get(scope)
        vtxt = f" — {vb['verdict']}" if vb else ""
        fig.suptitle(f"Fidelidade vs horizonte [{BIN_LABEL.get(scope, 'TODOS')}] "
                     f"(N={res['n_total']}, indep={res['n_indep']}){vtxt}")
        fig.tight_layout()
        base = outdir / f"M1_metrics_{scope}_{stamp}"
        fig.savefig(f"{base}.png", dpi=150); fig.savefig(f"{base}.svg")
        plt.close(fig)
        files.append(f"{base}.png")
    return files


def markdown_report(result: dict, fig_files=None) -> str:
    mcfg = result["mcfg"]; href = mcfg["verdict"]["h_ref"]
    nmin = mcfg["verdict"]["n_min_independent"]; ts = result["table"].ts
    L = ["# Relatorio — metrica SE(2) multi-horizonte (phase0_replay)",
         "",
         f"Config: horizontes(amostras)={list(result['table'].horizons)} "
         f"(s={[round(float(x),3) for x in result['table'].horizons*ts]}), "
         f"h_ref={href}, N_min={nmin}, alpha={mcfg['verdict']['alpha']}, "
         f"effect_min={mcfg['verdict']['effect_size_min']}, bloco_MBB={result['table'].block_windows} janelas.",
         "", "## Distribuicao de bins", "",
         "| bin | total (densas) | independentes |", "|---|---:|---:|"]
    d = result["bin_distribution"]
    for b in BIN_ORDER:
        L.append(f"| {BIN_LABEL[b]} | {d[b]['total']} | {d[b]['indep']} |")
    L.append(f"| **TODOS** | {d['all']['total']} | {d['all']['indep']} |")

    for scope in ("all",) + BIN_ORDER:
        res = result["scopes"][scope]
        L += ["", f"## {BIN_LABEL.get(scope,'TODOS (agregado)')}"]
        if res.get("n_total", 0) == 0 or "epos_twin_med" not in res:
            L.append("_(sem janelas)_"); continue
        L.append(f"N total={res['n_total']}, independentes={res['n_indep']}")
        L += ["", "| h (s) | twin e_pos | ideal e_pos | gap (m) | gap CI | p (nao-sobr.) | effect |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for k, h in enumerate(res["horizons"]):
            cik = res["gap_epos_ci"][k]
            L.append(f"| {h*ts:.2f} | {res['epos_twin_med'][k]:.4f} | {res['epos_def_med'][k]:.4f} "
                     f"| {res['gap_epos'][k]:+.4f} | [{cik[0]:+.4f},{cik[1]:+.4f}] "
                     f"| {res['p_wilcoxon'][k]:.3g} | {res['effect'][k]:+.3f} |")
        if res.get("divergence"):
            L.append("\n> ⚠ DIVERGENCIA: Wilcoxon significativo mas bootstrap CI inclui 0 em algum h "
                     "(a sobreposicao pode estar enganando).")

    L += ["", "## Vereditos por bin (gates aplicados)", "",
          "| bin | veredito | N indep | gap@h_ref | p@h_ref | effect@h_ref |",
          "|---|---|---:|---:|---:|---:|"]
    for b in BIN_ORDER:
        v = result["verdicts"][b]
        g = f"{v.get('gap_at_href'):+.4f}" if v.get("gap_at_href") is not None else "-"
        pp = f"{v.get('p_at_href'):.3g}" if v.get("p_at_href") is not None else "-"
        ee = f"{v.get('effect_at_href'):+.3f}" if v.get("effect_at_href") is not None else "-"
        L.append(f"| {BIN_LABEL[b]} | {v['verdict']} | {v['n_indep']} | {g} | {pp} | {ee} |")
    if fig_files:
        L += ["", "## Figuras"] + [f"- `{f}`" for f in fig_files]
    return "\n".join(L) + "\n"
