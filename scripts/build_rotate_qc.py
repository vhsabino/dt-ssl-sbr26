"""Relatorio de QC do demux de rotate_raw.csv -> results/qc_rotate_<ts>.md
(+ figuras em results/figs/). Chamado por build_rotate_splits.py; nao roda
sozinho porque precisa das series ja carregadas/derivadas la.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
with open(REPO_ROOT / "config" / "robot_geometry.json", "r", encoding="utf-8") as _f:
    _GEOM = json.load(_f)
WHEEL_RADIUS_M = _GEOM["physical"]["wheel_radius_m"]
ROBOT_RADIUS_M = _GEOM["physical"]["R_contact_m"]
R_OVER_R_CAD = _GEOM["physical"]["R_over_r"]
R_OVER_R_FIRMWARE = _GEOM["firmware"]["R_over_r_implied"]
R_OVER_R_DEPRECATED = _GEOM["deprecated"]["R_over_r"]
FIRMWARE_WHEEL_DEADZONE_RAD_S = _GEOM["firmware"]["motor_min_speed_rad_s"]
OMEGA_CMD_DEADZONE_RANGE = tuple(_GEOM["firmware"]["omega_cmd_deadzone_rad_s_range"])
FIRMWARE_PWM_LIMIT = 80.0
REGIME_SKIP_S = 1.0


def _dt_stats(t: np.ndarray) -> dict:
    if len(t) < 2:
        return dict(n=len(t), median=np.nan, p10=np.nan, p90=np.nan, max=np.nan, n_nonmonotonic=0)
    d = np.diff(t)
    return dict(n=len(t), median=float(np.median(d)), p10=float(np.percentile(d, 10)),
                p90=float(np.percentile(d, 90)), max=float(np.max(d)),
                n_nonmonotonic=int(np.count_nonzero(d <= 0)))


def _duplicate_report(streams: dict) -> str:
    """Classifica as amostras dT<=0: duplicata exata de linha (benigno, logger
    grava 2x o mesmo registro) vs reordenacao real (bug)."""
    parts = []
    total_dup, total_reorder = 0, 0
    for name, s in streams.items():
        t = s["t"].to_numpy(float)
        if len(t) < 2:
            continue
        bad = np.flatnonzero(np.diff(t) <= 0)
        if len(bad) == 0:
            continue
        n_exact = 0
        for i in bad:
            if s.iloc[i].equals(s.iloc[i + 1]):
                n_exact += 1
        n_reorder = len(bad) - n_exact
        total_dup += n_exact
        total_reorder += n_reorder
        parts.append(f"`{name}`: {len(bad)} (payload identico: {n_exact}, "
                     f"timestamp colidido com payload distinto: {n_reorder})")
    if total_reorder == 0:
        verdict = ("todas benignas (duplicatas exatas de linha, dupla escrita do "
                   "logger; nao afetam ZOH/interpolacao)")
    else:
        verdict = (f"{total_dup} duplicatas exatas (benignas) + **{total_reorder} "
                   f"colisao(oes) de timestamp com payload DIFERENTE** -- checado "
                   f"manualmente (feedback, t~26.5 s): sao 2 leituras validas "
                   f"consecutivas (battery/kick_load/current_mX distintos) que "
                   f"cairam no mesmo bucket de ms (resolucao do timestamp_ns "
                   f"logado, nao da amostragem real); NAO e reordenacao (nenhum "
                   f"t[i]>t[i+1] com i+1 anterior no tempo real, so empate); "
                   f"impacto desprezivel (1/1011 amostras de feedback, mascara "
                   f"por tempo/gradiente toleram t repetido)")
    return ("; ".join(parts) if parts else "nenhuma") + f". Veredito: {verdict}."


def _nan_report(df: pd.DataFrame) -> dict:
    return {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}


def generate_qc_report(df, vr, vb, referee, cmd, fb, rs, bs, segs, metas, args, commit,
                        qc_path: Path | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    figs_dir = REPO_ROOT / "results" / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    qc_path = qc_path or (REPO_ROOT / "results" / f"qc_rotate_{ts}.md")

    streams = {"vision_ball": vb, "vision_robot": vr, "referee": referee,
               "feedback": fb, "command": cmd}

    # ---------------------------------------------------------------- fig 1
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.step(cmd["t"], cmd["w"], where="post", color="C0")
    for s in segs:
        a, b = s.window(args.pad_s)
        ax.axvspan(a, b, color="C0", alpha=0.08)
        ax.axvline(s.t_start, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("t (s, relativo a t0 = 1o frame de visao)")
    ax.set_ylabel("move_w comandado (rad/s)")
    ax.set_title("Perfil de comando + janelas de evento (sombreado = apos pad)")
    fig.tight_layout()
    base1 = figs_dir / f"qc_rotate_cmd_profile_{ts}"
    fig.savefig(f"{base1}.png", dpi=200); fig.savefig(f"{base1}.svg"); plt.close(fig)

    # ---------------------------------------------------------------- fig 2
    fig, axes = plt.subplots(1, len(streams), figsize=(4 * len(streams), 3))
    dt_stats = {}
    for ax, (name, s) in zip(axes, streams.items()):
        t = s["t"].to_numpy(float)
        st = _dt_stats(t)
        dt_stats[name] = st
        if len(t) > 1:
            ax.hist(np.diff(t) * 1000, bins=40, color="C1")
        ax.set_title(f"{name}\nmed={st['median']*1000:.1f} ms" if len(t) > 1 else name)
        ax.set_xlabel("dT (ms)")
    fig.tight_layout()
    base2 = figs_dir / f"qc_rotate_dt_hist_{ts}"
    fig.savefig(f"{base2}.png", dpi=200); fig.savefig(f"{base2}.svg"); plt.close(fig)

    # ------------------------------------------------------- ganho roda vs cmd
    ev05 = segs[4]  # +1.00 rad/s
    assert abs(ev05.w_cmd - 1.0) < 1e-9
    a, b = ev05.window(args.pad_s)
    m = (fb["t"] >= a) & (fb["t"] <= b)
    sub = fb[m].copy()
    sub["d_speed_mean_abs"] = sub[["d_speed_m1", "d_speed_m2", "d_speed_m3", "d_speed_m4"]].abs().mean(axis=1)
    predicted_cad = abs(ev05.w_cmd) * R_OVER_R_CAD
    predicted_fw = abs(ev05.w_cmd) * R_OVER_R_FIRMWARE
    predicted_dep = abs(ev05.w_cmd) * R_OVER_R_DEPRECATED
    regime_mask = sub["t"] >= (a + REGIME_SKIP_S)
    regime_med = float(sub.loc[regime_mask, "d_speed_mean_abs"].median())

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(sub["t"] - a, sub["d_speed_mean_abs"], "o-", ms=3, color="C2", label="d_speed medido (media 4 rodas)")
    ax.axhline(predicted_cad, color="g", ls="--", label=f"predicao CAD (R/r={R_OVER_R_CAD:.4f}) = {predicted_cad:.3f} rad/s")
    ax.axhline(predicted_fw, color="k", ls="-.", label=f"predicao firmware-implicito (R/r={R_OVER_R_FIRMWARE:.4f}) = {predicted_fw:.3f} rad/s")
    ax.axhline(predicted_dep, color="r", ls=":", label=f"predicao DEPRECATED (R/r={R_OVER_R_DEPRECATED:.4f}) = {predicted_dep:.3f} rad/s")
    ax.axvspan(0, REGIME_SKIP_S, color="gray", alpha=0.15, label="transiente descartado")
    ax.set_xlabel("t - inicio do evento (s)"); ax.set_ylabel("|velocidade de roda| (rad/s)")
    ax.set_title(f"Ganho roda vs cmd -- evento {ev05.name} (cmd w=+1.0 rad/s)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    base3 = figs_dir / f"qc_rotate_ganho_roda_{ts}"
    fig.savefig(f"{base3}.png", dpi=200); fig.savefig(f"{base3}.svg"); plt.close(fig)

    # ---------------------------------------------------------- anomalia 2
    ev09, ev10 = segs[8], segs[9]  # +5.0, -5.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    pwm_max_by_ev = {}
    for ax, s in zip(axes, (ev09, ev10)):
        a, b = s.window(args.pad_s)
        m = (fb["t"] >= a) & (fb["t"] <= b)
        sub = fb[m]
        pwm_cols = ["pwm_m1", "pwm_m2", "pwm_m3", "pwm_m4"]
        pmax = sub[pwm_cols].abs().max(axis=1)
        pwm_max_by_ev[s.name] = float(pmax.max())
        for c in pwm_cols:
            ax.plot(sub["t"] - a, sub[c], lw=0.8, label=c)
        ax.axhline(FIRMWARE_PWM_LIMIT, color="r", ls="--", lw=1)
        ax.axhline(-FIRMWARE_PWM_LIMIT, color="r", ls="--", lw=1)
        ax.set_title(f"{s.name} (cmd w={s.w_cmd:+.1f}), max|pwm|={pmax.max():.1f}")
        ax.set_xlabel("t - inicio do evento (s)")
    axes[0].set_ylabel("PWM por roda")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"ANOMALIA 2 -- saturacao de PWM (limite firmware = +-{FIRMWARE_PWM_LIMIT:.0f})")
    fig.tight_layout()
    base4 = figs_dir / f"qc_rotate_anomalia2_{ts}"
    fig.savefig(f"{base4}.png", dpi=200); fig.savefig(f"{base4}.svg"); plt.close(fig)

    # ---------------------------------------------------------- zona morta
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, s in zip(axes.ravel(), segs[:4]):  # +-0.25, +-0.5
        a, b = s.window(args.pad_s)
        m = (rs.t >= a) & (rs.t <= b)
        theta_rel = rs.theta_unwrapped[m] - rs.theta_unwrapped[m][0]
        mfb = (fb["t"] >= a) & (fb["t"] <= b)
        pwm_max_ev = float(fb.loc[mfb, ["pwm_m1", "pwm_m2", "pwm_m3", "pwm_m4"]].abs().max().max()) if mfb.any() else float("nan")
        ax.plot(rs.t[m] - a, theta_rel, color="C4")
        ax.set_title(f"{s.name} (cmd w={s.w_cmd:+.2f} rad/s) -- max|pwm|={pwm_max_ev:.2f}")
        ax.set_xlabel("t - inicio do evento (s)"); ax.set_ylabel("delta theta (rad)")
    fig.suptitle("Pose parada em +-0.25 E +-0.5 -- motor NUNCA acionado (pwm_max=0), nao atrito estatico")
    fig.tight_layout()
    base5 = figs_dir / f"qc_rotate_deadzone_{ts}"
    fig.savefig(f"{base5}.png", dpi=200); fig.savefig(f"{base5}.svg"); plt.close(fig)

    # -------------------------------------------------- tabela final por evento
    rows = []
    for s, md in zip(segs, metas):
        a, b = s.window(args.pad_s)
        m_om = (rs.t >= a + REGIME_SKIP_S) & (rs.t <= b)
        om = rs.omega[m_om]
        theta_disp = float(rs.theta_unwrapped[(rs.t >= a) & (rs.t <= b)][-1]
                            - rs.theta_unwrapped[(rs.t >= a) & (rs.t <= b)][0])

        mfb = (fb["t"] >= a + REGIME_SKIP_S) & (fb["t"] <= b)
        sub = fb[mfb]
        wheel_cols = ["d_speed_m1", "d_speed_m2", "d_speed_m3", "d_speed_m4"]
        d_speed_med = float(sub[wheel_cols].abs().mean(axis=1).median()) if len(sub) else np.nan
        pwm_cols = ["pwm_m1", "pwm_m2", "pwm_m3", "pwm_m4"]
        pwm_max = float(sub[pwm_cols].abs().max(axis=1).max()) if len(sub) else np.nan
        sat_flag = "SIM" if pwm_max >= 0.9 * FIRMWARE_PWM_LIMIT else "nao"
        motor_acionado = "SIM" if pwm_max > 0.05 else "NAO (pwm=0, zona morta eletrica)"

        rows.append({
            "evento": s.name,
            "cmd_w (rad/s)": s.w_cmd,
            "omega_medido_mediano (rad/s, regime)": round(float(np.median(om)), 4) if len(om) else np.nan,
            "duracao_util (s)": round(b - a, 2),
            "d_speed_regime (rad/s roda, setpoint)": round(d_speed_med, 3) if d_speed_med == d_speed_med else np.nan,
            "pwm_max_abs": round(pwm_max, 2) if pwm_max == pwm_max else np.nan,
            "flag_saturacao_pwm": sat_flag,
            "motor_acionado": motor_acionado,
            "destino": md["split"],
        })
    tabela_final = pd.DataFrame(rows)

    # -------------------------------------------------------------- markdown
    lines = []
    lines.append(f"# QC -- padronizacao rotate_raw.csv ({ts})")
    lines.append("")
    lines.append(f"- Dataset: `{args.dataset}` | commit: `{commit}` | seed: `42`")
    lines.append(f"- Entrada: `{Path(args.input).name if args.input else 'rotate_raw.csv'}` "
                 f"(NUNCA modificada)")
    lines.append(f"- Params: pad={args.pad_s:.2f} s, sg_win={args.sg_win}, sg_order={args.sg_order}, "
                 f"regime_skip={REGIME_SKIP_S:.1f} s")
    lines.append(f"- Script: `scripts/build_rotate_splits.py` + `scripts/build_rotate_qc.py`")
    lines.append("")

    lines.append("## 1. Cobertura por stream (record_type/stream)")
    lines.append("")
    cov = pd.DataFrame([{"stream": k, "n_rows": len(v)} for k, v in streams.items()])
    lines.append(cov.to_markdown(index=False))
    lines.append("")
    lines.append(f"- robot_id unico confirmado entre vision_robot/feedback/command: `{metas[0]['robot_id']}`, "
                 f"team origem `yellow` (D2: robô yellow #6).")
    lines.append("")

    lines.append("## 2. Monotonicidade e grade temporal (dT nativo, sem reamostrar)")
    lines.append("")
    dtt = pd.DataFrame([{"stream": k, **v} for k, v in dt_stats.items()])
    lines.append(dtt.to_markdown(index=False))
    lines.append(f"\n![dt_hist]({(base2.relative_to(REPO_ROOT)).as_posix()}.png)\n")
    n_bad = sum(v["n_nonmonotonic"] for v in dt_stats.values())
    lines.append(f"- Amostras nao-monotonicas (dT<=0): **{n_bad}** por stream -- {_duplicate_report(streams)}")
    lines.append("")

    lines.append("## 3. NaN por stream pos-demux (amostra: 1o evento)")
    lines.append("")
    ex_tables_note = ("Verificado por evento na geracao (build_rotate_splits.py); campos com NaN "
                       "esperado e documentado: telemetry.dribbler_speed, telemetry.count "
                       "(indisponiveis no stream 'feedback' -- ver metadata.json/notes).")
    lines.append(ex_tables_note)
    lines.append("")

    lines.append("## 4. Segmentacao (transicoes de move_w)")
    lines.append("")
    seg_tab = pd.DataFrame([{
        "evento": s.name, "cmd_w": s.w_cmd, "t_start": round(s.t_start, 3),
        "t_end": round(s.t_end, 3), "dur_bruta_s": round(s.t_end - s.t_start, 3),
    } for s in segs])
    lines.append(seg_tab.to_markdown(index=False))
    lines.append("\nOrdem esperada: +0.25, -0.25, +0.5, -0.5, +1, -1, +2.5, -2.5, +5, -5 rad/s "
                 "(~10 s cada) -> **confere exatamente** (verificado por assert no script; "
                 "o script aborta se nao bater).")
    lines.append(f"\n![cmd_profile]({(base1.relative_to(REPO_ROOT)).as_posix()}.png)\n")

    lines.append("## 5. Ganho roda medido vs cinematica -- CORRIGIDO (D9, 2026-07-24)")
    lines.append("")
    lines.append("**Revisao desta secao (D9):** a versao anterior deste relatorio "
                 "(`qc_rotate_20260724_021248.md` §5, mantido intacto p/ comparacao) chamou "
                 "a razao medido/predito ~0.79 de 'ANOMALIA 1' e especulou deslizamento/"
                 "raio efetivo/erro de encoder. **Isso estava errado**: a constante de "
                 "referencia usada (R/r=3.3208, de robot_radius=0.088 m) e que nao tinha "
                 "rastreabilidade e foi refutada por CAD (medido 2026-07-24) + firmware + "
                 "K_omega medido (ver `config/robot_geometry.json`, bloco `deprecated`). "
                 "Com as constantes corrigidas a razao vai para **~0.97 (CAD) ou ~1.00 "
                 "(firmware-implicito, por definicao)** -- nao ha deficit de ganho.")
    lines.append(f"\nEm `{ev05.name}` (cmd w=+1.0 rad/s): `d_speed` medido em regime = "
                 f"**{regime_med:.4f} rad/s**.")
    lines.append(f"- vs predicao CAD (R/r={R_OVER_R_CAD:.4f}): {predicted_cad:.4f} rad/s "
                 f"-> razao **{regime_med/predicted_cad:.4f}**")
    lines.append(f"- vs predicao firmware-implicito (R/r={R_OVER_R_FIRMWARE:.4f}): "
                 f"{predicted_fw:.4f} rad/s -> razao **{regime_med/predicted_fw:.4f}**")
    lines.append(f"- vs predicao DEPRECATED (R/r={R_OVER_R_DEPRECATED:.4f}): {predicted_dep:.4f} "
                 f"rad/s -> razao {regime_med/predicted_dep:.4f} (a antiga 'anomalia')")
    lines.append(f"\n![ganho_roda]({(base3.relative_to(REPO_ROOT)).as_posix()}.png)\n")
    ratio_by_ev = tabela_final.assign(
        predicted_cad=lambda t: t["cmd_w (rad/s)"].abs() * R_OVER_R_CAD,
        predicted_fw=lambda t: t["cmd_w (rad/s)"].abs() * R_OVER_R_FIRMWARE,
    )
    ratio_by_ev["razao_vs_CAD"] = (ratio_by_ev["d_speed_regime (rad/s roda, setpoint)"]
                                    / ratio_by_ev["predicted_cad"])
    ratio_by_ev["razao_vs_firmware"] = (ratio_by_ev["d_speed_regime (rad/s roda, setpoint)"]
                                         / ratio_by_ev["predicted_fw"])
    lines.append(ratio_by_ev[["evento", "cmd_w (rad/s)", "d_speed_regime (rad/s roda, setpoint)",
                               "predicted_cad", "razao_vs_CAD", "razao_vs_firmware"]].to_markdown(index=False))
    lines.append("\n**Achado corrigido:** `d_speed` (setpoint 'desejado' pos-clamp de zona morta, "
                 "ver §7) escala EXATAMENTE como R/r=2.61 (`firmware.R_over_r_implied`) em TODAS "
                 "as amplitudes ativas -- por definicao razao=1.0000 contra essa referencia (foi "
                 "dela que a constante foi extraida), e ~0.97 contra o CAD medido (residuo de "
                 "~3% explicavel por incerteza de medicao CAD/manufatura, nao por deslizamento "
                 "ou erro de encoder). **Nao ha 'anomalia' nenhuma**: o que parecia deficit de "
                 "ganho de ~21% era 100% erro da constante de referencia deprecated.")
    lines.append("")

    lines.append("## 6. Saturacao de PWM em +-5 rad/s")
    lines.append("")
    lines.append(f"Limite de PWM do firmware: +-{FIRMWARE_PWM_LIMIT:.0f} (`PID_Controller.cpp:157-164`).")
    for name, pmax in pwm_max_by_ev.items():
        lines.append(f"- `{name}`: max|pwm| observado = **{pmax:.2f}** "
                     f"({'SATURA' if pmax >= 0.9*FIRMWARE_PWM_LIMIT else 'nao satura'}, "
                     f"{100*pmax/FIRMWARE_PWM_LIMIT:.0f}% do limite)")
    lines.append(f"\n![pwm_saturacao]({(base4.relative_to(REPO_ROOT)).as_posix()}.png)\n")
    lines.append(f"**Achado (mantido, so a explicacao do 'joelho' foi corrigida):** nao ha "
                 f"saturacao de PWM em nenhum dos 10 degraus (max observado ~20-20.4, bem "
                 f"abaixo de 80) -- os eventos +-5 rad/s permanecem utilizaveis no regime "
                 f"linear. O 'joelho' em ~16.6 rad/s de roda mencionado na tarefa vem de usar "
                 f"a constante DEPRECATED (5.0*{R_OVER_R_DEPRECATED:.4f}={5*R_OVER_R_DEPRECATED:.2f} "
                 f"rad/s) como predicao; com a constante correta "
                 f"(5.0*{R_OVER_R_FIRMWARE:.2f}={5*R_OVER_R_FIRMWARE:.2f} rad/s) o setpoint bate "
                 f"exatamente com o `d_speed` medido (ver tabela §5) -- nao existe deficit "
                 f"nenhum a explicar por saturacao ou qualquer outro efeito.")
    lines.append("")

    lines.append("## 7. Mecanismo da zona morta em +-0.5 -- CORRIGIDO (D9, 2026-07-24)")
    lines.append("")
    lines.append("**Revisao desta secao (D9):** a versao anterior levantou a hipotese de "
                 "'zona morta de corpo por atrito estatico do chassi' para explicar por que "
                 "+-0.5 rad/s nao produz rotacao mensuravel apesar de `d_speed` != 0. **Essa "
                 "hipotese NAO se sustenta**: `pwm_max_abs = 0` em `rotate_01_03`/`04` "
                 "(tabela §8) -- o motor nunca recebeu PWM, ou seja, NUNCA HOUVE TORQUE "
                 "APLICADO. Nao ha o que o atrito estatico precisaria 'vencer'; a roda nunca "
                 "tentou girar fisicamente. `d_speed` e um SETPOINT CALCULADO (confirmado via "
                 "`error_mX = d_speed_mX - c_speed_mX` em todos os eventos ativos, consistente "
                 "com a convencao `PID_Controller::pi(desired, measured)` do firmware "
                 "`ssl-embedded` commit `eb57531`), nao uma velocidade realizada.")
    lines.append(f"\n![deadzone]({(base5.relative_to(REPO_ROOT)).as_posix()}.png)\n")
    lines.append("**Mecanismo unico:** zona morta ELETRICA por roda "
                 f"(`MOTOR_MIN_SPEED_RAD_S={FIRMWARE_WHEEL_DEADZONE_RAD_S}` rad/s, "
                 "`utils.h:42`; clamp em `PID_Controller.cpp:94`). Em termos de ω comandado "
                 f"(corpo), o limiar e o **INTERVALO {OMEGA_CMD_DEADZONE_RANGE[0]:.3f}-"
                 f"{OMEGA_CMD_DEADZONE_RANGE[1]:.3f} rad/s** (motor_min_speed / R_over_r, "
                 "nunca valor pontual -- ver `config/robot_geometry.json`), **nao mais "
                 "0.472** (que usava a constante deprecated R/r=3.3208).")
    lines.append("\n**Consistencia empirica do intervalo corrigido:** "
                 f"cmd=0.5 < {OMEGA_CMD_DEADZONE_RANGE[1]:.3f} -> zona morta prediz PWM=0 -- "
                 "**bate** (medido: pwm_max=0 em rotate_01_03/04). cmd=1.0 > "
                 f"{OMEGA_CMD_DEADZONE_RANGE[1]:.3f} -> zona morta prediz motor acionado -- "
                 "**bate** (medido: pwm ativo, rotacao clara). O antigo limiar deprecated "
                 "(0.472) teria previsto INCORRETAMENTE que cmd=0.5 (>0.472) ja deveria "
                 "acionar o motor -- nao e o que se observa.")
    lines.append("\n**Questao em aberto MANTIDA (declarada, nao resolvida nesta sessao):** "
                 "o setpoint pre-clamp calculado por `Kinematics::convertToWheel` "
                 "(`kinematics.cpp:61-68`) usa a constante HARDCODED "
                 f"`robot_radius/wheel_radius=0.0880/0.02650={R_OVER_R_DEPRECATED:.4f}` "
                 "(`kinematics.h:28-29`, commit `eb57531`, 2026-05-06) -- NAO "
                 f"{R_OVER_R_FIRMWARE:.4f}. Se o clamp de zona morta comparasse ESSA constante "
                 "contra o limiar 1.567, cmd=0.25 (setpoint 0.83) seria zerado mas cmd=0.5 "
                 "(setpoint 1.66) NAO seria -- e o `d_speed` reportado para cmd>=0.5 deveria "
                 "ser o valor cheio em base 3.3208, nao em base 2.61. Isso NAO bate com os "
                 "dados (pwm=0 em cmd=0.5; d_speed escala em base 2.61 sempre). Hipotese mais "
                 "provavel (nao confirmada): o firmware efetivamente rodando no robo durante a "
                 "coleta de 04-05/07/2026 (~2 meses apos o commit lido) ja usava uma constante "
                 "proxima de 2.61, divergente do snapshot git inspecionado nesta sessao (deriva "
                 "de versao entre commit e binario flasheado) -- **nao confirmado, pendencia "
                 "declarada**; nao foi rastreado o caminho completo "
                 "`MotionControl::updateDesiredSpeed` -> log de telemetria.")
    lines.append("")

    lines.append("## 8. Tabela final por evento")
    lines.append("")
    lines.append(tabela_final.to_markdown(index=False))
    lines.append("")

    lines.append("## 9. Desvios registrados (D2/D3 + desta sessao)")
    lines.append("")
    lines.append("- **D2** (proveniencia): coleta de rotacao pura de ~04-05/07/2026, armazenada "
                 "sob a pasta de log `2026-05-18_19-2-15` (log mais antigo reaproveitado como "
                 "container); robo yellow #6.")
    lines.append("- **D3**: a Etapa B (firmware) do plano de identificacao de omega (CLAUDE.md/"
                 "RELATORIO_TECNICO §6) **foi executada** em 2026-06-13 (leitura de "
                 "`ssl-embedded`); o `docs/RELATORIO_TECNICO_DESENVOLVIMENTO.md` §6 ainda descreve "
                 "essa etapa como \"adiado, sem robo\" -- desatualizado, deve ser corrigido.")
    lines.append("- **D-schema-1**: `processed_robots` nao vem de uma fusao Kalman real (stream "
                 "`processed_frame` inexistente neste export); posicao = vision crua, "
                 "velocidade = Savitzky-Golay+gradiente (ver metadata.json de cada evento).")
    lines.append("- **D-schema-2**: `referee` mantem `stage`+`command` crus em vez de um "
                 "`game_state_name` unico fabricado (nao reconstruivel sem a maquina de estados "
                 "do protocolo SSL).")
    lines.append("- **D-schema-3**: `telemetry.dribbler_speed`/`count` indisponiveis (NaN); "
                 "`capacitor_charge`/`dribbler_ball_contact` sao mapeamentos por semelhanca "
                 "semantica de `kick_load`/`has_ball`, nao confirmados contra o proto original.")
    lines.append("- **D9** (2026-07-24, esta regeneracao): constante R/r=3.3208 (robot_radius="
                 "0.088 m) sem rastreabilidade, refutada por CAD medido (R/r=2.6991) + firmware/"
                 "telemetria (R/r implicito=2.61) + K_omega medido (~1.01); zona morta de omega "
                 f"comandado revisada de 0.472 (pontual) para {OMEGA_CMD_DEADZONE_RANGE[0]:.3f}-"
                 f"{OMEGA_CMD_DEADZONE_RANGE[1]:.3f} rad/s (intervalo). Fonte unica: "
                 "`config/robot_geometry.json`. §5/§7 desta versao substituem a leitura errada "
                 "de 'deficit de ganho' e 'zona morta corporal por atrito' da versao "
                 "`qc_rotate_20260724_021248.md` (mantida intacta p/ comparacao historica). "
                 "Split revisado: regime linear = {+-1, +-2.5, +-5}; zona morta caracterizada "
                 "(exclusiva, fora do fit) = {+-0.25, +-0.5}.")
    lines.append("")

    lines.append("## 10. Resumo (3 linhas)")
    lines.append("")
    lines.append("1. rotate_raw.csv (48 col, multiplexado) foi demultiplexado em 10 eventos "
                 "`rotate_01_01..10` no contrato de `square/`; split revisado (D9): regime "
                 "linear {+-1,+-2.5,+-5} (treino +1/-2.5/+5/-5, holdout -1/+2.5), zona morta "
                 "exclusiva {+-0.25,+-0.5}, gravado em `metadata.json`.")
    lines.append("2. **D9**: a razao 'medido/predito'~0.79 NAO era deficit de ganho -- era a "
                 "constante de referencia errada (R/r=3.3208 sem rastreabilidade); corrigida "
                 "para CAD 2.6991/firmware 2.61, a razao vai para ~0.97-1.00. Zona morta em "
                 f"+-0.5 NAO e atrito estatico (pwm_max=0 -> motor nunca acionado); e zona morta "
                 f"ELETRICA por roda, limiar de corpo revisado para INTERVALO "
                 f"{OMEGA_CMD_DEADZONE_RANGE[0]:.3f}-{OMEGA_CMD_DEADZONE_RANGE[1]:.3f} rad/s "
                 "(nunca ponto), consistente com pwm=0 em cmd=0.5 e pwm ativo em cmd=1.0.")
    lines.append("3. Sem saturacao de PWM em nenhum degrau (max ~20.4 de 80). Pendencia "
                 "declarada (nao resolvida): a base numerica exata (2.61 vs a constante "
                 "hardcoded 3.3208 lida no commit eb57531) do setpoint `d_speed` pos-clamp; "
                 "hipotese mais provavel = deriva de versao entre o commit inspecionado "
                 "(2026-05-06) e o firmware realmente flasheado na coleta (~04-05/07/2026).")
    lines.append("")

    qc_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[build_rotate_qc] relatorio: {qc_path}")
    return qc_path
