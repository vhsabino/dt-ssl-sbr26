#!/usr/bin/env python3
"""
analyze_omega_excitation.py.

HIPÓTESE TESTADA:
    A zona morta é ELÉTRICA e POR RODA, aplicada DEPOIS da cinemática inversa.

"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/extracted/2026-05-18_19-2-15/splits"
GEOM = json.loads((ROOT / "config/robot_geometry.json").read_text(encoding="utf-8"))

R = GEOM["physical"]["R_contact_m"]          # 0.079601 m
r = GEOM["physical"]["wheel_radius_m"]       # 0.029492 m
R_OVER_R_FW = GEOM["firmware"]["R_over_r_implied"]        # 2.61
MOTOR_MIN = GEOM["firmware"]["motor_min_speed_rad_s"]     # 1.567 rad/s
DZ_LO, DZ_HI = GEOM["firmware"]["omega_cmd_deadzone_rad_s_range"]  # 0.581, 0.600

COMPETITION_LABELS = ["front_to_back", "side_to_side", "shoot_to_goal", "square"]
# grade de limiares para a sensibilidade pedida pelo revisor (C9):
# bracket empírico (0.5, 1.0] + os dois limiares geométricos
THRESHOLDS = [0.5, DZ_LO, DZ_HI, 1.0]


def load_commands() -> pd.DataFrame:
    """Concatena os commands.csv de todos os eventos de competição."""
    rows = []
    for label in COMPETITION_LABELS:
        d = SPLITS / label
        if not d.is_dir():
            continue
        for ev in sorted(p for p in d.iterdir() if p.is_dir()):
            f = ev / "commands.csv"
            if not f.is_file():
                continue
            df = pd.read_csv(f, usecols=["move_x", "move_y", "move_w"])
            df["label"] = label
            df["event"] = ev.name
            rows.append(df)
    if not rows:
        sys.exit(f"ERRO: nenhum commands.csv encontrado sob {SPLITS}")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd = load_commands()
    cmd = cmd.dropna(subset=["move_x", "move_y", "move_w"])

    w = cmd["move_w"].to_numpy(float)
    vmag = np.hypot(cmd["move_x"].to_numpy(float), cmd["move_y"].to_numpy(float))

    # termos do setpoint por roda (rad/s), independentes de ângulo
    t_rms = vmag / (r * np.sqrt(2.0))       # translação, RMS entre as 4 rodas
    y_term = R_OVER_R_FW * np.abs(w)        # yaw, idêntico nas 4 rodas

    n = len(cmd)
    out = {
        "_meta": {
            "gerado_em": ts,
            "script": "scripts/analyze_omega_excitation.py",
            "n_amostras": int(n),
            "n_eventos": int(cmd["event"].nunique()),
            "labels": COMPETITION_LABELS,
            "constantes": {
                "R_contact_m": R, "wheel_radius_m": r,
                "R_over_r_firmware": R_OVER_R_FW,
                "motor_min_speed_rad_s": MOTOR_MIN,
                "deadzone_corpo_rad_s": [DZ_LO, DZ_HI],
            },
            "fonte_constantes": "config/robot_geometry.json",
        }
    }

    # ---- C9: sensibilidade da fração sub-limiar ---------------------------
    sens = []
    for th in THRESHOLDS:
        frac_all = float(np.mean(np.abs(w) < th))
        sub = cmd["label"] == "shoot_to_goal"
        frac_stg = float(np.mean(np.abs(w[sub.to_numpy()]) < th))
        sens.append({"limiar_rad_s": th,
                     "frac_abaixo_pooled_pct": 100 * frac_all,
                     "frac_abaixo_shoot_to_goal_pct": 100 * frac_stg})
    out["C9_sensibilidade_limiar"] = sens

    # ---- B4: a translação mantém as rodas acima do clamp? ----------------
    below = np.abs(w) < DZ_LO          # amostras "sub-limiar" do argumento atual
    nb = int(below.sum())
    b4 = {
        "definicao": ("entre as amostras com |w_cmd| < limiar de corpo, qual "
                      "fração tem setpoint de roda (RMS) acima do clamp do motor"),
        "n_amostras_sub_limiar": nb,
        "frac_sub_limiar_pct": 100.0 * nb / n,
        "t_rms_mediana_rad_s": float(np.median(t_rms[below])),
        "t_rms_p10_rad_s": float(np.percentile(t_rms[below], 10)),
        "t_rms_p90_rad_s": float(np.percentile(t_rms[below], 90)),
        "frac_rodas_acima_do_clamp_pct": 100.0 * float(np.mean(t_rms[below] > MOTOR_MIN)),
        "vmag_mediana_m_s": float(np.median(vmag[below])),
    }
    # razão sinal-ruído do termo de yaw dentro do setpoint de roda
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(t_rms[below] > 0, y_term[below] / t_rms[below], np.nan)
    b4["snr_yaw_sobre_translacao_mediana"] = float(np.nanmedian(snr))
    b4["snr_yaw_sobre_translacao_p90"] = float(np.nanpercentile(snr, 90))
    # controle: rotação quase pura em competição (onde o clamp REALMENTE morde)
    quase_pura = below & (t_rms < MOTOR_MIN)
    b4["frac_rotacao_quase_pura_pct"] = 100.0 * float(np.mean(quase_pura))
    out["B4_zona_morta_vs_translacao"] = b4

    # ---- robustez: varredura sobre conjuntos plausíveis de ângulos -------
    # A métrica RMS acima é independente de ângulo, mas o CLAMP é aplicado
    # por roda ao setpoint TOTAL |t_i + y|. Como os ângulos de montagem não
    # estão documentados, varremos conjuntos plausíveis de robôs SSL
    # (par dianteiro +-a, par traseiro +-b) e verificamos se a conclusão muda.
    vx_b = cmd["move_x"].to_numpy(float)[below]
    vy_b = cmd["move_y"].to_numpy(float)[below]
    w_b = w[below]
    sweep = []
    for a_deg in (30.0, 45.0, 60.0):
        for b_deg in (120.0, 135.0, 150.0):
            ang = np.radians([a_deg, b_deg, -b_deg, -a_deg])
            # setpoint por roda: (1/r)(-sin a * vx + cos a * vy) + (R/r) * w
            t_i = (-np.sin(ang)[:, None] * vx_b + np.cos(ang)[:, None] * vy_b) / r
            sp = t_i + R_OVER_R_FW * w_b[None, :]
            clamped = np.abs(sp) < MOTOR_MIN            # (4, N)
            sweep.append({
                "angulos_deg": [a_deg, b_deg, -b_deg, -a_deg],
                "frac_rodas_clampadas_pct": 100.0 * float(clamped.mean()),
                "frac_amostras_com_4_rodas_clampadas_pct":
                    100.0 * float(clamped.all(axis=0).mean()),
                "frac_amostras_com_alguma_roda_ativa_pct":
                    100.0 * float((~clamped).any(axis=0).mean()),
            })
    out["B4_robustez_angulos"] = {
        "nota": ("varredura sobre conjuntos plausíveis de ângulos porque os "
                 "ângulos reais não estão em config/robot_geometry.json; a "
                 "conclusão de B4 deve ser invariante ao conjunto escolhido"),
        "varredura": sweep,
        "frac_amostras_com_alguma_roda_ativa_min_pct":
            min(s["frac_amostras_com_alguma_roda_ativa_pct"] for s in sweep),
        "frac_amostras_com_alguma_roda_ativa_max_pct":
            max(s["frac_amostras_com_alguma_roda_ativa_pct"] for s in sweep),
    }

    # ---- contraste com o dataset dedicado (rotação pura) -----------------
    out["B4_contraste_rotate"] = {
        "nota": ("no dataset `rotate` vx=vy=0 por construção, logo t_rms=0 e o "
                 "clamp morde exatamente no limiar de corpo — é o unico regime "
                 "em que o joelho de zona morta é observável no yaw de corpo"),
        "t_rms_rad_s": 0.0,
    }

    # ---- saída ------------------------------------------------------------
    res_dir = ROOT / "results"
    jf = res_dir / f"omega_excitation_analysis_{ts}.json"
    cf = res_dir / f"omega_threshold_sensitivity_{ts}.csv"
    jf.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(sens).to_csv(cf, index=False)

    # ---- relatório em tela ------------------------------------------------
    print(f"amostras={n}  eventos={cmd['event'].nunique()}  (esperado 46)")
    print("\n--- C9: sensibilidade da fração sub-limiar ---")
    print(pd.DataFrame(sens).to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    print("\n--- B4: entre as amostras sub-limiar ---")
    for k, v in b4.items():
        if k != "definicao":
            print(f"  {k:42s} {v}")
    print("\n--- B4: robustez aos ângulos de montagem ---")
    rb = out["B4_robustez_angulos"]
    print(pd.DataFrame(rb["varredura"]).to_string(index=False,
                                                  float_format=lambda x: f"{x:8.2f}"))
    print(f"  alguma roda ativa: min={rb['frac_amostras_com_alguma_roda_ativa_min_pct']:.2f}% "
          f"max={rb['frac_amostras_com_alguma_roda_ativa_max_pct']:.2f}%")
    print(f"\nartefatos:\n  {jf.relative_to(ROOT)}\n  {cf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
