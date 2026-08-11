"""Padroniza rotate_raw.csv """
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DEFAULT = "2026-05-18_19-2-15"
LOG_IDX = "01"          # unico log de origem (D2: coleta dedicada armazenada sob este log)
SEED = 42
PAD_S_DEFAULT = 0.2
SG_WIN_DEFAULT = 7
SG_ORDER_DEFAULT = 2
REGIME_SKIP_S = 1.0      # descarta o transiente inicial de cada evento p/ estatisticas de regime


def load_robot_geometry(path: Path | None = None) -> dict:
    """Fonte da geometria (config/robot_geometry.json"""
    p = path or (REPO_ROOT / "config" / "robot_geometry.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


_GEOM = load_robot_geometry()
WHEEL_RADIUS_M = _GEOM["physical"]["wheel_radius_m"]
ROBOT_RADIUS_M = _GEOM["physical"]["R_contact_m"]
R_OVER_R_CAD = _GEOM["physical"]["R_over_r"]
R_OVER_R_FIRMWARE = _GEOM["firmware"]["R_over_r_implied"]
R_OVER_R_DEPRECATED = _GEOM["deprecated"]["R_over_r"]
FIRMWARE_WHEEL_DEADZONE_RAD_S = _GEOM["firmware"]["motor_min_speed_rad_s"]
OMEGA_CMD_DEADZONE_RANGE = tuple(_GEOM["firmware"]["omega_cmd_deadzone_rad_s_range"])
FIRMWARE_PWM_LIMIT = 80.0

# ordem cronologica esperada dos 10 degraus (rad/s)
EXPECTED_AMPLITUDES = [0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 2.5, -2.5, 5.0, -5.0]
# Regime linear = {+-1, +-2.5, +-5} (fora da zona cinzenta 0.58-0.60 rad/s do
# limiar de zona morta, ver config/robot_geometry.json/D9); zona morta
# CARACTERIZADA (nao ajustada no fit) = {+-0.25, +-0.5} -- +-0.5 saiu do fit
# linear porque cai perto do limiar (setpoint pre-deadzone ~1.31-1.66 rad/s vs
# MOTOR_MIN_SPEED 1.567): u!=0 mas y quase 0 corrompe P1D linear.
SPLIT_BY_AMPLITUDE = {
    0.25: "deadzone", -0.25: "deadzone",
    0.5: "deadzone", -0.5: "deadzone",
    1.0: "treino", -1.0: "holdout",
    2.5: "holdout", -2.5: "treino",
    5.0: "treino", -5.0: "treino",
}


def git_commit() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                     cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO_ROOT) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "no-commit"


# ============================================================================
# 1) leitura + demultiplexacao
# ============================================================================
@dataclass
class Segment:
    idx: int
    w_cmd: float
    t_start: float   # 1a amostra de commands neste patamar (s, relativo a t0)
    t_end: float      # ultima amostra de commands neste patamar (s, relativo a t0)

    @property
    def name(self) -> str:
        return f"rotate_{LOG_IDX}_{self.idx:02d}"

    def window(self, pad_s: float) -> tuple[float, float]:
        return self.t_start + pad_s, self.t_end - pad_s


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    t0 = df.loc[df["record_type"] == 1, "timestamp_ns"].min()
    df["t"] = (df["timestamp_ns"] - t0) / 1e9
    return df


def stream(df: pd.DataFrame, record_type: int, name: str) -> pd.DataFrame:
    s = df[(df["record_type"] == record_type) & (df["stream"] == name)].sort_values("t", kind="stable")
    return s.reset_index(drop=True)


def detect_command_segments(cmd: pd.DataFrame) -> list[Segment]:
    w = cmd["w"].to_numpy()
    t = cmd["t"].to_numpy()
    change_idx = np.flatnonzero(np.diff(w) != 0)
    raw_segs = []
    start = 0
    for c in change_idx:
        raw_segs.append((w[start], t[start], t[c]))
        start = c + 1
    raw_segs.append((w[start], t[start], t[-1]))
    return [Segment(i + 1, float(wv), float(ts), float(te))
            for i, (wv, ts, te) in enumerate(raw_segs)]


def sg_derivative(pos: np.ndarray, t: np.ndarray, win: int, order: int) -> np.ndarray:
    """Savitzky-Golay smoothing (mesma janela/ordem de identify_trans_dof.m) +
    gradiente pelo t real (grade nativa, possivelmente irregular; nao reamostra)."""
    w = min(win, len(pos) if len(pos) % 2 == 1 else len(pos) - 1)
    if w <= order or w < 3:
        return np.gradient(pos, t)
    return np.gradient(savgol_filter(pos, w, order), t)


# ============================================================================
# 2) series derivadas de log inteiro (evita artefatos de filtro nas bordas
#    dos eventos -- so entao fatiadas por janela)
# ============================================================================
@dataclass
class RobotSeries:
    t: np.ndarray
    theta_wrapped: np.ndarray
    theta_unwrapped: np.ndarray
    omega: np.ndarray          # rad/s, SG(theta_unwrap) -> gradiente
    vel_x_world: np.ndarray    # mm/s
    vel_y_world: np.ndarray    # mm/s


def build_robot_series(vr: pd.DataFrame, sg_win: int, sg_order: int) -> RobotSeries:
    t = vr["t"].to_numpy(float)
    theta_w = vr["orientation"].to_numpy(float)
    theta_u = np.unwrap(theta_w)
    omega = sg_derivative(theta_u, t, sg_win, sg_order)
    vx = sg_derivative(vr["x"].to_numpy(float), t, sg_win, sg_order)
    vy = sg_derivative(vr["y"].to_numpy(float), t, sg_win, sg_order)
    return RobotSeries(t, theta_w, theta_u, omega, vx, vy)


@dataclass
class BallSeries:
    t: np.ndarray
    vel_x: np.ndarray
    vel_y: np.ndarray
    acc_x: np.ndarray
    acc_y: np.ndarray


def build_ball_series(vb: pd.DataFrame, sg_win: int, sg_order: int) -> BallSeries:
    t = vb["t"].to_numpy(float)
    vx = sg_derivative(vb["x"].to_numpy(float), t, sg_win, sg_order)
    vy = sg_derivative(vb["y"].to_numpy(float), t, sg_win, sg_order)
    ax = sg_derivative(vx, t, sg_win, sg_order)
    ay = sg_derivative(vy, t, sg_win, sg_order)
    return BallSeries(t, vx, vy, ax, ay)


# ============================================================================
# 3) montagem por evento (fatia por tempo; timestamp_event relativo ao inicio
#    NOMINAL (padded) do evento, uniforme entre todos os streams)
# ============================================================================
def mask_window(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return (t >= a) & (t <= b)


def build_event_tables(df: pd.DataFrame, seg: Segment, pad_s: float, robot_id: int,
                        rs: RobotSeries, bs: BallSeries,
                        vr: pd.DataFrame, vb: pd.DataFrame,
                        referee: pd.DataFrame, cmd: pd.DataFrame, fb: pd.DataFrame
                        ) -> dict[str, pd.DataFrame]:
    a, b = seg.window(pad_s)

    def tev(t):
        return t - a

    out: dict[str, pd.DataFrame] = {}

    # --- commands ---
    m = mask_window(cmd["t"].to_numpy(float), a, b)
    c = cmd[m]
    out["commands"] = pd.DataFrame({
        "timestamp": c["t"].to_numpy(float),
        "robot_id": robot_id,
        "move_x": c["vx"].to_numpy(float),
        "move_y": c["vy"].to_numpy(float),
        "move_w": c["w"].to_numpy(float),
        "actuation_kick_strength": c["kick_strength"].to_numpy(float),
        "actuation_front": c["front"].to_numpy(bool),
        "actuation_chip": c["chip"].to_numpy(bool),
        "actuation_charge": c["charge"].to_numpy(bool),
        "actuation_dribbler": c["dribbler"].to_numpy(bool),
        "actuation_dribbler_velocity": c["dribbler_speed"].to_numpy(float),
        "timestamp_event": tev(c["t"].to_numpy(float)),
    })

    # --- raw_robots (passthrough; team = 'robots_' + team de origem) ---
    m = mask_window(vr["t"].to_numpy(float), a, b)
    r = vr[m]
    out["raw_robots"] = pd.DataFrame({
        "timestamp": r["t"].to_numpy(float),
        "team": "robots_" + r["team"].astype(str),
        "robot_id": robot_id,
        "position_x": r["x"].to_numpy(float),
        "position_y": r["y"].to_numpy(float),
        "position_w": r["orientation"].to_numpy(float),
        "timestamp_event": tev(r["t"].to_numpy(float)),
    })

    # --- processed_robots (team='allies'; posicao = vision crua [sem fusao
    #     Kalman disponivel neste export -- ver notas/QC]; velocidade =
    #     Savitzky-Golay(pos)+gradiente, mundo, mm/s) ---
    mrs = mask_window(rs.t, a, b)
    vx_w = rs.vel_x_world[mrs]
    vy_w = rs.vel_y_world[mrs]
    out["processed_robots"] = pd.DataFrame({
        "timestamp": rs.t[mrs],
        "team": "allies",
        "robot_id": robot_id,
        "position_x": r["x"].to_numpy(float),
        "position_y": r["y"].to_numpy(float),
        "position_w": r["orientation"].to_numpy(float),
        "velocity_x": vx_w,
        "velocity_y": vy_w,
        "velocity_norm": np.hypot(vx_w, vy_w),
        "timestamp_event": tev(rs.t[mrs]),
    })

    # --- raw_ball / ball ---
    m = mask_window(vb["t"].to_numpy(float), a, b)
    bb = vb[m]
    out["raw_ball"] = pd.DataFrame({
        "timestamp": bb["t"].to_numpy(float),
        "position_x": bb["x"].to_numpy(float),
        "position_y": bb["y"].to_numpy(float),
        "timestamp_event": tev(bb["t"].to_numpy(float)),
    })
    mbs = mask_window(bs.t, a, b)
    vxb = bs.vel_x[mbs]; vyb = bs.vel_y[mbs]
    axb = bs.acc_x[mbs]; ayb = bs.acc_y[mbs]
    out["ball"] = pd.DataFrame({
        "timestamp": bs.t[mbs],
        "position_x": bb["x"].to_numpy(float),
        "position_y": bb["y"].to_numpy(float),
        "velocity_x": vxb, "velocity_y": vyb,
        "acceleration_x": axb, "acceleration_y": ayb,
        "velocity_norm": np.hypot(vxb, vyb),
        "acceleration_norm": np.hypot(axb, ayb),
        "timestamp_event": tev(bs.t[mbs]),
    })

    # --- referee (passthrough; ver nota D5 -- schema historico tem
    #     game_state_name unico, que nao e reconstruivel a partir de
    #     stage+command sem uma maquina de estados do protocolo SSL) ---
    m = mask_window(referee["t"].to_numpy(float), a, b)
    rf = referee[m]
    out["referee"] = pd.DataFrame({
        "timestamp": rf["t"].to_numpy(float),
        "stage": rf["stage"].astype(str),
        "command": rf["command"].astype(str),
        "blue_score": rf["blue_score"].to_numpy(float),
        "yellow_score": rf["yellow_score"].to_numpy(float),
        "timestamp_event": tev(rf["t"].to_numpy(float)),
    })

    # --- telemetry (feedback -> contrato + extensoes p/ QC de anomalias) ---
    m = mask_window(fb["t"].to_numpy(float), a, b)
    f = fb[m]
    out["telemetry"] = pd.DataFrame({
        "timestamp": f["t"].to_numpy(float),
        "robot_id": robot_id,
        "position_x": 0.0, "position_y": 0.0, "position_w": 0.0,   # indisponivel (idem square)
        "velocity_x": f["vx"].to_numpy(float),
        "velocity_y": f["vy"].to_numpy(float),
        "velocity_w": f["w"].to_numpy(float),
        "dribbler_speed": np.nan,                                   # indisponivel no stream feedback
        "capacitor_charge": f["kick_load"].to_numpy(float),         # mapeamento por semelhanca semantica
        "dribbler_ball_contact": f["has_ball"].to_numpy(bool),
        "battery": f["battery"].to_numpy(float),
        "count": np.nan,                                            # indisponivel no stream feedback
        "wheel1": f["d_speed_m1"].to_numpy(float),
        "wheel2": f["d_speed_m2"].to_numpy(float),
        "wheel3": f["d_speed_m3"].to_numpy(float),
        "wheel4": f["d_speed_m4"].to_numpy(float),
        "timestamp_event": tev(f["t"].to_numpy(float)),
        # --- extensoes (fora do contrato historico; necessarias p/ ANOMALIA 2) ---
        "wheel1_cmd": f["c_speed_m1"].to_numpy(float),
        "wheel2_cmd": f["c_speed_m2"].to_numpy(float),
        "wheel3_cmd": f["c_speed_m3"].to_numpy(float),
        "wheel4_cmd": f["c_speed_m4"].to_numpy(float),
        "wheel1_pwm": f["pwm_m1"].to_numpy(float),
        "wheel2_pwm": f["pwm_m2"].to_numpy(float),
        "wheel3_pwm": f["pwm_m3"].to_numpy(float),
        "wheel4_pwm": f["pwm_m4"].to_numpy(float),
        "wheel1_error": f["error_m1"].to_numpy(float),
        "wheel2_error": f["error_m2"].to_numpy(float),
        "wheel3_error": f["error_m3"].to_numpy(float),
        "wheel4_error": f["error_m4"].to_numpy(float),
        "wheel1_current": f["current_m1"].to_numpy(float),
        "wheel2_current": f["current_m2"].to_numpy(float),
        "wheel3_current": f["current_m3"].to_numpy(float),
        "wheel4_current": f["current_m4"].to_numpy(float),
    })

    return out


def write_tables(event_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    event_dir.mkdir(parents=True, exist_ok=True)
    for name, tdf in tables.items():
        tdf.to_csv(event_dir / f"{name}.csv", index=False)
        tdf.to_parquet(event_dir / f"{name}.parquet", index=False)


def write_metadata(event_dir: Path, seg: Segment, pad_s: float, sg_win: int, sg_order: int,
                    dataset: str, source_csv: str, robot_id: int, commit: str) -> dict[str, Any]:
    a, b = seg.window(pad_s)
    md = {
        "event": seg.name,
        "label": "rotate",
        "dataset": dataset,
        "log": dataset,
        "source_file": source_csv,
        "robot_id": robot_id,
        "team_source": "yellow",
        "team_processed": "allies",
        "cmd_w_amplitude_rad_s": seg.w_cmd,
        "segment_index": seg.idx,
        "split": SPLIT_BY_AMPLITUDE[seg.w_cmd],
        "pad_s": pad_s,
        "window_raw_s": [seg.t_start, seg.t_end],
        "window_padded_s": [a, b],
        "duration_padded_s": b - a,
        "sg_win": sg_win,
        "sg_order": sg_order,
        "seed": SEED,
        "seed_note": ("atribuicao treino/holdout/deadzone por amplitude, literal "
                      "(instrucao explicita do experimento), nao regenerada via rng"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "geometry_source": "config/robot_geometry.json (D9, results/CHANGELOG.md 2026-07-24)",
        "notes": [
            "processed_robots.position_* = vision crua (raw_robots); nao ha stream "
            "'processed_frame' (fusao Kalman) neste export multiplexado, so "
            "'vision_robot' raw -- velocity_x/velocity_y sao estimadas via "
            "Savitzky-Golay(win=7,order=2)+gradiente da posicao (mundo, mm/s), "
            "igual identify_trans_dof.m/identify_omega_v3_linear.m.",
        ],
    }
    (event_dir / "metadata.json").write_text(json.dumps(md, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    return md


# ============================================================================
# main
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET_DEFAULT)
    ap.add_argument("--input", default=None, help="path para rotate_raw.csv")
    ap.add_argument("--out-root", default=None, help="dir pai dos rotate_01_XX (default: splits/rotate)")
    ap.add_argument("--pad-s", type=float, default=PAD_S_DEFAULT)
    ap.add_argument("--sg-win", type=int, default=SG_WIN_DEFAULT)
    ap.add_argument("--sg-order", type=int, default=SG_ORDER_DEFAULT)
    ap.add_argument("--no-qc", action="store_true", help="pula o relatorio de QC")
    ap.add_argument("--qc-report", default=None)
    args = ap.parse_args()

    splits_root = REPO_ROOT / "data" / "extracted" / args.dataset / "splits" / "rotate"
    input_path = Path(args.input) if args.input else splits_root / "rotate_raw.csv"
    out_root = Path(args.out_root) if args.out_root else splits_root
    commit = git_commit()

    print(f"[build_rotate_splits] lendo {input_path}")
    df = load_raw(input_path)

    vr = stream(df, 1, "vision_robot")
    vb = stream(df, 1, "vision_ball")
    referee = stream(df, 2, "referee")
    fb = stream(df, 4, "feedback")
    cmd = stream(df, 5, "command")

    robot_ids = pd.unique(pd.concat([vr["robot_id"], fb["robot_id"], cmd["robot_id"]]))
    if len(robot_ids) != 1:
        raise ValueError(f"robot_id inconsistente entre streams: {robot_ids}")
    robot_id = int(robot_ids[0])

    segs = detect_command_segments(cmd)
    print(f"[build_rotate_splits] {len(segs)} segmentos detectados: "
          + ", ".join(f"{s.w_cmd:+.2f}" for s in segs))
    got = [s.w_cmd for s in segs]
    if len(got) != 10 or any(abs(g - e) > 1e-9 for g, e in zip(got, EXPECTED_AMPLITUDES)):
        raise ValueError(f"segmentos detectados {got} != esperado {EXPECTED_AMPLITUDES}")

    rs = build_robot_series(vr, args.sg_win, args.sg_order)
    bs = build_ball_series(vb, args.sg_win, args.sg_order)

    metas = []
    for seg in segs:
        tables = build_event_tables(df, seg, args.pad_s, robot_id, rs, bs, vr, vb, referee, cmd, fb)
        event_dir = out_root / seg.name
        write_tables(event_dir, tables)
        md = write_metadata(event_dir, seg, args.pad_s, args.sg_win, args.sg_order,
                             args.dataset, str(input_path.relative_to(REPO_ROOT)), robot_id, commit)
        metas.append(md)
        print(f"[build_rotate_splits] {seg.name}: w={seg.w_cmd:+.2f} rad/s -> {md['split']:9s} "
              f"({md['duration_padded_s']:.2f} s)")

    if not args.no_qc:
        from build_rotate_qc import generate_qc_report  # local import: mesma pasta scripts/
        qc_path = Path(args.qc_report) if args.qc_report else None
        generate_qc_report(df, vr, vb, referee, cmd, fb, rs, bs, segs, metas,
                            args, commit, qc_path)


if __name__ == "__main__":
    main()
