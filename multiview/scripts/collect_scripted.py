"""Collect successful scripted ColorButtonBox demonstrations for one level.

The driver starts --workers worker processes on one GPU. Worker w runs seeds
w, w + W, w + 2W, ... and stages each episode under <out_root>/staging/L<level>/:
    seed_000123/            success: lowdim.npz (GR1 state/action, MuJoCo states, robosuite
                            actions), model.xml.gz, meta.json (incl. ep_meta), <stream>.mp4
    failed/seed_000123.json failure (kept for the success-rate record)
Episodes are written to a temp dir and renamed, so a killed worker leaves nothing
half-written; rerunning the same command resumes. Rendering contexts occasionally
hang under contention, so the driver restarts any worker whose heartbeat stalls.
Once the level has --n_success successes the driver stops the workers.
Convert the staged episodes with scripts/export_lerobot.py (LeRobot + replayable HDF5).

Usage (robocasa_uv python, repo root):
    python multiview/scripts/collect_scripted.py --level 0 --gpu 0 --n_success 200 \
        --out_root /nfs/data/sohyun/data/multiview_gr1/ColorButtonBox
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

PROFILE = "collect"


def staging_dir(out_root: Path, level: int) -> Path:
    return out_root / "staging" / f"L{level}"


def n_successes(staging: Path) -> int:
    return sum(1 for p in staging.glob("seed_*") if p.is_dir())


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------
def run_worker(args):
    import gzip

    import gymnasium as gym
    import numpy as np

    import multiview_tasks  # noqa: F401  (registers gr1_mv/* envs)
    from multiview_tasks.lerobot_io import pack_gr1, video_streams, write_mp4
    from multiview_tasks.registry import env_id
    from multiview_tasks.replay_io import env_args, flat_env_action
    from multiview_tasks.scripted.color_button import ColorButtonScriptedPolicy, trajectory_rng
    from multiview_tasks.tasks import TASK_ENV_NAMES

    staging = staging_dir(args.out_root, args.level)
    failed = staging / "failed"
    failed.mkdir(parents=True, exist_ok=True)
    heartbeat = staging / f".heartbeat_w{args.worker}"
    streams = video_streams(PROFILE)
    env = gym.make(env_id(TASK_ENV_NAMES["ColorButtonBox"][args.level], PROFILE), enable_render=True)
    env_meta_path = staging / "env_args.json"
    if not env_meta_path.exists():
        env_meta_path.write_text(json.dumps(env_args(env.unwrapped)))

    for seed in range(args.worker, args.max_seed, args.workers):
        name = f"seed_{seed:06d}"
        if (staging / name).exists() or (failed / f"{name}.json").exists():
            continue
        if n_successes(staging) >= args.n_success:
            break
        heartbeat.touch()
        obs, _ = env.reset(seed=seed)
        robosuite_env = env.unwrapped.env
        model_xml = robosuite_env.sim.model.get_xml()
        ep_meta = robosuite_env.get_ep_meta()
        policy = ColorButtonScriptedPolicy(env.unwrapped)
        policy.reset(obs, rng=trajectory_rng(seed))
        states, actions, sim_states, env_actions = [], [], [], []
        frames = {k: [] for k, _, _, _ in streams}
        language = obs["annotation.human.coarse_action"]
        for t in range(policy.total_steps + args.idle_steps):
            action = policy.act()
            states.append(pack_gr1(obs, "state."))
            actions.append(pack_gr1(action, "action."))
            sim_states.append(robosuite_env.sim.get_state().flatten())
            env_actions.append(flat_env_action(env.unwrapped, action))
            for key in frames:
                frames[key].append(obs[key])
            obs, _, _, _, info = env.step(action)
            if t % 20 == 0:
                heartbeat.touch()
        success = bool(robosuite_env._check_success())
        meta = dict(
            seed=seed,
            level=args.level,
            success=success,
            language=language,
            target=robosuite_env.target_color,
            button_colors=list(robosuite_env.button_colors),
            length=len(actions),
            phases={k: bool(v) for k, v in info.items() if k.startswith("phase/")},
            ep_meta=ep_meta,
            trajectory={k: np.asarray(v).tolist() for k, v in policy.params.items()},
        )
        if not success:
            (failed / f"{name}.json").write_text(json.dumps(meta))
            continue
        tmp = staging / f".tmp_{name}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        np.savez(
            tmp / "lowdim.npz",
            state=np.stack(states),
            action=np.stack(actions),
            sim_states=np.stack(sim_states),
            env_actions=np.stack(env_actions),
        )
        with gzip.open(tmp / "model.xml.gz", "wt") as f:
            f.write(model_xml)
        for key, lerobot_key, _, pix_fmt in streams:
            write_mp4(tmp / f"{lerobot_key}.mp4", frames[key], pix_fmt)
        (tmp / "meta.json").write_text(json.dumps(meta))
        tmp.rename(staging / name)
    env.close()


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run_driver(args):
    from multiview_tasks.egl_devices import render_env_for_gpu

    staging = staging_dir(args.out_root, args.level)
    staging.mkdir(parents=True, exist_ok=True)
    for tmp in staging.glob(".tmp_seed_*"):  # left by a killed worker
        shutil.rmtree(tmp)
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # --gpu is the nvidia-smi index; robosuite needs the matching EGL device index.
    env = render_env_for_gpu(args.gpu)
    print(f"[L{args.level}] GPU {args.gpu} -> EGL device {env['CUDA_VISIBLE_DEVICES']}", flush=True)

    def start(w):
        cmd = [sys.executable, __file__, "--worker", str(w)] + [
            a for pair in (
                ("--level", str(args.level)),
                ("--workers", str(args.workers)),
                ("--n_success", str(args.n_success)),
                ("--max_seed", str(args.max_seed)),
                ("--idle_steps", str(args.idle_steps)),
                ("--out_root", str(args.out_root)),
            ) for a in pair
        ]
        (staging / f".heartbeat_w{w}").touch()
        log = open(log_dir / f"L{args.level}_w{w}.log", "a")
        return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def kill(proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    procs = {w: start(w) for w in range(args.workers)}
    restarts = 0
    t0 = time.time()
    try:
        while True:
            time.sleep(10)
            done = n_successes(staging)
            n_failed = len(list((staging / "failed").glob("*.json")))
            print(
                f"[L{args.level}] {time.time() - t0:6.0f}s  success {done}/{args.n_success}  "
                f"failed {n_failed}  restarts {restarts}",
                flush=True,
            )
            if done >= args.n_success:
                break
            for w, proc in list(procs.items()):
                stalled = time.time() - (staging / f".heartbeat_w{w}").stat().st_mtime > args.stall_timeout
                if proc.poll() is None and not stalled:
                    continue
                if proc.poll() == 0:  # seed range exhausted
                    procs.pop(w)
                    continue
                print(f"[L{args.level}] restarting worker {w} (exit={proc.poll()}, stalled={stalled})", flush=True)
                kill(proc)
                for tmp in staging.glob(".tmp_seed_*"):
                    if int(tmp.name.split("_")[-1]) % args.workers == w:
                        shutil.rmtree(tmp)
                procs[w] = start(w)
                restarts += 1
            if not procs:
                print(f"[L{args.level}] all seeds below --max_seed used", flush=True)
                break
    finally:
        for proc in procs.values():
            kill(proc)
    print(f"[L{args.level}] finished: {n_successes(staging)} successes in {time.time() - t0:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0, help="nvidia-smi index of the render GPU")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n_success", type=int, default=200)
    parser.add_argument("--max_seed", type=int, default=100_000)
    parser.add_argument("--idle_steps", type=int, default=10)
    parser.add_argument("--stall_timeout", type=float, default=180.0)
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is None:
        run_driver(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
