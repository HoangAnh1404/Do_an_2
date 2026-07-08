from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Env.road_graph_builder import RoadGraphBuilder
from FedDQN.client import DQNConfig
from FedDQN.colight_trainer import CoLightTrainer


def run_colight_dqn(
    env_name: str = "4nodes",
    tls_action_type: str = "next_or_not",
    tls_ids: Optional[List[str]] = None,
    rounds: int = 3,
    local_steps: int = 5000,
    hidden_dim: int = 128,
    heads: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    batch_size: int = 512,
    warmup_steps: int = 200,
    target_update_interval: int = 200,
    tau: float = 0.0,
    grad_clip_norm: float = 5.0,
    num_envs: int = 4,
    replay_capacity: int = 20000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 5000,
    max_steps_per_ep: int = 300,
    num_seconds: int = 500,
    use_gui: bool = False,
    device: Optional[str] = None,
    save_dir: str = "FedDQN/result",
    scenario: Optional[str] = None,
    dueling: bool = False,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sumo_cfg = str(Path(f"Scenario/{env_name}/env/vehicle.sumocfg").resolve())
    net_file = str(Path(f"Scenario/{env_name}/env/{env_name}.net.xml").resolve())
    save_root = Path(save_dir) / env_name / tls_action_type
    if scenario:
        save_root = save_root / scenario
    log_path = save_root / "log" / "global.monitor.csv"
    ckpt_root = save_root / "global"

    graph = RoadGraphBuilder.build_from_net_file(
        net_file=net_file,
        tls_ids=tls_ids,
        directed=True,
        make_bidirectional=False,
        include_self_loops=False,
        max_hops_between_tls=1,
        neighbor_strategy="hop",
        neighbor_hop_k=1,
        include_self_in_neighbor=True,
    )
    node_id_list = graph.node_id_list or graph.idx_to_tls_id
    print(
        f"[INFO] Running centralized CoLight-DQN on tls_ids={node_id_list}, "
        f"device={device}, num_envs={num_envs}, batch_size={batch_size}, "
        f"save_root={save_root}"
    )

    env_cfg = {
        "sumo_cfg": sumo_cfg,
        "net_file": net_file,
        "num_seconds": num_seconds,
        "use_gui": use_gui,
        "trip_info": None,
        "tls_action_type": tls_action_type,
        "log_path": str(log_path),
    }
    dqn_cfg = DQNConfig(
        gamma=gamma,
        tau=tau,
        target_update_interval=target_update_interval,
        huber_delta=1.0,
        grad_clip_norm=grad_clip_norm,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
    )

    trainer = CoLightTrainer(
        graph=graph,
        env_cfg=env_cfg,
        dqn_config=dqn_cfg,
        hidden_dim=hidden_dim,
        heads=heads,
        learning_rate=learning_rate,
        dueling=dueling,
        num_envs=num_envs,
        replay_capacity=replay_capacity,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        max_steps_per_ep=max_steps_per_ep,
        device=device,
    )

    try:
        best_score = None
        for rnd in range(1, rounds + 1):
            metrics = trainer.train_steps(local_steps)
            trainer.save(ckpt_root, tag=f"round{rnd}")
            score = metrics.mean_episode_return
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                trainer.save(ckpt_root, tag="best")
                best_meta = {
                    "round": rnd,
                    "score": best_score,
                    "metric": "mean_episode_return",
                    "best_episode_return": metrics.best_episode_return,
                    "episode_returns": metrics.episode_returns,
                }
                (ckpt_root / "best_meta.json").write_text(json.dumps(best_meta, indent=2))
            print(
                f"[ROUND {rnd}] steps={metrics.steps}, "
                f"loss_mean={metrics.loss_mean}, epsilon={metrics.epsilon:.4f}, "
                f"mean_episode_return={metrics.mean_episode_return}, best_score={best_score}"
            )
    finally:
        trainer.close()

    print("Centralized CoLight-DQN training finished.")


def parse_args():
    ap = argparse.ArgumentParser(description="Centralized CoLight-style DQN with GAT for TSC")
    ap.add_argument("--env-name", type=str, default="4nodes")
    ap.add_argument("--tls-action-type", type=str, default="next_or_not")
    ap.add_argument("--tls-ids", type=str, default=None, help="Comma-separated tls ids")
    ap.add_argument("--rounds", type=int, default=3, help="Checkpointing/training chunks")
    ap.add_argument("--local-steps", type=int, default=5000, help="Training steps per round")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--target-update-interval", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--grad-clip-norm", type=float, default=5.0)
    ap.add_argument("--num-envs", type=int, default=4, help="Number of full-network SUMO environments to run in parallel.")
    ap.add_argument("--replay-capacity", type=int, default=20000)
    ap.add_argument("--epsilon-start", type=float, default=1.0)
    ap.add_argument("--epsilon-end", type=float, default=0.05)
    ap.add_argument("--epsilon-decay", type=int, default=5000)
    ap.add_argument("--max-steps-per-ep", type=int, default=300)
    ap.add_argument("--num-seconds", type=int, default=500)
    ap.add_argument("--use-gui", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save-dir", type=str, default="FedDQN/result")
    ap.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional traffic-demand scenario name, e.g. medium_vehicle. Results are saved under save-dir/env-name/tls-action-type/scenario.",
    )
    ap.add_argument("--dueling", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tls_list = args.tls_ids.split(",") if args.tls_ids else None
    run_colight_dqn(
        env_name=args.env_name,
        tls_action_type=args.tls_action_type,
        tls_ids=tls_list,
        rounds=args.rounds,
        local_steps=args.local_steps,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        target_update_interval=args.target_update_interval,
        tau=args.tau,
        grad_clip_norm=args.grad_clip_norm,
        num_envs=args.num_envs,
        replay_capacity=args.replay_capacity,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        max_steps_per_ep=args.max_steps_per_ep,
        num_seconds=args.num_seconds,
        use_gui=args.use_gui,
        device=args.device,
        save_dir=args.save_dir,
        scenario=args.scenario,
        dueling=args.dueling,
    )
