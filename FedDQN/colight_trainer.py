from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from Env.make_env import make_multi_envs
from Env.road_graph_builder import GraphSpec
from FedDQN.buffer import ReplayBufferGraph
from FedDQN.client import DQNConfig, GraphDQNAgent
from FedDQN.networks import QNetwork
from FedDQN.utils import build_action_dict, reorder_state


@dataclass
class TrainMetrics:
    steps: int
    loss_mean: Optional[float]
    epsilon: float
    episode_returns: List[float]
    mean_episode_return: Optional[float]
    best_episode_return: Optional[float]


class CoLightTrainer:
    """Centralized CoLight-style trainer over the full traffic-light graph."""

    def __init__(
        self,
        graph: GraphSpec,
        env_cfg: Dict[str, Any],
        dqn_config: DQNConfig,
        hidden_dim: int = 128,
        heads: int = 4,
        learning_rate: float = 3e-4,
        dueling: bool = False,
        num_envs: int = 4,
        replay_capacity: int = 5000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        max_steps_per_ep: int = 300,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.graph = graph
        self.node_id_list = graph.node_id_list or graph.idx_to_tls_id
        self.tls_id_to_idx = {tls_id: idx for idx, tls_id in enumerate(self.node_id_list)}
        self.max_steps_per_ep = max_steps_per_ep
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.global_step = 0
        self.last_epsilon = epsilon_start
        self.num_envs = max(1, num_envs)

        edge_index = graph.neighbors_edge_index(add_reverse=True, add_self_loops=True)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        self.edge_index = edge_index.to(self.device)

        log_path = Path(env_cfg["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.envs = []
        for env_idx in range(self.num_envs):
            env_log_path = log_path.with_name(f"global_{env_idx}")
            env = make_multi_envs(
                tls_ids=self.node_id_list,
                sumo_cfg=env_cfg["sumo_cfg"],
                num_seconds=env_cfg["num_seconds"],
                use_gui=env_cfg.get("use_gui", False),
                net_file=env_cfg["net_file"],
                trip_info=env_cfg.get("trip_info"),
                tls_action_type=env_cfg["tls_action_type"],
                log_path=str(env_log_path),
            )
            self.envs.append(env)

        init_state, _ = self.envs[0].reset()
        sample_entry = next(iter(init_state.values()))
        self.feature_spec = {
            "occupancy_dim": len(sample_entry.get("occupancy", [])),
            "phase_dim": len(sample_entry.get("phase", [])),
        }
        self.action_dim = 2 if env_cfg["tls_action_type"] == "next_or_not" else max(1, self.feature_spec["phase_dim"] * 2)

        def optimizer_builder(model: nn.Module):
            return torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.q_net = QNetwork(
            occupancy_dim=self.feature_spec["occupancy_dim"],
            phase_dim=self.feature_spec["phase_dim"],
            hidden_dim=hidden_dim,
            action_dim=self.action_dim,
            heads=heads,
            tau=1.0,
            dropout=0.1,
            dueling=dueling,
        ).to(self.device)
        self.target_net = QNetwork(
            occupancy_dim=self.feature_spec["occupancy_dim"],
            phase_dim=self.feature_spec["phase_dim"],
            hidden_dim=hidden_dim,
            action_dim=self.action_dim,
            heads=heads,
            tau=1.0,
            dropout=0.1,
            dueling=dueling,
        ).to(self.device)
        self.agent = GraphDQNAgent(
            q_network=self.q_net,
            target_network=self.target_net,
            edge_index=self.edge_index,
            action_dim=self.action_dim,
            config=dqn_config,
            optimizer_builder=optimizer_builder,
            device=self.device,
        )
        self.replay = ReplayBufferGraph(capacity=replay_capacity, device=self.device)
        self.env_states = [self._init_env_state(self.envs[0], state=init_state, episode_idx=0)]
        for env in self.envs[1:]:
            self.env_states.append(self._init_env_state(env, episode_idx=0))

    def _init_env_state(self, env, state: Optional[Dict[str, Any]] = None, episode_idx: int = 0) -> Dict[str, Any]:
        if state is None:
            state, _ = env.reset()
        x_tensor = reorder_state(
            state,
            tls_id_to_idx=self.tls_id_to_idx,
            node_id_list=self.node_id_list,
            phase_dim=self.feature_spec["phase_dim"],
            device=self.device,
        )
        num_nodes = len(self.node_id_list)
        return {
            "env": env,
            "x": x_tensor,
            "done_mask": torch.zeros(num_nodes, dtype=torch.bool, device=self.device),
            "action_mask": torch.ones((num_nodes, self.action_dim), device=self.device),
            "episode_step": 0,
            "episode_idx": episode_idx,
            "episode_return": 0.0,
        }

    def train_steps(self, num_steps: int) -> TrainMetrics:
        steps = 0
        losses: List[float] = []
        episode_returns: List[float] = []
        while steps < num_steps:
            epsilon = max(
                self.epsilon_end,
                self.epsilon_start - self.global_step / float(self.epsilon_decay),
            )
            active_indices = []
            x_batch = []
            action_mask_batch = []

            for idx, env_state in enumerate(self.env_states):
                if env_state["episode_step"] >= self.max_steps_per_ep or env_state["done_mask"].all():
                    episode_returns.append(float(env_state["episode_return"]))
                    self.env_states[idx] = self._init_env_state(
                        env_state["env"],
                        episode_idx=env_state["episode_idx"] + 1,
                    )
                    env_state = self.env_states[idx]

                active_indices.append(idx)
                x_batch.append(env_state["x"])
                action_mask_batch.append(env_state["action_mask"])

            x_batch_t = torch.stack(x_batch, dim=0)
            action_mask_t = torch.stack(action_mask_batch, dim=0)
            actions_batch = self.agent.act(
                x_batch_t,
                action_mask=action_mask_t,
                epsilon=epsilon,
            )

            last_action_mask_next = None
            for batch_idx, env_idx in enumerate(active_indices):
                env_state = self.env_states[env_idx]
                actions = actions_batch[batch_idx].view(-1)
                actions_dict = build_action_dict(actions, self.node_id_list)

                next_state, reward_dict, truncated_dict, done_dict, infos = env_state["env"].step(actions_dict)
                rewards = torch.tensor(
                    [reward_dict[tls] for tls in self.node_id_list],
                    device=self.device,
                    dtype=torch.float32,
                )
                step_return = float(rewards.sum().item())
                dones = torch.tensor(
                    [done_dict[tls] or truncated_dict[tls] for tls in self.node_id_list],
                    device=self.device,
                    dtype=torch.bool,
                )
                next_x = reorder_state(
                    next_state,
                    tls_id_to_idx=self.tls_id_to_idx,
                    node_id_list=self.node_id_list,
                    phase_dim=self.feature_spec["phase_dim"],
                    device=self.device,
                )
                next_action_mask = torch.tensor(
                    [infos.get(tls, {}).get("can_perform_action", True) for tls in self.node_id_list],
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(-1).expand(-1, self.action_dim)

                self.replay.add(env_state["x"], actions, rewards, next_x, dones)
                self.env_states[env_idx]["x"] = next_x
                self.env_states[env_idx]["done_mask"] = dones
                self.env_states[env_idx]["action_mask"] = next_action_mask
                self.env_states[env_idx]["episode_step"] += 1
                self.env_states[env_idx]["episode_return"] += step_return
                last_action_mask_next = next_action_mask

            if last_action_mask_next is not None:
                loss_val = self.agent.optimize(self.replay, action_mask_next=last_action_mask_next)
                if loss_val is not None:
                    losses.append(loss_val)

            self.global_step += len(active_indices)
            steps += len(active_indices)
            self.last_epsilon = epsilon

        return TrainMetrics(
            steps=steps,
            loss_mean=float(sum(losses) / len(losses)) if losses else None,
            epsilon=self.last_epsilon,
            episode_returns=episode_returns,
            mean_episode_return=float(sum(episode_returns) / len(episode_returns)) if episode_returns else None,
            best_episode_return=max(episode_returns) if episode_returns else None,
        )

    def save(self, save_dir: Path, tag: str) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), save_dir / f"q_net_{tag}.pt")
        torch.save(self.target_net.state_dict(), save_dir / f"target_net_{tag}.pt")

    def close(self) -> None:
        for env_state in self.env_states:
            env_state["env"].close()
