from typing import Optional, Type, TypeVar, Union

import numpy as np
import torch as th

from torch.nn import functional as F
from stable_baselines3.sac import SAC


from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy

SelfSAC = TypeVar("SelfSAC", bound="SAC")


class RSAC(SAC):
    """
        Risk Sensitive Soft Actor-Critic (RSAC)

        Citation: This code is based on the stable-baselines3 SAC 
        @article{stable-baselines3,
            author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
            title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
            journal = {Journal of Machine Learning Research},
            year    = {2021},
            volume  = {22},
            number  = {268},
            pages   = {1-8},
            url     = {http://jmlr.org/papers/v22/20-1364.html}
            }
    """
    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        ent_coef: float = 0.1,
        risk_coef: float = 0.01,
        tensorboard_log: Optional[str] = None, 
        verbose: int = 0, 
        seed: Optional[int] = None, 
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True
        ):
        super().__init__(policy,env,learning_rate, buffer_size, 100, batch_size,
             tau, gamma, 1, 1, None, None, None, None, ent_coef, 1, "auto", False,
             -1, False, 100, tensorboard_log, None, verbose, seed, device, _init_setup_model)
        
        self.ent_coef = ent_coef
        self.risk_coef = risk_coef

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]

        # Update learning rate
        self._update_learning_rate(optimizers)

        ent_coef = self.ent_coef    # coefficient for the entropy term
        risk_coef = self.risk_coef  # eta in the paper

        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
         
            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
        
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # Risk sensitive SAC is modified by the original version SAC with
                # following transformation:
                # T_eta(v) := (e^(eta*v) - 1)/eta [line 265, page 8]
                # Note that we omit -1/eta because it does not influence 
                # the gradient or MSE calculation 
                target_q_values = (1 - replay_data.dones) * self.gamma * next_q_values # the second term of J_V
                target_q_values = 1 / risk_coef * th.exp(risk_coef * target_q_values) # Do the T_eta(v) transformation for the second term of J_V


            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(1/risk_coef*th.exp(risk_coef*(current_q-replay_data.rewards)), target_q_values) for current_q in current_q_values)
            # Do the T_eta(v) transformation for the first term of J_Q
            # Then calculate the MSE loss
            # Note: In practice, we only maintain Q and policy networks
            #       just like the stable-baselines3 version did. 
            
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            # Do the T_eta(v) transformation for J_pi
            actor_loss = 1/risk_coef*th.exp(risk_coef*(ent_coef * log_prob - min_qf_pi)).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", ent_coef)
        self.logger.record("train/risk_coef", risk_coef)
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))