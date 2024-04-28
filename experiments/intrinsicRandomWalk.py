from typing import Callable
from numpy import ndarray
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, Schedule, TrainFreq
from stable_baselines3.dqn.policies import DQNPolicy
import torch as th 

from stable_baselines3.common.vec_env import VecEnv
from max_sb3.dqn.upolicies import UncertaintyMlpPolicy

from max_sb3.common.uncertainties import RNDUncertaintyStateAction
from max_sb3.common.ubuffers import UncertaintyReplayBuffer
from max_sb3.dqn.udqn import UncertaintyDQN

from max_sb3.common.type_aliases import Schedule
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from typing import Any, Dict, List, Optional, Type
import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
class IntrinsicRandomWalkPolicy(UncertaintyMlpPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        beta: float,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_amp: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            use_amp=use_amp,
            beta=beta
        )
    
    def intrinsicStep(self, obs: th.Tensor) -> th.Tensor:
        values = self.u_net(obs)
        action = values.argmax(dim=1).reshape(-1).detach().cpu().numpy()
        return action

    def forward(self, obs: th.Tensor) -> th.Tensor:
        q_values = self.q_net(obs)
        uncertainties = self.u_net(obs)
        return q_values + self.beta * uncertainties


class IntrinsicRandomWalk(UncertaintyDQN):
    
    def __init__(self, policy: str | type[DQNPolicy], env: GymEnv | VecEnv | str, beta: float = 0.5, random_steps:int=5, rnd_config:dict=None, embed_dim:int=512, learning_rate: float | Callable[[float], float] = 0.0001, buffer_size: int = 1000000, learning_starts: int = 50000, batch_size: int = 32, tau: float = 1, gamma: float = 0.99, train_freq: int | th.Tuple[int | str] = 4, gradient_steps: int = 1, replay_buffer_class: type[ReplayBuffer] | None = None, replay_buffer_kwargs: th.Dict[str, th.Any] | None = None, optimize_memory_usage: bool = False, target_update_interval: int = 10000, double_q: bool = False, exploration_fraction: float = 0.1, exploration_initial_eps: float = 1, exploration_final_eps: float = 0.05, max_grad_norm: float = 10, tensorboard_log: str | None = None, policy_kwargs: th.Dict[str, th.Any] | None = None, verbose: int = 0, seed: int | None = None, device: th.device | str = "auto", _init_setup_model: bool = True, use_amp: bool = False):
        
        replay_buffer_class=UncertaintyReplayBuffer
        policy=IntrinsicRandomWalkPolicy
        
        uncertainty_policy_kwargs = dict(activation_fn = th.nn.ReLU, net_arch=[1024, 1024], learning_rate=0.0001)
        if rnd_config != None:
            uncertainty_policy_kwargs.update(rnd_config)

        uncertainty = RNDUncertaintyStateAction(
                beta, 
                env, 
                embed_dim, 
                buffer_size, 
                uncertainty_policy_kwargs, 
                device=device, 
                flatten_input=True, 
                normalize_images=False)

        # And then add the following replay buffer kwargs: 
        base_replay_buffer_kwargs = {
                        "uncertainty": uncertainty, 
                        "state_action_bonus": True, 
                        "handle_timeout_termination":True, 
                        "uncertainty_of_sampling":True,
                    }
        
        if replay_buffer_kwargs != None:
            base_replay_buffer_kwargs.update(replay_buffer_kwargs)

        base_policy_kwargs = {
            'beta': beta
        }

        if policy_kwargs != None:
            base_policy_kwargs.update(policy_kwargs)

        self.embed_dim=embed_dim
        self.random_steps=random_steps
        
        self.beta = beta
        self.num_envs=env.num_envs
        self.episode_steps_taken=[0]*env.num_envs

        super().__init__(policy, env, beta, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, replay_buffer_class, base_replay_buffer_kwargs, optimize_memory_usage, target_update_interval, double_q, exploration_fraction, exploration_initial_eps, exploration_final_eps, max_grad_norm, tensorboard_log, base_policy_kwargs, verbose, seed, device, _init_setup_model, use_amp)
        self.u_scaler = th.cuda.amp.GradScaler(enabled=self.use_amp)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            # Increase step counts
            self.episode_steps_taken = [x+1 if not done else 0 for x, done in zip(self.episode_steps_taken, dones)]

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:

                    # Reset the random steps taken (again)
                    self.episode_steps_taken[idx] = 0

                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def predict(self, observation: ndarray | Dict[str, ndarray], state: th.Tuple[ndarray] | None = None, episode_start: ndarray | None = None, deterministic: bool = False) -> th.Tuple[ndarray | th.Tuple[ndarray] | None]:
        assert len(self.episode_steps_taken) == 1, "IntrinsicRandomWalk only supports one environment atm"
        if self.episode_steps_taken[0] < self.random_steps and not deterministic:
            obs_tens = th.tensor(observation).to(self.device)
            return self.policy.intrinsicStep(obs_tens), observation
        return super().predict(observation, state, episode_start, deterministic)

if __name__ == "__main__":
    from test_agent import Baseline_CNN,train_config, test_0_config, test_100_config, make_env_fn, DummyVecEnv, EvalCallback
    import torch
    from max_sb3.common.uncertainties import RNDUncertaintyStateAction
    from max_sb3.common.ubuffers import UncertaintyReplayBuffer
    from max_sb3.dqn.upolicies import UncertaintyMlpPolicy
    eps=1.5
    bs=500
    num_envs=1
    embed_dim = 512
    # 0.5 eps==300, 1.0 eps == 150, 1.5 eps == 600
    for eps in [0.5,1.0,1.5]:
        beta=[150,300,600][int(eps*2)-1]
        for rn in range(5):
            env_train=DummyVecEnv([make_env_fn(train_config, seed=0, rank=0)])
            env_test0=DummyVecEnv([make_env_fn(test_0_config, seed=0, rank=0)])
            env_test100=DummyVecEnv([make_env_fn(test_100_config, seed=0, rank=0)])
            path=f"experiments/logs/rnd_{bs}k/{eps}/"

            print("Testing RND")
            config={'policy':UncertaintyMlpPolicy,
                        'env':env_train,
                        'buffer_size': bs*1000,
                        'batch_size': 256,
                        'gamma': 0.99,
                        'learning_starts': 256,
                        'max_grad_norm': 1.0,
                        'gradient_steps': 1,
                        'train_freq': (10//num_envs, 'step'),
                        'target_update_interval': 10,
                        'tau': 0.01,
                        'exploration_fraction': 0.5,
                        'exploration_initial_eps': 1.0,
                        'exploration_final_eps': 0.01,
                        'learning_rate': 2.5e-4,
                        'verbose': 0,
                        'device': 'cuda',
                        'policy_kwargs':{
                            'activation_fn': torch.nn.ReLU,
                            'net_arch': [],
                            'features_extractor_class': Baseline_CNN,
                            'features_extractor_kwargs':{'features_dim': 512},
                            'optimizer_class':torch.optim.Adam,
                            'optimizer_kwargs':{'weight_decay': 1e-5},
                            'normalize_images':False,
                        },
                        'beta': beta,

                    }

            eval_tr_callback = EvalCallback(env_train, log_path=f"{path}/tr/{rn}/", eval_freq=(25000//num_envs),
                                            n_eval_episodes=20, deterministic=True, render=False, verbose=0)

            eval_0_callback = EvalCallback(env_test0, log_path=f"{path}/0/{rn}/", eval_freq=(25000//num_envs),
                                            n_eval_episodes=20, deterministic=True, render=False, verbose=0)

            eval_100_callback = EvalCallback(env_test100, log_path=f"{path}/100/{rn}/", eval_freq=(25000//num_envs),
                                                n_eval_episodes=20, deterministic=True, render=False, verbose=0)

            callbacks=[eval_tr_callback, eval_0_callback, eval_100_callback]


            model = IntrinsicRandomWalk(**config)
            
            model.learn(total_timesteps=500000, progress_bar=True,  log_interval=10,callback=callbacks)



