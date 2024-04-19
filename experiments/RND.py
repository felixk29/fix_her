import numpy as np
import torch as th
from torch.nn import functional as F
from torch import nn

#by felix, used in a couple of files
class RND(nn.Module):
    """
    simple RND module for thesis
    """

    def __init__(self, input_dim, output_dim:int = 512, hidden_dim=1024, device="cpu"):
        super(RND, self).__init__()

        if not isinstance(input_dim, int):
            input_dim = np.prod(input_dim)

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = th.optim.Adam(self.predictor.parameters(), lr=2.5e-4, weight_decay=1e-5)

        self.to(device)

    def forward(self, x: th.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.flatten(start_dim=1)
        x=x.float()

        pred = self.predictor(x)
        target = self.target(x)
        return pred, target

    def train(self, x):
        pred, target = self.forward(x)
        loss = F.l1_loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    


#max version
#https://github.com/MWeltevrede/stable-baselines3/tree/feature/deep-exploration


if __name__ == "__main__":
    from test_agent import Baseline_CNN,train_config, test_0_config, test_100_config, make_env_fn, DummyVecEnv, EvalCallback
    import torch


    eps=0.5
    bs=50
    rn=0
    num_envs=1
    path=f"experiments/logs/rnd_{bs}k/{eps}_{rn}/"


    env_train=DummyVecEnv([make_env_fn(train_config, seed=0, rank=0)])
    env_test0=DummyVecEnv([make_env_fn(test_0_config, seed=0, rank=0)])
    env_test100=DummyVecEnv([make_env_fn(test_100_config, seed=0, rank=0)])

    print(train_config)

    print("Testing RND")
    config={'policy':'CnnPolicy',
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
                'exploration_fraction': eps,
                'exploration_initial_eps': 0.5,
                'exploration_final_eps': 0.01,
                'learning_rate': 2.5e-4,
                'verbose': 0,
                'device': 'cuda',
                'policy_kwargs':{
                    'activation_fn': torch.nn.ReLU,
                    'net_arch': [128, 64],
                    'features_extractor_class': Baseline_CNN,
                    'features_extractor_kwargs':{'features_dim': 512},
                    'optimizer_class':torch.optim.Adam,
                    'optimizer_kwargs':{'weight_decay': 1e-5},
                    'normalize_images':False
                }
            }
    beta=0.5


    uncertainty_policy_kwargs = dict(activation_fn = torch.nn.ReLU, net_arch=[1024, 1024], learning_rate=0.0001)
    config["novelty_kwargs"] = uncertainty_policy_kwargs
    embed_dim = 512
    config["novelty_embed_dim"] = embed_dim
    uncertainty = RNDUncertaintyStateAction(
            beta, 
            train_env, 
            embed_dim, 
            config['buffer_size'], 
            uncertainty_policy_kwargs, 
            device="cuda", 
            flatten_input=True, 
            normalize_images=False)

    # And then add the following replay buffer kwargs: 
    config["replay_buffer_kwargs"] = {
                    "uncertainty": uncertainty, 
                    "state_action_bonus": True, 
                    "handle_timeout_termination":True, 
                    "uncertainty_of_sampling":True,
                }

    eval_tr_callback = EvalCallback(env_train, log_path=f"{path}/tr/{rn}/", eval_freq=(25000//num_envs),
                                    n_eval_episodes=20, deterministic=True, render=False, verbose=0)

    eval_0_callback = EvalCallback(env_test0, log_path=f"{path}/0/{rn}/", eval_freq=(25000//num_envs),
                                    n_eval_episodes=20, deterministic=True, render=False, verbose=0)

    eval_100_callback = EvalCallback(env_test100, log_path=f"{path}/100/{rn}/", eval_freq=(25000//num_envs),
                                        n_eval_episodes=4, deterministic=True, render=False, verbose=0)