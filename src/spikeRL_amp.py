import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import itertools
from copy import deepcopy
import math
import argparse
import time

"""
How to run:
requirements after creating pyframework environment: torch, gym, mujoco==2.3.3, numpy==1.24.0
python spikeRL_risp.py --env Walker2d-v4 --epochs 5
"""
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.

"""
Parameters for SNN
"""

ENCODER_REGULAR_VTH = 0.999
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5


class PseudoEncoderSpikeRegular(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Regular Spike for encoder """
    @staticmethod
    def forward(ctx, input):
        return input.gt(ENCODER_REGULAR_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class PseudoEncoderSpikePoisson(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Poisson Spike for encoder """
    @staticmethod
    def forward(ctx, input):
        return torch.bernoulli(input).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class SpikingEncoderRegularSpike(nn.Module):
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        super().__init__()
        # Ensure obs_dim and pop_dim are integers
        self.obs_dim = int(obs_dim)
        self.pop_dim = int(pop_dim)
        self.encoder_neuron_num = self.obs_dim * self.pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoEncoderSpikeRegular.apply

        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros((1, self.obs_dim, self.pop_dim))  # Ensure dimensions are integers
        delta_mean = (mean_range[1] - mean_range[0]) / (self.pop_dim - 1)
        for num in range(self.pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros((1, self.obs_dim, self.pop_dim)) + std  # Ensure dimensions are integers
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_volt = torch.zeros(batch_size, self.encoder_neuron_num, device=self.device)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # Generate Regular Spike Trains
        for step in range(self.spike_ts):
            pop_volt = pop_volt + pop_act
            pop_spikes[:, :, step] = self.pseudo_spike(pop_volt)
            pop_volt = pop_volt - pop_spikes[:, :, step] * ENCODER_REGULAR_VTH
        return pop_spikes



class SpikingEncoderPoissonSpike(SpikingEncoderRegularSpike):
    """ Learnable Population Coding Spike Encoder with Poisson Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__(obs_dim, pop_dim, spike_ts, mean_range, std, device)
        self.pseudo_spike = PseudoEncoderSpikePoisson.apply

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # Generate Poisson Spike Trains
        for step in range(self.spike_ts):
            pop_spikes[:, :, step] = self.pseudo_spike(pop_act)
        return pop_spikes


class SpikingDecoder(nn.Module):
    """ Population Coding Spike Decoder """
    def __init__(self, act_dim, pop_dim, output_activation=nn.Tanh):
        """
        :param act_dim: action dimension
        :param pop_dim:  population dimension
        :param output_activation: activation function added on output
        """
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.output_activation = output_activation()

    def forward(self, pop_act):
        """
        :param pop_act: output population activity
        :return: raw_act
        """
        pop_act = pop_act.view(-1, self.act_dim, self.pop_dim)
        raw_act = self.output_activation(self.decoder(pop_act).view(-1, self.act_dim))
        return raw_act


class PseudoSpikeRect(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()


class SpikeMLP(nn.Module):
    """ Spike MLP with Input and Output population neurons """
    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device):
        """
        :param in_pop_dim: input population dimension
        :param out_pop_dim: output population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        self.device = device
        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = out_pop_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoSpikeRect.apply
        # Define Layers (Hidden Layers + Output Population)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_pop_dim, hidden_sizes[0]).to(device)])
        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                self.hidden_layers.extend([nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer]).to(device)])
        self.out_pop_layer = nn.Linear(hidden_sizes[-1], out_pop_dim).to(device)

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """
        LIF Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, in_pop_spikes, batch_size):
        """
        :param in_pop_spikes: input population spikes
        :param batch_size: batch size
        :return: out_pop_act
        """
        # Define LIF Neuron states: Current, Voltage, and Spike
        hidden_states = []
        for layer in range(self.hidden_num):
            hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                  for _ in range(3)])
        out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device)
                          for _ in range(3)]
        out_pop_act = torch.zeros(batch_size, self.out_pop_dim, device=self.device)
        # Start Spike Timestep Iteration
        for step in range(self.spike_ts):
            in_pop_spike_t = in_pop_spikes[:, :, step]
            hidden_states[0][0], hidden_states[0][1], hidden_states[0][2] = self.neuron_model(
                self.hidden_layers[0], in_pop_spike_t,
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2]
            )
            if self.hidden_num > 1:
                for layer in range(1, self.hidden_num):
                    hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2] = self.neuron_model(
                        self.hidden_layers[layer], hidden_states[layer-1][2],
                        hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2]
                    )
            out_pop_states[0], out_pop_states[1], out_pop_states[2] = self.neuron_model(
                self.out_pop_layer, hidden_states[-1][2],
                out_pop_states[0], out_pop_states[1], out_pop_states[2]
            )
            out_pop_act += out_pop_states[2]
        out_pop_act = out_pop_act / self.spike_ts
        return out_pop_act


class SpikingActor(nn.Module):
    """ Population Coding Spike Actor with Fix Encoder """
    def __init__(self, obs_dim, act_dim, en_pop_dim, de_pop_dim, hidden_sizes,
                 mean_range, std, spike_ts, act_limit, device, use_poisson):
        """
        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param en_pop_dim: encoder population dimension
        :param de_pop_dim: decoder population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param mean_range: mean range for encoder
        :param std: std for encoder
        :param spike_ts: spike timesteps
        :param act_limit: action limit
        :param device: device
        :param use_poisson: if true use Poisson spikes for encoder
        """
        super().__init__()
        self.act_limit = act_limit
        if use_poisson:
            self.encoder = SpikingEncoderPoissonSpike(obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        else:
            self.encoder = SpikingEncoderRegularSpike(obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        self.snn = SpikeMLP(obs_dim*en_pop_dim, act_dim*de_pop_dim, hidden_sizes, spike_ts, device)
        self.decoder = SpikingDecoder(act_dim, de_pop_dim)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: action scale with action limit
        """
        in_pop_spikes = self.encoder(obs, batch_size)
        out_pop_activity = self.snn(in_pop_spikes, batch_size)
        action = self.act_limit * self.decoder(out_pop_activity)
        return action


class SpikingAC_Network(nn.Module):

    def __init__(self, observation_space, action_space,
                 encoder_pop_dim, decoder_pop_dim, mean_range, std, spike_ts, device, use_poisson,
                 hidden_sizes=(256, 256), activation=nn.ReLU):
        """
        :param observation_space: observation space from gym
        :param action_space: action space from gym
        :param encoder_pop_dim: encoder population dimension
        :param decoder_pop_dim: decoder population dimension
        :param mean_range: mean range for encoder
        :param std: std for encoder
        :param spike_ts: spike timesteps
        :param device: device
        :param use_poisson: if true use Poisson spikes for encoder
        :param hidden_sizes: list of hidden layer sizes
        :param activation: activation function for critic network
        """
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        # build policy and value functions
        self.spikeRL = SpikingActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_sizes,
                                    mean_range, std, spike_ts, act_limit, device, use_poisson)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, batch_size):
        with torch.no_grad():
            return self.spikeRL(obs, batch_size).to('cpu').numpy()

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.

    with Running Mean and Var from hill-a/stable-baselines
    """

    def __init__(self, obs_dim, act_dim, size, clip_limit, norm_update_every=1000):
        """
        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param size: buffer sizes
        :param clip_limit: limit for clip value
        :param norm_update_every: update freq
        """
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        # Running z-score normalization parameters
        self.clip_limit = clip_limit
        self.norm_update_every = norm_update_every
        self.norm_update_batch = np.zeros(combined_shape(norm_update_every, obs_dim), dtype=np.float32)
        self.norm_update_count = 0
        self.norm_total_count = np.finfo(np.float32).eps.item()
        self.mean, self.var = np.zeros(obs_dim, dtype=np.float32), np.ones(obs_dim, dtype=np.float32)

    def combined_shape(length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def store(self, obs, act, rew, next_obs, done):
        """
        Insert entry into memory
        :param obs: observation
        :param act: action
        :param rew: reward
        :param next_obs: observation after action
        :param done: if true then episode done
        """
        # Function to extract array from a tuple
        def extract_array(data):
            if isinstance(data, tuple):
                for item in data:
                    if isinstance(item, np.ndarray):
                        return item
                # Fallback in case no array is found (should not happen)
                raise ValueError("No array found in the tuple.")
            return data
    
        # Apply the extraction function to obs and next_obs
        obs = extract_array(obs)
        next_obs = extract_array(next_obs)
        # Ensure observations have expected dimensions before storing them
        if obs.shape != self.obs_buf[self.ptr].shape:
            raise ValueError(f"Obs shape mismatch. Expected: {self.obs_buf[self.ptr].shape}, Got: {obs.shape}")
        if next_obs.shape != self.obs2_buf[self.ptr].shape:
            raise ValueError(f"Next_obs shape mismatch. Expected: {self.obs2_buf[self.ptr].shape}, Got: {next_obs.shape}")

        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        # Update Mean and Variance
        # Have to at least update mean and variance once before training starts
        self.norm_update_batch[self.norm_update_count] = obs
        self.norm_update_count += 1
        if self.norm_update_count == self.norm_update_every:
            self.norm_update_count = 0
            batch_mean, batch_var = self.norm_update_batch.mean(axis=0), self.norm_update_batch.var(axis=0)
            tmp_total_count = self.norm_total_count + self.norm_update_every
            delta_mean = batch_mean - self.mean
            self.mean += delta_mean * (self.norm_update_every / tmp_total_count)
            m_a = self.var * self.norm_total_count
            m_b = batch_var * self.norm_update_every
            m_2 = m_a + m_b + np.square(delta_mean) * self.norm_total_count * self.norm_update_every / tmp_total_count
            self.var = m_2 / tmp_total_count
            self.norm_total_count = tmp_total_count

    def sample_batch(self, device, batch_size=32):
        """
        Sample batch from memory
        :param device: pytorch device
        :param batch_size: batch size
        :return: batch
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.normalize_obs(self.obs_buf[idxs]),
                     obs2=self.normalize_obs(self.obs2_buf[idxs]),
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}

    def normalize_obs(self, obs):
        """
        Do z-score normalization on observation
        :param obs: observation
        :return: norm_obs
        """
        # Function to extract array from a tuple
        def extract_array(data):
            if isinstance(data, tuple):
                for item in data:
                    if isinstance(item, np.ndarray):
                        return item
                # Fallback in case no array is found (should not happen)
                raise ValueError("No array found in the tuple.")
            return data
    
        # Apply the extraction function to obs and next_obs
        obs = extract_array(obs)

        eps = np.finfo(np.float32).eps.item()
        norm_obs = np.clip((obs - self.mean) / np.sqrt(self.var + eps),
                           -self.clip_limit, self.clip_limit)
        return norm_obs
        
def spikeRL_train(env_fn, actor_critic=SpikingAC_Network, ac_kwargs=dict(), seed=0,
              steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,
              polyak=0.995, learning_rate=1e-4, q_lr=1e-3, batch_size=100, start_steps=10000,
              update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
              noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
              save_freq=5, norm_clip_limit=3, norm_update=50, use_cuda=True):
    # Set device
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    # print(f'Total parameters in custom TD3: {count_parameters(ac)}')
    ac_targ = deepcopy(ac)
    ac.to(device)
    ac_targ.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
                                 clip_limit=norm_clip_limit, norm_update_every=norm_update)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        # print("\n[Debug] Data types in compute_loss_q:")
        # print("Observation dtype:", o.dtype)
        # print("Action dtype:", a.dtype)
        # print("Reward dtype:", r.dtype)
        # print("Next observation dtype:", o2.dtype)
        # print("Done dtype:", d.dtype)

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)
        # print("Q1 output dtype:", q1.dtype)
        # print("Q2 output dtype:", q2.dtype)

        # Bellman backup for Q functions
        with torch.no_grad():
            spikeRL_targ = ac_targ.spikeRL(o2, batch_size)

            # Target policy smoothing
            epsilon = torch.randn_like(spikeRL_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = spikeRL_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_spikeRL_targ = ac_targ.q1(o2, a2)
            q2_spikeRL_targ = ac_targ.q2(o2, a2)
            q_spikeRL_targ = torch.min(q1_spikeRL_targ, q2_spikeRL_targ)
            backup = r + gamma * (1 - d) * q_spikeRL_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Convert q1 and q2 to float32 before converting to numpy
        loss_info = dict(
            Q1Vals=q1.float().to('cpu').detach().numpy(),
            Q2Vals=q2.float().to('cpu').detach().numpy()
        )

        return loss_q, loss_info


    # Set up function for computing TD3 spikeRL loss
    def compute_loss_spikeRL(data):
        o = data['obs']
        # print("\n[Debug] Data types in compute_loss_spikeRL:")
        # print("Observation dtype:", o.dtype)
        action = ac.spikeRL(o, batch_size)
        # print("Action dtype from spikeRL:", action.dtype)
        q1_spikeRL = ac.q1(o, action)
        # print("q1_spikeRL dtype:", q1_spikeRL.dtype)
        return -q1_spikeRL.mean()


    # Set up optimizers for policy and q-function
    spikeRL_optimizer = optim.Adam(ac.spikeRL.parameters(), lr=learning_rate)
    q_optimizer = optim.Adam(q_params, lr=q_lr)

    scaler = torch.amp.GradScaler()

    def update(data, timer):
        # Before entering autocast
        # print("\n[Debug] Data types before autocast:")
        # for name, param in ac.q1.named_parameters():
        #     print(f"Data type of q1 parameter '{name}': {param.dtype}")
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_q, loss_info = compute_loss_q(data)
            # print("\n[Debug] Inside autocast for Q-networks:")
            # print("Data type of loss_q:", loss_q.dtype)
            # for name, param in ac.q1.named_parameters():
            #     print(f"Data type of q1 parameter '{name}': {param.dtype}")
            # for name, param in ac.q2.named_parameters():
            #     print(f"Data type of q2 parameter '{name}': {param.dtype}")
        scaler.scale(loss_q).backward()
        scaler.step(q_optimizer)
        scaler.update()

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for spikeRL.
            spikeRL_optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss_spikeRL = compute_loss_spikeRL(data)
                # print("\n[Debug] Inside autocast for spikeRL:")
                # print("Data type of loss_spikeRL:", loss_spikeRL.dtype)
                # for name, param in ac.spikeRL.named_parameters():
                #     print(f"Data type of spikeRL parameter '{name}': {param.dtype}")
            scaler.scale(loss_spikeRL).backward()
            scaler.step(spikeRL_optimizer)
            scaler.update()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)



    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32, device=device), 1)
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        ###
        # compuate the return mean test reward
        ###
        test_reward_sum = 0
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                action = get_action(replay_buffer.normalize_obs(o), 0)[0]
                o, r, d, _, info = test_env.step(action)
                ep_ret += r
                ep_len += 1
            test_reward_sum += ep_ret
        return test_reward_sum / num_test_episodes

    save_test_reward = []
    save_test_reward_steps = []

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(replay_buffer.normalize_obs(o), act_noise)[0]
        else:
            a = env.action_space.sample()
        # Step the env
        o2, r, d, _, info = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(device, batch_size)
                update(data=batch, timer=j)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            # Test the performance of the deterministic version of the agent.
            test_mean_reward = test_agent()
            save_test_reward.append(test_mean_reward)
            save_test_reward_steps.append(t + 1)
            print(" Steps: ", t + 1, " Mean Reward: ", test_mean_reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Walker2d-v4')
    parser.add_argument('--encoder_pop_dim', type=int, default=10)
    parser.add_argument('--decoder_pop_dim', type=int, default=10)
    parser.add_argument('--encoder_var', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    USE_POISSON = False
    if args.env == 'Hopper-v4' or args.env == 'Walker2d-v4':
        USE_POISSON = True
    AC_KWARGS = dict(hidden_sizes=[256, 256],
                     encoder_pop_dim=args.encoder_pop_dim,
                     decoder_pop_dim=args.decoder_pop_dim,
                     mean_range=(-3, 3),
                     std=math.sqrt(args.encoder_var),
                     spike_ts=5,
                     device=torch.device(device="cuda" if torch.cuda.is_available() else "cpu"),
                     use_poisson=USE_POISSON)

    # Measure Execution Time
    print(f"Running SpikeRL TD3 Algorithm on {args.env}")
    start_time = time.time()
    spikeRL_train(lambda : gym.make(args.env), actor_critic=SpikingAC_Network, ac_kwargs=AC_KWARGS,
                learning_rate=1e-4, gamma=0.99, seed=10, epochs=args.epochs, norm_clip_limit=3.0)
    print(f"SpikeRL Execution Time for {args.env}: {time.time() - start_time}")