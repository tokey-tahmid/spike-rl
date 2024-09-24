import os
import numpy as np
import torch
import torch.nn as nn
import gym
import math
import argparse
import pickle
# from tqdm import tqdm
# import time
# from codecarbon import EmissionsTracker
"""
How to run:
requirements after creating pyframework environment: torch, gym, mujoco==2.3.3, numpy==1.24.0
python spikeRL_inference.py --env Humanoid-v4
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
                 hidden_sizes=(4000, 4000), activation=nn.ReLU):
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

    def sample_batch(self, device, batch_size=4000):
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


def spikeRL_inference(spikeRL_model_file, mean_var_file, env_fn,
                       encoder_pop_dim, decoder_pop_dim, mean_range, std, spike_ts,
                       hidden_size=(4000, 4000), norm_clip_limit=3):
    """
    Test and render Mojuco Tasks

    :param spikeRL_model_file: file dir for spikeRL model
    :param mean_var_file: file dir for mean and var of replay buffer
    :param env_fn: function of create environment
    :param encoder_pop_dim: encoder population dimension
    :param decoder_pop_dim: decoder population dimension
    :param mean_range: mean range for encoder
    :param std: std for encoder
    :param spike_ts: spike timesteps
    :param hidden_size: list of hidden layer sizes
    :param norm_clip_limit: clip limit
    """
    # Set device
    device = torch.device("cpu")
    # Set environment
    test_env = env_fn()
    obs_dim = test_env.observation_space.shape[0]
    act_dim = test_env.action_space.shape[0]
    act_limit = test_env.action_space.high[0]

    # Replay buffer for running z-score norm
    b_mean_var = pickle.load(open(mean_var_file, "rb"))
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=1,
                                 clip_limit=norm_clip_limit, norm_update_every=1)
    replay_buffer.mean = b_mean_var[0]
    replay_buffer.var = b_mean_var[1]

    # spikeRL
    spikeRL = SpikingActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_size,
                           mean_range, std, spike_ts, act_limit, device, False)
    spikeRL.load_state_dict(torch.load(spikeRL_model_file))

    def get_action(o):
        a = spikeRL(torch.as_tensor(o, dtype=torch.float32, device=device), 1).numpy()
        return np.clip(a, -act_limit, act_limit)

    # Start testing
    o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
    while not (d or (ep_len == 1000)):
        test_env.render()
        with torch.no_grad():
            o, r, d, _, info = test_env.step(get_action(replay_buffer.normalize_obs(o))[0])
        ep_ret += r
        ep_len += 1
    print("Reward: ", ep_ret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file_dir', type=str, default='./models')
    parser.add_argument('--buffer_file_dir', type=str, default='./models')
    parser.add_argument('--env', type=str, default='Walker2d-v4')
    parser.add_argument('--encoder_pop_dim', type=int, default=100)
    parser.add_argument('--decoder_pop_dim', type=int, default=100)
    parser.add_argument('--encoder_var', type=float, default=0.15)
    args = parser.parse_args()
    mean_range = (-3, 3)
    std = math.sqrt(args.encoder_var)
    spike_ts = 5
    model_file = os.path.join(args.model_file_dir, f'{args.env}_e1.pt')
    buffer_file = os.path.join(args.buffer_file_dir, f'{args.env}_e1_mean_var.p')

    spikeRL_inference(model_file, buffer_file,
                       lambda : gym.make(args.env, render_mode="rgb_array"), 
                       args.encoder_pop_dim, 
                       args.decoder_pop_dim, 
                       mean_range, std, spike_ts)
