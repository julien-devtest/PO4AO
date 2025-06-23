import torch
import numpy as np 
# nn is not used directly by EfficientExperienceReplay or ReplaySample
# from torch import nn

class EfficientExperienceReplay():
    """
    A replay buffer for storing and sampling experiences in reinforcement learning.

    This buffer stores transitions (state, action, next_state) as PyTorch tensors
    on a CUDA device. It uses a circular buffer mechanism and supports sampling
    both random individual transitions and contiguous sequences of transitions,
    which is crucial for training models that rely on a history of states/actions.
    An optional 'warmup_memory' section can be preserved from overwriting.

    Attributes:
        max_size (int): Maximum number of transitions the buffer can hold.
        states (torch.Tensor): Tensor storing all states. Shape: (max_size, *state_shape).
        next_states (torch.Tensor): Tensor storing all next states. Shape: (max_size, *state_shape).
        actions (torch.Tensor): Tensor storing all actions. Shape: (max_size, *action_shape).
        len (int): Current number of transitions stored in the buffer.
        index_write (int): The next index at which a new transition will be written.
                           Used for circular buffer logic.
        warmup_memory (int): The number of initial transitions that are preserved
                             from being overwritten when the buffer becomes full and wraps around.
    """

    def __init__(self, state_shape, action_shape, max_size=100000, warmup_memory = 0):
        """
        Initializes the EfficientExperienceReplay buffer.

        Args:
            state_shape (tuple): The shape of a single state observation (e.g., (channels, height, width)).
            action_shape (tuple): The shape of a single action.
            max_size (int, optional): The maximum number of transitions that can be stored.
                                      Defaults to 100000.
            warmup_memory (int, optional): The number of initial memory slots to reserve,
                                           preventing them from being overwritten immediately when
                                           the buffer wraps around. Defaults to 0.
        """
        self.max_size = max_size

        self.states = torch.empty(max_size, *state_shape).to("cuda:0")#.share_memory_()
        self.next_states = torch.empty(max_size, *state_shape).to("cuda:0")#.share_memory_()
        self.actions = torch.empty(max_size, *action_shape).to("cuda:0")#.share_memory_()

        self.len   = 0
        self.index_write = 0
        self.warmup_memory = warmup_memory

    # The 'add' and '__add__' methods appear unused in the current po4ao.py context.
    # If they were to be used, 'replay.rewards' in 'add' would need to be addressed
    # as ReplaySample does not have a rewards attribute.
    # def add(self, replay):
    #     cur_len = self.len
    #     new_len = self.len + len(replay)
        
    #     if isinstance(replay, EfficientExperienceReplay):
    #         # Assuming ReplaySample should not have rewards, or it needs to be handled
    #         replay = ReplaySample(replay.states[:len(replay)], replay.actions[:len(replay)], replay.next_states[:len(replay)])
        
    #     self.states[cur_len:new_len] = replay.state()
    #     self.next_states[cur_len:new_len] = replay.next_state()
    #     self.actions[cur_len:new_len] = replay.action()

    #     self.len = new_len

    # def __add__(self, replay):
    #     self.add(replay)
    #     return self
    
    def append(self, obs, action, next_obs):
        """
        Adds a single transition to the replay buffer.

        The transition consists of an observation, the action taken, and the resulting
        next observation. Implements circular buffer logic.

        Args:
            obs (torch.Tensor): The observation (state).
            action (torch.Tensor): The action taken.
            next_obs (torch.Tensor): The next observation (state) received after taking the action.

        Raises:
            TypeError: If 'obs' is a NumPy array instead of a PyTorch tensor.
        """
        if isinstance(obs, np.ndarray):
            raise TypeError('Input observations should be torch.Tensor, not np.ndarray.')

        self.states[self.index_write] = obs
        self.next_states[self.index_write] = next_obs
        self.actions[self.index_write] = action

        self.index_write += 1

        if self.len < self.max_size:
            self.len += 1

        if self.index_write == self.max_size:
            print('Experience Replay Full')
            self.index_write = self.warmup_memory


    def sample_contiguous(self, horizon, max_ts, batch_size=32):
        """
        Samples a batch of contiguous sequences of transitions.

        This method is designed for training models that require a history of
        states and actions. It assumes that data within the buffer is organized
        contiguously by episode. Each sampled sequence will contain `horizon + 1` transitions.

        Args:
            horizon (int): The number of transitions in the history part of a sequence.
                           The total length of each sampled sequence will be `horizon + 1`.
            max_ts (int): The maximum number of timesteps in an episode (episode_length).
                          Used to determine episode boundaries for sampling valid sequences.
            batch_size (int, optional): The number of contiguous sequences to sample.
                                        Defaults to 32.

        Returns:
            ReplaySample: A ReplaySample object containing the batch of sampled
                          contiguous sequences (states, actions, next_states).
                          Each tensor in ReplaySample will have its first dimension
                          as `batch_size * (horizon + 1)`.
        """
        # Ensure there's enough data to sample even one full episode sequence
        if len(self) < max_ts or len(self) < (horizon + 1) : # Or handle more gracefully
             raise ValueError("Not enough data in replay buffer to sample contiguous sequences.")
        if len(self) // max_ts == 0 and max_ts > len(self) : # Avoid division by zero if less than one episode stored
             raise ValueError("Not enough data for even one full episode to sample from.")


        inds = torch.randint(0, max_ts - (horizon + 1), size=(batch_size, )) # Start index within an episode
        
        # Determine which episode to draw from.
        # Ensure num_episodes_available is at least 1 if len(self) >= max_ts
        num_episodes_available = max(1, len(self) // max_ts)
        episode_indices = torch.randint(0, num_episodes_available, size=(batch_size, ))

        inds += episode_indices * max_ts # Offset by episode start

        # Ensure indices are within the current data length 'self.len'
        # This check is more of a safeguard; ideally, sampling logic should prevent out-of-bounds.
        inds = torch.clamp(inds, 0, self.len - (horizon + 1) -1)


        indices = torch.cat([torch.arange(ind, ind + horizon + 1) for ind in inds])
        # TODO check correct, original comment:
        #indices = torch.from_numpy(vrange(inds.numpy(), np.ones_like(inds) * horizon + 1))
        
        return ReplaySample(self.states[indices], self.actions[indices], self.next_states[indices])

    
    def next_state(self):
        """Returns all 'next_states' currently stored in the buffer."""
        return self.next_states[:self.len]
    
    def state(self):
        return self.states[:self.len]

    def action(self):
        return self.actions[:self.len]

    def __len__(self):
        return self.len
        
    def set_len(self,index):
        self.len = index
        self.index_write = index

    def sample(self, size=512):
        inds = torch.randperm(self.len)[:size]
        return ReplaySample(self.states[inds], self.actions[inds], self.next_states[inds])  

    def clear(self):
        self.len = 0

class ReplaySample():
    def __init__(self, states, actions, next_states):
        self.states = states
        self.next_states = next_states
        self.actions = actions

    def state(self):
        return self.states
    
    def prev_action(self):
        return self.prev_actions

    def next_state(self):
        return self.next_states 

    def action(self):
        return self.actions

    def __len__(self):
        return len(self.states)

    def to(self, device):
        self.states = self.states.to(device)
        self.next_states = self.next_states.to(device)
        self.actions = self.actions.to(device)
        return self

import contextlib
import os

@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:


    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()