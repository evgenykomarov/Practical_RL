# -*- coding: utf-8 -*-

import os
import sys
import math
import random
import time
from collections import defaultdict, deque
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.core import ObservationWrapper
from IPython.display import clear_output
from scipy.special import softmax
from tqdm import tqdm, trange

SEED = 42


def moving_average(x, span=100):
    return pd.DataFrame({"x": np.asarray(x)}).x.ewm(span=span).mean().values


def seed_everything(env, seed=None):
    if seed is None:
        seed = SEED
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    env.reset(seed=seed)


def visualize_agent(env, agent, max_steps=100, delay=0.1):
    """
    Visualize the agent's behavior in the environment.

    Args:
        env: The environment
        agent: The trained agent
        max_steps: Maximum number of steps to take
        delay: Time delay between steps for visualization
    """
    s, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Render the environment
        clear_output(True)
        plt.figure(figsize=(8, 6))
        plt.imshow(env.render())
        plt.title(f"Step: {step}, Total Reward: {total_reward:.2f}")
        plt.axis("off")
        plt.show()

        # Get action from the agent
        a = agent.get_best_action(s)  # Use best action for visualization

        # Take a step in the environment
        next_s, r, done, _, _ = env.step(a)

        # Update state and reward
        s = next_s
        total_reward += r

        # Add delay for better visualization
        time.sleep(delay)

        if done:
            # Show final state
            clear_output(True)
            plt.figure(figsize=(8, 6))
            plt.imshow(env.render())
            plt.title(f"Final State - Steps: {step + 1}, Total Reward: {total_reward:.2f}")
            plt.axis("off")
            plt.show()
            break


def benchmark_agents(
    exp_setups,
    num_episodes=1000,
    plot_every=100,
    t_max=10000,
    span=100,
    patch_every=None,
    patch_foo=None,
    num_seeds=3,
):
    all_rewards = {}
    envs = {exp_setup["name"]: exp_setup["env"]() for exp_setup in exp_setups}
    agents_buiders = {exp_setup["name"]: exp_setup["agent_builder"] for exp_setup in exp_setups}
    train_foo = {exp_setup["name"]: exp_setup["train_foo"] for exp_setup in exp_setups}

    for seed in range(num_seeds):
        SEED = seed + 42  # Using different seeds
        agents = {agent_name: agent() for agent_name, agent in agents_buiders.items()}

        # Create a separate environment for each agent using the env function
        for agent_name, agent in agents.items():
            agents[agent_name].env = envs[agent_name]

        seed_rewards = {agent_name: [] for agent_name in agents_buiders}

        # Seed each environment separately
        for agent_name in agents:
            seed_everything(envs[agent_name], seed=SEED)

        tbar = trange(num_episodes)
        tbar.set_description(f"Seed {seed + 1}/{num_seeds}")
        for i in tbar:
            for agent_name, agent in agents.items():
                seed_rewards[agent_name].append(train_foo[agent_name](envs[agent_name], agent))
            if i % 10 == 0:
                tbar.set_postfix({agent_name: seed_rewards[agent_name][-1] for agent_name in agents}, refresh=True)

        # Store rewards for this seed
        for agent_name, rewards_list in seed_rewards.items():
            if agent_name not in all_rewards:
                all_rewards[agent_name] = []
            all_rewards[agent_name].append(rewards_list)

        # Average rewards across seeds
        avg_rewards = {
            agent_name: np.mean(np.array(seed_results), axis=0) for agent_name, seed_results in all_rewards.items()
        }

        # Calculate standard deviation for confidence intervals
        std_rewards = {
            agent_name: np.std(np.array(seed_results), axis=0) for agent_name, seed_results in all_rewards.items()
        }

        # Plot average performance across seeds with confidence tubes
        clear_output(True)
        plt.figure(figsize=(10, 6))
        for agent_name, rewards_list in avg_rewards.items():
            mean_rewards = moving_average(rewards_list, span=span)
            std_rewards_smoothed = moving_average(std_rewards[agent_name], span=span)

            # Plot mean line
            plt.plot(mean_rewards, label=f"{agent_name} (avg of {num_seeds} seeds)")

            # Plot confidence tubes (mean ± std)
            plt.fill_between(
                range(len(mean_rewards)),
                mean_rewards - std_rewards_smoothed,
                mean_rewards + std_rewards_smoothed,
                alpha=0.2,
            )

            # Draw solid contour lines for the confidence tube borders
            plt.plot(range(len(mean_rewards)), mean_rewards - std_rewards_smoothed, "--", color="gray", alpha=0.7)
            plt.plot(range(len(mean_rewards)), mean_rewards + std_rewards_smoothed, "--", color="gray", alpha=0.7)

        plt.title(
            f"{envs[list(envs.keys())[0]].spec.id} - Average performance across {num_seeds} seeds with confidence intervals"
        )
        plt.legend()
        plt.show()

    return avg_rewards


class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, env):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.env = env
        self._qvalues = defaultdict(lambda: defaultdict(int))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_legal_actions(self, _state):
        return list(range(self.env.action_space.n))

    def get_qvalue(self, state, action):
        """Returns Q(state,action)"""
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """Sets the Qvalue for [state,action] to the given value"""
        self._qvalues[state][action] = value

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if not possible_actions:
            return 0.0

        value = max([self.get_qvalue(state, action) for action in possible_actions])

        return value

    def update(self, state, action, reward, next_state, *args, **kwargs):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        new_q_value = (1.0 - learning_rate) * self.get_qvalue(state, action) + learning_rate * (reward + gamma * self.get_value(next_state))

        self.set_qvalue(state, action, new_q_value)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        best_action = None
        best_qvalue = float('-inf')
        for action in possible_actions:
            qvalue = self.get_qvalue(state, action)
            if qvalue < best_qvalue:
                continue
            best_action = action
            best_qvalue = qvalue

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)
        action = None

        # If there are no legal actions, return None
        if not possible_actions:
            return None

        # agent parameters:
        epsilon = self.epsilon
        is_exploration = self.env.np_random.random() <= epsilon

        if is_exploration:
            chosen_action = np.random.choice(possible_actions)
            return chosen_action

        chosen_action = self.get_best_action(state)

        return chosen_action


def play_and_train(env, agent: QLearningAgent, t_max=10**4):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    state, _ = env.reset()

    for t in range(t_max):
        action = agent.get_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated

        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        if done:
            break

    return total_reward


def make_env():
    return gym.make("CartPole-v0", render_mode="rgb_array").env  # .env unwraps the TimeLimit wrapper


def visualize_cartpole_observation_distribution(seen_observations):
    seen_observations = np.array(seen_observations)

    # The meaning of the observations is documented in
    # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

    # Get the number of dimensions from the state
    n_dims = seen_observations.shape[1]

    f, axarr = plt.subplots(1, n_dims, figsize=(16, 4), sharey=True)
    titles = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Velocity At Tip"]

    for i in range(n_dims):
        ax = axarr[i]
        ax.hist(seen_observations[:, i], bins=20)
        ax.set_title(titles[i])
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(min(xmin, -xmax), max(-xmin, xmax))
        ax.grid()
    f.tight_layout()


def gather_samples(env, max_steps=100000):
    seen_observations = []
    total_steps = 0

    while total_steps < max_steps:
        s, _ = env.reset()
        seen_observations.append(s)
        done = False

        while not done and total_steps < max_steps:
            s, r, done, _, _ = env.step(env.action_space.sample())
            seen_observations.append(s)
            total_steps += 1

        if total_steps >= max_steps:
            break

    return seen_observations


class Discretizer(ObservationWrapper):
    def __init__(self, env, n_digits):
        super().__init__(env)
        self.n_digits = n_digits

    def observation(self, state):
        state = np.round(state, self.n_digits)
        return tuple(state)  # tuple to make it hashable


class DiscretizerFractional(ObservationWrapper):
    def __init__(self, env, precision):
        super().__init__(env)
        self.precision = precision

    def observation(self, state):
        state = np.round(state / self.precision, 0) * self.precision
        return tuple(state)  # tuple to make it hashable


def train_env_with_epsilon_decay(discretizer, precision_or_ndigits=1, alpha=0.5, discount=0.99, init_epsilon=0.25, target_epsilon=0.25, n_warmup=5000, total_steps=10000):
    env = discretizer(make_env(), precision_or_ndigits)
    agent = QLearningAgent(alpha=alpha, epsilon=init_epsilon, discount=discount, env=env)

    rewards, epsilons = [], []
    seed_everything(env)

    for i in range(total_steps):
        reward = play_and_train(env, agent)
        rewards.append(reward)

        # (y - y0) = k * (x - x0)
        # y = y0 + (y1 - y0) / (x1 - x0) * (x - x0)
        agent.epsilon = max(target_epsilon, target_epsilon + (target_epsilon - init_epsilon) / (n_warmup - 0) * (i - n_warmup))
        epsilons.append(agent.epsilon)

        if (i + 1) % 1000 == 0:
            rewards_ewma = moving_average(rewards)

            clear_output(True)
            plt.plot(rewards, label="rewards")
            plt.plot(rewards_ewma, label="rewards ewma@100")
            plt.legend()
            plt.grid()
            plt.title("eps = {:e}, rewards ewma@100 = {:.1f}".format(agent.epsilon, rewards_ewma[-1]))
            plt.show()
    return agent, rewards, epsilons


class EVSarsaAgent(QLearningAgent):
    """
    An agent that changes some of q-learning functions to implement Expected Value SARSA.
    Note: this demo assumes that your implementation of QLearningAgent.update uses get_value(next_state).
    If it doesn't, please add
        def update(self, state, action, reward, next_state):
            and implement it for Expected Value SARSA's V(s')
    """

    def get_value(self, state):
        """
        Returns Vpi for current state under epsilon-greedy policy:
          V_{pi}(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}

        Hint: all other methods from QLearningAgent are still accessible.
        """
        epsilon = self.epsilon
        possible_actions = self.get_legal_actions(state)
        n_actions = len(possible_actions)

        # If there are no legal actions, return 0.0
        if not possible_actions:
            return 0.0

        # (1 - e) * Q_best + (e) / n * sum(Q_random_action)
        best_action = self.get_best_action(state)
        avg_q_random_action = sum([self.get_qvalue(state, action) for action in possible_actions]) / n_actions
        state_value = (1.0 - epsilon) * self.get_qvalue(state, best_action) + epsilon * avg_q_random_action

        return state_value


def get_ascii_policy(agent):
    """Returns CliffWalkingEnv policy with arrows as a string. Hard-coded."""

    env = gym.make("CliffWalking-v1", render_mode="ansi")
    env.reset()
    grid = [x.split("  ") for x in env.render().split("\n")[:4]]

    n_rows, n_cols = 4, 12
    start_state_index = 36
    actions = "^>v<"

    policy_str = ""
    for yi in range(n_rows):
        for xi in range(n_cols):
            if grid[yi][xi] == "C":
                policy_str += " C "
            elif (yi * n_cols + xi) == start_state_index:
                policy_str += " X "
            elif (yi * n_cols + xi) == n_rows * n_cols - 1:
                policy_str += " T "
            else:
                policy_str += " %s " % actions[agent.get_best_action(yi * n_cols + xi)]
        policy_str += "\n"

    return policy_str


def draw_policy(agent):
    """Prints CliffWalkingEnv policy with arrows."""
    print(get_ascii_policy(agent))


class SoftmaxEVSarsaAgent(EVSarsaAgent):
    def __init__(self, alpha, tau, discount, env):
        super().__init__(alpha, None, discount, env)
        assert tau > 0
        self.tau = tau

    def _get_softmax_prob(self, state):
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if not possible_actions:
            return None, None

        qvalues = np.array([self.get_qvalue(state, action) for action in possible_actions])
        prob = softmax(qvalues / self.tau)
        return prob, qvalues

    def get_value(self, state):
        """
        Returns V_{pi} for current state under softmax policy:
          V_{pi}(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}

        Hint: all other methods from QLearningAgent are still accessible.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if not possible_actions:
            return 0.0

        prob, qvalues = self._get_softmax_prob(state)
        value = prob @ qvalues

        return value

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        We should take a random action with probability equaled softmax of q values.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if not possible_actions:
            return None

        prob, _ = self._get_softmax_prob(state)
        action = self.env.np_random.choice(possible_actions, p=prob)

        return action


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._storage = deque(maxlen=size)
        self._maxsize = size

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        """
        Make sure, _storage will not exceed _maxsize.
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        """
        self._storage.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        state_batch: np.array
            batch of states
        action_batch: np.array
            batch of actions executed given state_batch
        reward_batch: np.array
            rewards received as results of executing action_batch
        next_state_batch: np.array
            next set of states seen after executing action_batch
        done_mask: np.array
            done_mask[i] = 1 if executing action_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """

        batch_indices = np.random.choice(range(len(self._storage)), size=batch_size, replace=True)
        state_batch, action_batch, reward_batch, next_state_batch, done_mask = zip(*[self._storage[i] for i in batch_indices])

        # collect <s,a,r,s',done> for each index
        return (
            np.array(state_batch),
            np.array(action_batch),
            np.array(reward_batch),
            np.array(next_state_batch),
            np.array(done_mask),
        )


class ReplayBufferNparray(object):
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._state = np.zeros(size, dtype=str)
        self._action = np.zeros(size, dtype=str)
        self._reward = np.zeros(size, dtype=float)
        self._next_state = np.zeros(size, dtype=str)
        self._done = np.zeros(size, dtype=bool)
        self._idx = 0
        self._curr_size = 0
        self._maxsize = size

    def __len__(self):
        return self._curr_size

    def add(self, state, action, reward, next_state, done):
        """
        Make sure, _storage will not exceed _maxsize.
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        """
        i = self._idx % self._maxsize
        self._state[i] = state
        self._action[i] = action
        self._reward[i] = reward
        self._next_state[i] = next_state
        self._done[i] = done

        self._idx = (i + 1) % self._maxsize
        self._curr_size = min(self._maxsize, self._curr_size + 1)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        state_batch: np.array
            batch of states
        action_batch: np.array
            batch of actions executed given state_batch
        reward_batch: np.array
            rewards received as results of executing action_batch
        next_state_batch: np.array
            next set of states seen after executing action_batch
        done_batch: np.array
            done_batch[i] = 1 if executing action_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """

        if self._curr_size < self._maxsize:
            state_batch = np.random.choice(self._state[:self._curr_size], size=batch_size, replace=True)
            action_batch = np.random.choice(self._action[:self._curr_size], size=batch_size, replace=True)
            reward_batch = np.random.choice(self._reward[:self._curr_size], size=batch_size, replace=True)
            next_state_batch = np.random.choice(self._next_state[:self._curr_size], size=batch_size, replace=True)
            done_batch = np.random.choice(self._done[:self._curr_size], size=batch_size, replace=True)
        else:
            state_batch = np.random.choice(self._state, size=batch_size, replace=True)
            action_batch = np.random.choice(self._action, size=batch_size, replace=True)
            reward_batch = np.random.choice(self._reward, size=batch_size, replace=True)
            next_state_batch = np.random.choice(self._next_state, size=batch_size, replace=True)
            done_batch = np.random.choice(self._done, size=batch_size, replace=True)

        # collect <s,a,r,s',done> for each index
        return (
            np.array(state_batch),
            np.array(action_batch),
            np.array(reward_batch),
            np.array(next_state_batch),
            np.array(done_batch),
        )


def obj2arrays(obj):
    for x in obj:
        yield np.array([x])


def obj2sampled(obj):
    return tuple(obj2arrays(obj))


def play_and_train_with_replay(env, agent, replay=None, t_max=10**4, replay_batch_size=32):
    """
    This function should
    - run a full game, actions given by agent.get_action(s)
    - train agent using agent.update(...) whenever possible
    - return total reward
    :param replay: ReplayBuffer where agent can store and sample (s,a,r,s',done) tuples.
        If None, do not use experience replay
    """
    total_reward = 0.0
    state, _ = env.reset()

    for t in range(t_max):
        action = agent.get_action(state)

        next_state, reward, done, trunc, _ = env.step(action)

        agent.update(state, action, reward, next_state)

        if replay is not None:
            replay.add(state, action, reward, next_state, done)

            state_batch, action_batch, reward_batch, next_state_batch, done_mask = replay.sample(replay_batch_size)
            for i in range(replay_batch_size):
                agent.update(state_batch[i], action_batch[i], reward_batch[i], next_state_batch[i])

        state = next_state
        total_reward += reward
        if done:
            break

    return total_reward


class NstepAgentABC(QLearningAgent):
    """
    N-step Sarsa (Sutton & Barto, Chapter 7).

    Unlike NStepEVSarsaAgent (which bootstraps with the expected value V(S_{τ+n})
    under π), this uses the actual sampled action's Q-value Q(S_{τ+n}, A_{τ+n})
    as the bootstrap target — the defining difference of Sarsa vs EV-Sarsa.

    The internal buffer stores (S_t, A_t, R_{t+1}) for each step t, which lets
    _update_tau() compute the n-step return G_{τ:τ+n} and update Q(S_τ, A_τ).
    Call reset() at the start of each episode (play_and_train_nstep does this).
    """

    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.reset()

    def reset(self):
        self._t = 0
        self._T = float('inf')
        self._buf = []

    def update(self, state, action, reward, next_state, done=False, *args, **kwargs):
        self._buf.append((state, action, reward))  # (S_t, A_t, R_{t + 1})

        if done:
            self._T = self._t + 1

        # intended deviation from the original algo from Sutton: we want to store the
        # current (S_t, A_t, R_{t + 1}) and not storing the future one (next_state, ...) = (S_{t + 1}, ...)
        # this would entail selecting next_action which we would rather delegate to the outer loop
        # of q-learning. Thus, we delay calculation from tau by 1 step, and set tau := t - n rather than tau := t - n + 1
        tau = self._t - self.n

        if tau >= 0:
            self._update_tau(tau)

        self._t += 1
        if not done:
            return

        # Flush remaining tau values once the episode ends
        for flush_tau in range(max(tau + 1, 0), int(self._T)):
            self._update_tau(flush_tau)

    def _update_tau(self, tau):
        """Compute G_{τ:τ+n} and apply the Sarsa update to Q(S_τ, A_τ)."""
        gamma = self.discount
        n = self.n
        end = min(self._T, tau + n)
        G = sum(gamma**j * self._buf[tau + j][2] for j in range(end - tau))
        if tau + n < self._T:
            state_tau_plus_n, action_tau_plus_n, _ = self._buf[tau + n]
            G += gamma**n * self._gamma_estimator(state_tau_plus_n, action_tau_plus_n)
        state_tau, action_tau, _ = self._buf[tau]
        qvalue = self.get_qvalue(state_tau, action_tau)
        self.set_qvalue(state_tau, action_tau, qvalue + self.alpha * (G - qvalue))


class NstepSarsaAgent(NstepAgentABC):
    """
    N-step Sarsa (Sutton & Barto, Chapter 7).

    Unlike NStepEVSarsaAgent (which bootstraps with the expected value V(S_{τ+n})
    under π), this uses the actual sampled action's Q-value Q(S_{τ+n}, A_{τ+n})
    as the bootstrap target — the defining difference of Sarsa vs EV-Sarsa.

    The internal buffer stores (S_t, A_t, R_{t+1}) for each step t, which lets
    _update_tau() compute the n-step return G_{τ:τ+n} and update Q(S_τ, A_τ).
    Call reset() at the start of each episode (play_and_train_nstep does this).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _gamma_estimator(self, state_tau_plus_n, action_tau_plus_n):
        return self.get_qvalue(state_tau_plus_n, action_tau_plus_n)


class NStepEVSarsaAgent(NstepSarsaAgent):
    """
    N-step EV-Sarsa (Sutton & Barto, Chapter 7).

    Unlike NStepEVSarsaAgent (which bootstraps with the expected value V(S_{τ+n})
    under π), this uses the actual sampled action's Q-value Q(S_{τ+n}, A_{τ+n})
    as the bootstrap target — the defining difference of Sarsa vs EV-Sarsa.

    The internal buffer stores (S_t, A_t, R_{t+1}) for each step t, which lets
    _update_tau() compute the n-step return G_{τ:τ+n} and update Q(S_τ, A_τ).
    Call reset() at the start of each episode (play_and_train_nstep does this).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _gamma_estimator(self, state_tau_plus_n, _action_tau_plus_n):
        return self.get_value(state_tau_plus_n)


def play_and_train_nstep(env, agent, t_max=10**4):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    state, _ = env.reset()
    agent.reset()

    for t in range(t_max):
        # get agent to pick action given state s
        action = agent.get_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated

        # train (update) agent for state s
        agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        if done:
            agent.reset()
            break

    return total_reward


def evaluate_nstep_evsarsa(env_builder, n_episodes=1000, t_max=10000, n_seeds=3, span=10):
    params = {"alpha": 0.1, "epsilon": 0.1, "discount": 0.99, "env": env_builder()}

    exp_setups = [
        {
            "name": f"nstep_evsarsa_{n_step}",
            "agent_builder": partial(NStepEVSarsaAgent, n=n_step, **params),
            "env": env_builder,
            "train_foo": play_and_train_nstep,
        }
        for n_step in [1, 2, 4]
    ]
    _ = benchmark_agents(
        exp_setups, num_episodes=n_episodes, t_max=t_max, plot_every=1000, span=span, num_seeds=n_seeds
    )


def evaluate_nstep_sarsa_vs_evsarsa(env_builder, n_episodes=1000, t_max=10000, n_seeds=3, span=10):
    params = {"alpha": 0.1, "epsilon": 0.1, "discount": 0.99, "env": env_builder()}
    n_step = 4
    def gen_name(agent):
        if agent == NStepEVSarsaAgent:
            return f"nstep_evsarsa_{n_step}"
        if agent == NstepSarsaAgent:
            return f"nstep_sarsa_{n_step}"
        raise Exception

    exp_setups = [
        {
            "name": gen_name(agent),
            "agent_builder": partial(agent, n=n_step, **params),
            "env": env_builder,
            "train_foo": play_and_train_nstep,
        }
        for agent in [NStepEVSarsaAgent, NstepSarsaAgent]
    ]
    _ = benchmark_agents(
        exp_setups, num_episodes=n_episodes, t_max=t_max, plot_every=1000, span=span, num_seeds=n_seeds
    )


class QLambdaAgent(QLearningAgent):
    def __init__(self, alpha, epsilon, discount, lambda_factor, env):
        assert 0 <= lambda_factor <= 1, "lambda must be between 0 and 1"
        super().__init__(alpha, epsilon, discount, env)
        self.eligibility_trace = defaultdict(float)
        self.lambda_factor = lambda_factor

    def reset(self):
        self.eligibility_trace.clear()

    def _decay(self, eps=1e-6):
        decay = self.lambda_factor * self.discount
        for s, a in list(self.eligibility_trace.keys()):
            self.eligibility_trace[(s, a)] *= decay
            # Remove traces close to zero
            if abs(self.eligibility_trace[(s, a)]) < eps:
                del self.eligibility_trace[(s, a)]

    def update(self, state, action, reward, next_state, _done):
        """
        Implements Q(λ) update rule:
        - Updates Q-values using eligibility traces.
        - Adjusts traces based on whether the selected action matches the best action.
        """
        lambda_factor = self.lambda_factor
        # Update eligibility traces
        if action == self.get_best_action(state):
            self._decay()
        else:
            self.reset()

        # Compute TD error with check for terminal state
        next_value = self.get_value(next_state) if next_state is not None else 0
        delta = reward + self.discount * next_value - self.get_qvalue(state, action)

        # Update eligibility trace for (state, action)
        self.eligibility_trace[(state, action)] += 1

        # Iterate over all stored (s, a) pairs in eligibility trace
        for s, a in self.eligibility_trace.keys():
            # 𝑄(𝑠,𝑎)←𝑄(𝑠,𝑎)+𝛼𝛿𝑒(𝑠,𝑎)
            new_qvalue = self.get_qvalue(s, a) + self.alpha * delta * self.eligibility_trace[(s, a)]
            self.set_qvalue(s, a, new_qvalue)


def evaluate_lambda_qlearning(env_builder, n_episodes=1000, t_max=10000, n_seeds=3, span=10):
    params = {"alpha": 0.1, "epsilon": 0.1, "discount": 0.99, "env": env_builder()}
    exp_setups = []
    exp_setups.extend(
        [
            {
                "name": f"lambda_qlearning_{lambda_factor}",
                "agent_builder": partial(QLambdaAgent, lambda_factor=lambda_factor, **params),
                "env": env_builder,
                "train_foo": play_and_train_nstep,
            }
            for lambda_factor in [0.0, 0.2, 0.5, 0.95]
        ]
    )
    _ = benchmark_agents(
        exp_setups, num_episodes=n_episodes, t_max=t_max, plot_every=1000, span=span, num_seeds=n_seeds
    )


def play_and_train_retrace(env, agent, replay, t_max, n_train_steps, trajectory_len):
    total_reward = 0.0
    s, _ = env.reset()
    replay._episodes.append([])

    for t in range(t_max):
        # Get action from agent
        a = agent.get_action(s)
        prob = agent.get_action_probability(s, a)

        next_s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        replay._episodes[-1].append((s, a, r, next_s, done, prob))

        # Update state and total reward
        s = next_s
        total_reward += r

        if replay is not None and len(replay) > 0:
            for _ in range(n_train_steps):
                trajectory = replay.sample(max_trajectory_len=trajectory_len)  # whole trajectory
                agent.retrace_update(trajectory)

        if done:
            break

    return total_reward


class SequenceBuffer:
    def __init__(self, episode_capacity):
        self._episodes = deque(maxlen=episode_capacity)

    def __len__(self):
        return len(self._episodes)

    def add_sequence(self, sequence):
        self._episodes.append(sequence)

    def sample(self, max_trajectory_len=None):
        seq_idx = np.random.randint(0, len(self))
        sequence = self._episodes[seq_idx]

        if max_trajectory_len is None:
            return sequence

        start_idx = np.random.randint(0, max(len(sequence) - max_trajectory_len, 1))
        return sequence[start_idx : start_idx + max_trajectory_len]


class RetraceAgent(QLearningAgent):
    def __init__(self, alpha, epsilon, discount, env, lambda_factor=0.9):
        super().__init__(alpha, epsilon, discount, env)
        self.lambda_factor = lambda_factor

    def get_action_probability(self, state, action):
        n_actions = self.env.action_space.n
        best_action_prob = 1.0 - self.epsilon + self.epsilon / n_actions
        if action == self.get_best_action(state):
            return best_action_prob
        return self.epsilon / n_actions

    def retrace_update(self, trajectory):
        last_state, last_action, last_reward, last_next_state, last_done, last_behavior_prob = trajectory[-1]

        # Q^{ret}_{t} = r_t + gamma * ( Q_{t + 1} + \alpha * \lambda * c_{t + 1} * ( Q^{ret}_{t + 1} - Q_{t + 1} ) )
        G = last_reward

        for state, action, reward, next_state, done, behavior_prob in reversed(trajectory[:-1]):
            # 𝑐𝑡+1=min(1,𝜋(𝑎𝑡+1|𝑠𝑡+1)𝜇(𝑎𝑡+1|𝑠𝑡+1))
            # c = min(1, pi_target / pi_behavior)
            c = min(1.0, self.get_action_probability(state, action) / behavior_prob)

            if done:
                expected_Q = 0
            else:
                probs = np.array([self.get_action_probability(next_state, action) for action in self.get_legal_actions(next_state)])
                expected_Q = probs @ np.array([self.get_qvalue(next_state, action) for action in self.get_legal_actions(next_state)])

            G = reward + self.discount * (expected_Q + self.alpha * self.lambda_factor * c * (G - expected_Q))

            new_qvalue = (1 - self.alpha) * self.get_qvalue(state, action) + self.alpha * G

            self.set_qvalue(state, action, new_qvalue)
