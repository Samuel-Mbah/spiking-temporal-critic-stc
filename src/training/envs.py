"""Environment construction, vectorisation, and observation wrappers.

Supports frame-stacking, partial-observation masking, and running-statistics
normalisation via ``VecNormalize``.
"""
import cv2
import logging
import random
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from collections import deque
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorWrapper
from typing import Tuple, Optional

logger_mod = logging.getLogger(__name__)
_LOGGED_PARTIAL_OBS = False

def set_global_seeds(seed: int, deterministic_torch: bool = True, cudnn_benchmark: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
            
class PartialObs(gym.ObservationWrapper):
    def __init__(self, env, indices=None, mask=None):
        super().__init__(env)
        if indices is None and mask is None:
            raise ValueError("Provide indices or mask for PartialObs.")
        obs_space = env.observation_space
        if mask is not None:
            indices = np.where(np.asarray(mask, dtype=bool))[0].tolist()
        self.indices = np.asarray(indices, dtype=int)
        low = obs_space.low[self.indices]
        high = obs_space.high[self.indices]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=obs_space.dtype)

    def observation(self, obs):
        return obs[self.indices]


class PadEndVideoWrapper(gym.Wrapper):
    """Freezes the final frame for a set number of steps to extend video length."""
    def __init__(self, env: gym.Env, pad_steps: int = 20):
        super().__init__(env)
        self.pad_steps = pad_steps
        self._current_pad = 0
        self._last_obs = None

    def step(self, action):
        if self._current_pad > 0:
            self._current_pad -= 1
            done = (self._current_pad == 0)
            return self._last_obs, 0.0, done, False, {}

        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self._current_pad = self.pad_steps
            self._last_obs = obs
            # Hide the termination flag from the video recorder so it keeps recording!
            return obs, reward, False, False, info
            
        return obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        self._current_pad = 0
        return self.env.reset(**kwargs)

class FrameStack(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, num_stack: int=1, observe_stack: bool = True):
        gym.utils.RecordConstructorArgs.__init__(self, num_stack=num_stack, observe_stack=observe_stack)
        gym.ObservationWrapper.__init__(self, env)
        self.num_stack = num_stack
        self.unwrapped.num_stack = num_stack
        self.observe_stack = observe_stack
        self.frames = deque(maxlen=num_stack)
        self.frames_render = deque(maxlen=num_stack)
        
        self.last_action = None 
        self.action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

        if self.observe_stack:
            low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
            high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.array(self.frames)

    def _get_render_stack_size(self) -> int:
        length = getattr(self.unwrapped, "length", None)
        if isinstance(length, (int, np.integer)) and int(length) > 0:
            return max(1, min(self.num_stack, int(length)))
        return self.num_stack

    def step(self, action):
        self.last_action = int(action[0]) if isinstance(action, np.ndarray) else int(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. FIXED: Capture the render frame safely here, NOT in the render loop
        if self.env.render_mode == "rgb_array":
            if len(self.frames_render) == self.num_stack:
                self.frames_render.popleft()
            self.frames_render.append(self.env.render())

        if self.observe_stack:
            self.frames.append(observation)
            observation = self.observation(None)

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.last_action = None  
        obs, info = self.env.reset(**kwargs)

        self.frames_render = deque(maxlen=self.num_stack)
        if self.env.render_mode == "rgb_array":
            image = self.env.render()
            for _ in range(self.num_stack):
                self.frames_render.append(image.copy())

        if self.observe_stack:
            self.frames.clear()
            for _ in range(self.num_stack):
                self.frames.append(obs * 0)
            self.frames.append(obs)
            obs = self.observation(None)

        return obs, info

    def render(self, *args, **kwargs):
        if self.env.render_mode == "rgb_array":
            # Just safety check, no appending!
            if len(self.frames_render) == 0:
                image = self.env.render(*args, **kwargs)
                for _ in range(self.num_stack):
                    self.frames_render.append(image.copy())

            annotated_frames = []
            pad_height = 50  
            render_stack = self._get_render_stack_size()
            frames_to_render = list(self.frames_render)[-render_stack:]
            
            for i, frame in enumerate(frames_to_render):
                steps_back = render_stack - 1 - i
                is_current = (steps_back == 0)
                label = "t (Current)" if is_current else f"t - {steps_back}"
                    
                banner_color = (200, 240, 255) if is_current else (255, 255, 255)

                H, W, C = frame.shape
                padded_frame = np.ones((H + pad_height, W, C), dtype=np.uint8)
                padded_frame[:pad_height, :, 0] = banner_color[0]
                padded_frame[:pad_height, :, 1] = banner_color[1]
                padded_frame[:pad_height, :, 2] = banner_color[2]
                padded_frame[pad_height:, :, :] = frame

                font = cv2.FONT_HERSHEY_SIMPLEX
                
                text_size = cv2.getTextSize(label, font, 0.6, 2)[0]
                text_x = (W - text_size[0]) // 2
                text_y = 20 if is_current else pad_height - 15  
                cv2.putText(padded_frame, label, (text_x, text_y), font, 0.6, (0,0,0), 2, cv2.LINE_AA)

                if is_current and self.last_action is not None:
                    act_str = f"Act: {self.action_names.get(self.last_action, 'N/A')}"
                    act_size = cv2.getTextSize(act_str, font, 0.5, 2)[0]
                    act_x = (W - act_size[0]) // 2
                    cv2.putText(padded_frame, act_str, (act_x, text_y + 20), font, 0.5, (0, 100, 200), 2, cv2.LINE_AA)

                border_size = 2
                bordered_frame = cv2.copyMakeBorder(
                    padded_frame, border_size, border_size, border_size, border_size, 
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
                annotated_frames.append(bordered_frame)

            return np.concatenate(annotated_frames, axis=1)
        else:
            return self.env.render(*args, **kwargs)

class FlatObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.min(), high=self.observation_space.high.max(), 
            shape=self.observation(self.observation_space.sample()).shape, dtype=self.observation_space.dtype
        )
    def observation(self, observation):
        return observation.flatten()  

def apply_obs_wrappers(
    env,
    *,
    partial_obs=None,
    frame_stack=None,
    frame_stack_flatten=True,
    pad_video_tail: bool = True,
):
    if partial_obs:
        env = PartialObs(env, indices=partial_obs.get("indices"), mask=partial_obs.get("mask"))
    
    if frame_stack and frame_stack > 1:
        env = FrameStack(env, num_stack=frame_stack)
        if frame_stack_flatten:
            env = FlatObs(env)
            
    # 3. MAGIC FIX: Automatically apply PadEndVideoWrapper ONLY when the environment 
    # is set up to record video. This guarantees training is unaffected!
    if env.render_mode == "rgb_array" and pad_video_tail:
        env = PadEndVideoWrapper(env, pad_steps=20)
            
    return env

def make_single_env(env_id: str, seed: int, rank: int = 0, record_stats: bool = True, *,
                    env_kwargs: dict | None = None, partial_obs: dict | None = None,
                    frame_stack: int | None = None, frame_stack_flatten: bool = True) -> gym.Env:
    env_kwargs = env_kwargs or {} 
    def _init():
        # create the environment
        env = gym.make(env_id, **(env_kwargs or {}))
        
        # flatten dict observations if needed (e.g. for robotics envs)
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenObservation(env)
        
        #Apply wrappers
        if record_stats:
            env = RecordEpisodeStatistics(env)
        
        env = apply_obs_wrappers(env, partial_obs=partial_obs, frame_stack=frame_stack, frame_stack_flatten=frame_stack_flatten)
        
        global _LOGGED_PARTIAL_OBS
        if (partial_obs or frame_stack) and not _LOGGED_PARTIAL_OBS:
            _LOGGED_PARTIAL_OBS = True
            logger_mod.info("PartialObs config: indices=%s mask=%s -> obs_space=%s", None if not partial_obs else partial_obs.get("indices"), None if not partial_obs else partial_obs.get("mask"), env.observation_space)
            
        env.reset(seed=seed + rank)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed + rank)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed + rank)
        return env
    return _init

def make_envs(seed: int, env_id: str = "CartPole-v1", n_envs: int = 8, *,
              env_kwargs: dict | None = None, partial_obs: dict | None = None,
              frame_stack: int | None = None, frame_stack_flatten: bool = True) -> Tuple[gym.vector.VectorEnv, gym.Env]:
    env_kwargs = env_kwargs or {} 
    env_fns = [make_single_env(env_id, seed=seed, rank=i, partial_obs=partial_obs, frame_stack=frame_stack, frame_stack_flatten=frame_stack_flatten, env_kwargs=env_kwargs) for i in range(n_envs)]
    env_train = AsyncVectorEnv(env_fns) if n_envs > 1 else SyncVectorEnv(env_fns)
    env_eval = SyncVectorEnv([make_single_env(env_id, seed=seed, rank=n_envs + 1, partial_obs=partial_obs, frame_stack=frame_stack, frame_stack_flatten=frame_stack_flatten, env_kwargs=env_kwargs)])
    return env_train, env_eval

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.mean = new_mean
        self.count = tot_count

class VecNormalize(VectorWrapper):
    def __init__(self, venv, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, epsilon=1e-8):
        super().__init__(venv)
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.epsilon = epsilon
        obs_shape = self.env.single_observation_space.shape
        self.obs_rms = RunningMeanStd(shape=obs_shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = 0.99

    def step(self, actions):
        obs, rews, terms, truncs, infos = self.env.step(actions)
        raw_rews = rews
        dones = np.logical_or(terms, truncs)
        if self.norm_obs:
            if self.training: self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
        if self.norm_reward:
            if self.training:
                self.returns = self.returns * self.gamma + rews
                self.ret_rms.update(self.returns)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        try: infos["raw_reward"] = raw_rews
        except Exception: pass
        self.returns[dones] = 0
        return obs, rews, terms, truncs, infos

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        if self.norm_obs:
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
        return obs, infos
