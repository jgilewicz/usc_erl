import gymnasium as gym
import numpy as np

def reward_fetch_reach(obs_dict) -> float:
    gripper_pos = obs_dict['observation'][:3]
    target_pos = obs_dict['desired_goal']
    dist = np.linalg.norm(gripper_pos - target_pos)
    return -dist

def reward_fetch_push(obs_dict) -> float:
    gripper_pos = obs_dict['observation'][:3]
    object_pos = obs_dict['observation'][3:6]
    target_pos = obs_dict['desired_goal']
    
    dist_gripper_to_object = np.linalg.norm(gripper_pos - object_pos)
    dist_object_to_target = np.linalg.norm(object_pos - target_pos)
    
    if dist_gripper_to_object > 0.05:
        # Phase 1: gripper moves close to the object
        reward = -dist_gripper_to_object - dist_object_to_target
    else:
        # Phase 2: gripper pushes the object to target
        reward = 1.0 - dist_object_to_target
        
    return reward

def reward_fetch_pick_and_place(obs_dict) -> float:
    gripper_pos = obs_dict['observation'][:3]
    object_pos = obs_dict['observation'][3:6]
    target_pos = obs_dict['desired_goal']
    
    dist_gripper_to_object = np.linalg.norm(gripper_pos - object_pos)
    dist_object_to_target = np.linalg.norm(object_pos - target_pos)
    
    # Table height is ~0.4m, lifted threshold is 0.42m
    is_lifted = object_pos[2] > 0.42
    
    if not is_lifted:
        # Phase 1: gripper approaches object
        reward = -dist_gripper_to_object
    else:
        # Phase 2: object is lifted, transport to target
        reward = 2.0 - dist_object_to_target
        
    return reward

class FetchCustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, env_id: str):
        super().__init__(env)
        self.env_id = env_id

    def step(self, action):
        # obs is a dictionary since this wrapper is applied before FlattenObservation
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if "Reach" in self.env_id:
            custom_reward = reward_fetch_reach(obs)
        elif "Push" in self.env_id:
            custom_reward = reward_fetch_push(obs)
        elif "PickAndPlace" in self.env_id or "Pick" in self.env_id:
            custom_reward = reward_fetch_pick_and_place(obs)
        elif "Slide" in self.env_id:
            custom_reward = reward_fetch_push(obs)
        else:
            custom_reward = reward
            
        return obs, float(custom_reward), terminated, truncated, info
