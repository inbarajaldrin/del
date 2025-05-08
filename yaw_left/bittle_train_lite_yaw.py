from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from bittle_env_lite_yaw import OpenCatGymEnv, MAXIMUM_LENGTH

def train():
    # Use multiple environments for faster training
    parallel_envs = 24
    envs = make_vec_env(OpenCatGymEnv,
                        n_envs=parallel_envs,
                        vec_env_cls=SubprocVecEnv)

    # Save a checkpoint every 500,000 timesteps
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // parallel_envs,  # Adjusted for parallel envs
        save_path="./trained_models/checkpoints/",
        name_prefix="yaw_right"
    )

    # Simple MLP policy architecture
    custom_arch = dict(net_arch=[256, 256])

    # Create and train PPO model
    model = PPO(
        'MlpPolicy',
        envs,
        seed=42,
        policy_kwargs=custom_arch,
        n_steps=512,
        verbose=1,
        device="cpu"
    )

    model.learn(
        total_timesteps=int(MAXIMUM_LENGTH),
        callback=checkpoint_callback
    )

    # Save final model
    model.save("./trained_models/yaw_right")

if __name__ == "__main__":
    train()
