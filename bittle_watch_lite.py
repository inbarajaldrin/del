from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from bittle_env_lite import OpenCatGymEnv

def deploy_model():
    # Wrap the environment in DummyVecEnv for inference
    env = DummyVecEnv([lambda: OpenCatGymEnv()])

    # Load the trained model (ensure this path is correct)
    model = PPO.load("./trained_models/opencat_gym_model", device="cpu")

    # Reset the environment and get initial observation
    obs = env.reset()

    # Run the model inference loop
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Restart episode on done
        if done[0]:
            obs = env.reset()

if __name__ == "__main__":
    deploy_model()
