from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from bittle_env_lite_yaw import OpenCatGymEnv

def deploy_model():
    # Wrap the environment in DummyVecEnv for inference
    env = DummyVecEnv([lambda: OpenCatGymEnv()])

    # Load the trained model (ensure this path is correct)
    model = PPO.load("./trained_models/yaw_right", device="cpu")

    # Reset the environment and get initial observation
    obs = env.reset()

    # Run the model inference loop indefinitely
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Just ignore 'done' and continue feeding actions
        # No reset, so the robot keeps moving forward indefinitely

if __name__ == "__main__":
    deploy_model()
