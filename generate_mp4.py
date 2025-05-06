import os
import gymnasium as gym
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PPOAgent

# --- CONFIGURATION ---
AGENT_CONFIGS = [
    # (agent_name, agent_class, checkpoint_prefix, checkpoint_episodes)
    ("DQN", DQNAgent, "DQNAgent_episode_", range(100, 1001, 100)),
    ("DoubleDQN", DoubleDQNAgent, "DoubleDQNAgent_episode_", range(100, 1001, 100)),
    ("DuelingDQN", DuelingDQNAgent, "DuelingDQNAgent_episode_", range(100, 1001, 100)),
    ("PPO", PPOAgent, "PPOAgent_episode_", range(100, 1001, 100)),
]
MODELS_DIR = "models"
OUTPUT_DIR = "videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FONT_PATH = None  # Set to a .ttf file if you want a custom font
FONT_SIZE = 32
VIDEO_FPS = 30
VIDEO_SIZE = (600, 400)  # Resize frames to this size (width, height), or None for original

def get_env_and_dims():
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return env, state_dim, action_dim

def overlay_label(frame, label, font_path=None, font_size=32):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()
    # Draw a semi-transparent rectangle for better text visibility
    #text_w, text_h = font.getsize(label)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = 10
    rect_x0, rect_y0 = 0, 0
    rect_x1, rect_y1 = text_w + 2*margin, text_h + 2*margin
    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(0,0,0,180))
    draw.text((margin, margin), label, fill=(255,255,255), font=font)
    return np.array(img)

def record_episode(agent_class, model_path, label, video_path, video_size=None, font_path=None, font_size=32, fps=30):
    env, state_dim, action_dim = get_env_and_dims()
    agent = agent_class(state_dim=state_dim, action_dim=action_dim, hidden_dim=128)
    agent.load(model_path)
    state, _ = env.reset()
    frames = []
    done = False
    while not done:
        action = agent.select_action(state, evaluation=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame = env.render()
        frame = overlay_label(frame, label, font_path, font_size)
        if video_size:
            frame = cv2.resize(frame, video_size)
        frames.append(frame)
        state = next_state
    env.close()
    # Write frames to MP4
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved: {video_path}")

def main():
    for agent_name, agent_class, prefix, episodes in AGENT_CONFIGS:
        for ep in episodes:
            model_path = os.path.join(MODELS_DIR, f"{prefix}{ep}.pth")
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
            label = f"{agent_name} Episode {ep}"
            video_path = os.path.join(OUTPUT_DIR, f"{agent_name}_ep{ep}.mp4")
            record_episode(
                agent_class, model_path, label, video_path,
                video_size=VIDEO_SIZE, font_path=FONT_PATH, font_size=FONT_SIZE, fps=VIDEO_FPS
            )

    print("\nAll segments saved. You can concatenate them using a tool like ffmpeg if you want a single compilation.")

if __name__ == "__main__":
    main()