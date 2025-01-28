# Super Mario Bros Reinforcement Learning Agent

This project implements a reinforcement learning agent using Proximal Policy Optimization (PPO) to play Super Mario Bros. The agent learns to navigate through levels by maximizing rightward movement while avoiding obstacles and death.

## Project Overview

The agent uses a combination of:
- OpenAI Gym's Super Mario Bros environment
- Stable-Baselines3 PPO implementation
- Custom reward function
- Frame stacking for temporal awareness
- Grayscale observation processing

## Installation

```bash
pip install stable-baselines3
pip install gym-super-mario-bros
pip install numpy
pip install matplotlib
```

## Environment Setup

The environment is configured with:
- Grayscale observations (240x256 pixels)
- Frame stacking (4 frames)
- Simple movement action space
- Human-mode rendering
- Custom reward function combining:
  - Velocity (x-axis movement)
  - Time penalty (encourage faster completion)
  - Death penalty (discourage dying)

## Reward Function

The reward function optimizes for three key variables:

1. Velocity (v):
   - Calculated as difference in x-position between states
   - Positive reward for moving right (v > 0)
   - Negative reward for moving left (v < 0)
   - Zero reward for no movement (v = 0)

2. Clock (c):
   - Tracks time between frames
   - Implements a penalty for standing still
   - Encourages constant forward progress

3. Death Penalty (d):
   - Applied when the agent dies
   - Encourages survival and careful navigation

## Training

The model is trained using PPO with the following parameters:
```python
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",
    learning_rate=0.000001,
    n_steps=512
)

# Training for 1 million timesteps
model.learn(total_timesteps=1000000, callback=callback)
```

## Model Architecture

Key PPO features utilized:
- Clipped Objective Function: Prevents drastic policy updates
- Sample Efficiency: Effective use of collected data
- Advantage Estimation: Improved action evaluation
- Entropy Regularization: Encourages exploration
- Trust Region Constraint: Ensures stable learning

## Checkpointing

The project includes a custom training callback that:
- Saves model checkpoints every 10,000 steps
- Maintains best model based on performance
- Saves to configurable checkpoint directory

## Usage

To run a trained model:
```python
# Load the trained model
model = PPO.load("./train/best_model_1000000")

# Run the model
state = env.reset()
while True:
    action = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
```

## Project Structure

```
.
├── /train/          # Model checkpoints
├── /logs/           # Training logs
└── main.py          # Training script
```

## Performance

The model learns to:
- Progress through levels consistently
- Avoid obstacles
- Maintain forward momentum
- Handle basic platforming challenges

## Future Improvements

Potential enhancements:
1. Implement curriculum learning
2. Add more sophisticated reward shaping
3. Experiment with different architectures
4. Include power-up awareness in state
5. Extend to more complex action spaces
## Acknowledgements

This project would not have been possible without the support and resources from various contributors and communities. We would like to express our gratitude to the following:

- **[OpenAI Gym](https://gym.openai.com/)**: For providing a highly flexible and accessible environment for reinforcement learning, particularly the Super Mario Bros environment, which served as the foundation of this project.
- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)**: For their efficient and easy-to-use implementation of PPO, which made it possible to train the reinforcement learning agent with minimal setup.
- **[DeepMind and the Reinforcement Learning Community](https://deepmind.com/research)**: For their foundational research in reinforcement learning, which has been pivotal in shaping modern RL algorithms like PPO.
- **The [Super Mario Bros Community](https://www.mariowiki.com/Super_Mario_Bros.)**: For creating the beloved game that continues to inspire innovation in AI research and experimentation.
- **GitHub Contributors**: For their open-source contributions, allowing the reuse and enhancement of the tools and code utilized in this project.
- **My mentors, friends, and family**: For their continuous support and encouragement throughout the development of this project.

Thank you all for your valuable contributions to this project.
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

```pip install stable-baselines3==1.2.0 gym==0.22.0 gym-super-mario-bros==7.4.0 nes-py==8.2.1 matplotlib==3.7.5 numpy==1.24.1```


## License

[MIT](https://choosealicense.com/licenses/mit/)
##ScreenShots
![Tensorboard](https://github.com/user-attachments/assets/96d8888c-21ed-422b-80fe-afb1f8e08ca1)
![image](https://github.com/user-attachments/assets/dbf6735a-a4b1-4b88-b59e-a2fa3bdb1331)

