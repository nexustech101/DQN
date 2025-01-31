import gym
import sys
import numpy as np
from src.config import LEARNING_RATE, FUTURE_ACTION_WEIGHT, ITERATIONS

if __name__ == '__main__':

    # Environment parameters
    env = gym.make('MountainCar-v0', render_mode="human")
    highest_observation = env.observation_space.high
    lowest_observation = env.observation_space.low
    n_actions = env.action_space.n

    # Discretization parameters
    observation_size = [20] * len(highest_observation)
    observation_offset = (highest_observation -
                          lowest_observation) / observation_size

    # Try to load existing Q-table, otherwise initialize a new one
    try:
        q_table = np.load("q_table.npy")
        print("Loaded existing Q-table.")
    except FileNotFoundError:
        q_table = np.random.uniform(
            low=-2, high=0, size=tuple(observation_size + [n_actions]))
        print("Initialized new Q-table.")

    def get_discrete_state(state):
        """Convert continuous state into a discrete representation."""
        discrete_state = (state - lowest_observation) / observation_offset
        return tuple(discrete_state.astype(np.int64))  # Fix indexing issue

    epsilon = 0.1  # Exploration rate
    goal_position = 0.5  # Goal position for MountainCar-v0

    try:
        for epoch in range(ITERATIONS):
            state, _ = env.reset()
            # Reset environment and get initial state
            discrete_state = get_discrete_state(state)
            print(f"Epoch {epoch + 1}: Initial Discrete State: {discrete_state}")

            done = False
            while not done:
                # ε-Greedy: Explore randomly sometimes
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, n_actions)  # Random action
                else:
                    # Best known action
                    action = np.argmax(q_table[discrete_state])

                # Step environment
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                new_state = get_discrete_state(state)
                env.render()

                # Q-Learning update rule
                if not done:
                    max_future_q = np.max(q_table[new_state])
                    current_q = q_table[discrete_state + (action, )]
                    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * \
                            (reward + FUTURE_ACTION_WEIGHT * max_future_q)
                    q_table[discrete_state + (action, )] = new_q
                elif new_state[0] >= goal_position:
                    # Reward for reaching goal
                    q_table[discrete_state + (action, )] = 0

                discrete_state = new_state

        # Save Q-table after training
        np.save("q_table.npy", q_table)
        print("Q-table saved.")

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving Q-table...")
        np.save("q_table.npy", q_table)
        print("Q-table saved.")
        env.close()
        sys.exit()

    env.close()
