import gym

from playing_blackjack.main import DQN
import tensorflow as tf


env = gym.make('Blackjack-v1')
env.reset()
policy_net = DQN(env.action_space.n)
dummy_input = tf.zeros((1, 3))
policy_net(dummy_input)
policy_net.load_weights('blackjack_dqn_fianl.weights.h5')
episodes = 500
win_episodes = 0

for episode in range(episodes):
    state = env.reset()[0]
    episode_reward = 0
    done = False

    while not done:
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        q_values = policy_net(state)
        action = tf.argmax(q_values[0]).numpy()
        print(action)
        next_state, reward, done, *_ = env.step(action)

        state = next_state
        episode_reward += reward

    if episode_reward > 0:
        win_episodes += 1


print(f"Winrate {win_episodes / episode}")
