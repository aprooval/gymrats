#PLAY WITH TRAINED MODEL
from agent.dqnAgent.DQN import DQN as dqn

import torch as T
import numpy as np
import gymnasium as gym

#load model
mdl = dqn([4],2)
mdl.load_state_dict(T.load('./networks/cartPoleNet.pth'))
mdl.eval()

#init environment
env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()

#play game
done = False
actSpace = actSpace = np.array([0,1])
while not done:
    action = actSpace[mdl.forward(T.Tensor(state)).argmax().item()]
    nextState, reward, terminated, truncated, infos = env.step(action)
    env.render()

    state = nextState
    done  = truncated or terminated

env.close()