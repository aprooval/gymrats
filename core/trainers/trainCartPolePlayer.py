from agent.dqnAgent.Agent import Agent
from utils.PlotUtils import PlotUtils as plot

import numpy as np
import gymnasium as gym

#init environment and agent
env         = gym.make("CartPole-v1")
state, info = env.reset()
agent       = Agent([4],[1],np.array([0,1],dtype=np.int64))

scores   = []
epsHist  = []
numGames = 500

for i in range(numGames):
    score = 0
    done = False
    state, info = env.reset()
    
    while not done:
        #act, step, memorize, learn
        action = agent.actSpace[agent.act(state)]
        nextState, reward, terminated, truncated, infos = env.step(action)
        agent.memorize(state,action,nextState,reward,terminated)        
        agent.learn()
        
        state  = nextState
        score += reward

        done = truncated or terminated
    
    scores.append(score)
    epsHist.append(agent.epsilon)
    
    avgScore = (np.mean(scores[-100:]))
    print('episode',i,'score %.2f' % score,'avg. score %.2f' % avgScore,
          'epsilon %.2f' % agent.epsilon)

fileName = 'invertedPendulumDiscrete.png'
x = [i+1 for i in range(numGames)]
plot.plotLearningCurveDeepQ(x,scores,epsHist,fileName)

env.close()