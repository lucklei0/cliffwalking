#!/usr/bin/env python

import gymnasium
from stable_baselines3 import PPO 


"""# création de l'environnement
env = gymnasium.make('CliffWalking-v0')
env.reset()

# initialisation de l'agent TRPO
model = PPO("MlpPolicy",'CliffWalking-v0', verbose=1)


# apprentissage de la politique
TIMESTEPS = 300000

for i in range(10):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save("ppo_cliffwalking-vo")"""


# évaluation de la politique
env = gymnasium.make('CliffWalking-v0', render_mode= 'human')
obs, info = env.reset()

model = PPO.load("ppo_cliffwalking-vo")

done = False
trunc = False
while not done and not trunc:
    action, _states = model.predict(obs)
    obs, reward, done, trunc, info = env.step(action.item())


# fermeture de l'environnement
env.close()

