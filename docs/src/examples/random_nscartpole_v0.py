import gym
import dyna_gym.agents.my_random_agent as my_random_agent

### Parameters
env = gym.make('NSCartPole-v0')
# Note: Using a random agent will fail very easily!
agent = my_random_agent.MyRandomAgent(action_space=env.action_space)
timesteps = 100
verbose = False

### Run
env.reset()
done = False
for ts in range(timesteps):
    __, __, done, __ = env.step(agent.act(env,done))
    if verbose:
        env.print_state()
    env.render()
    if ts+1 == timesteps:
        print("Successfully reached end of episode ({} timesteps)".format(ts+1))
    if done:
        print("Episode finished after {} timesteps".format(ts+1))
        break
