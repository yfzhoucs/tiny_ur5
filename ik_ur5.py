from tiny_ur5 import TinyUR5Env

env = TinyUR5Env(render_mode='human')

observation, info = env.reset(seed=42, return_info=True)

for i in range(1000):
    print(i)
    input()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation)
    img = env.render()
    # print(img[0].shape)

    if done:
        observation, info = env.reset(return_info=True)
env.close()