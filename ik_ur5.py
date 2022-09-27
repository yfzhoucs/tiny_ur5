from tiny_ur5 import TinyUR5Env

env = TinyUR5Env(render_mode='human')

observation, info = env.reset(seed=42, return_info=True)

for i in range(12000):
    action = env.ik([0, 0])
    # action = env.ik([-40, 30])
    action[-1] = -0.1

    print(i, env._eef_())
    observation, reward, done, info = env.step(action)
    print(observation)
    img = env.render()
    # print(img[0].shape)

    if done:
        observation, info = env.reset(return_info=True)

for i in range(20):
    action = env.ik([-40, 30])
    action[-1] = 0.1

    print(i, env._eef_())
    observation, reward, done, info = env.step(action)
    print(observation)
    img = env.render()
    # print(img[0].shape)

    if done:
        observation, info = env.reset(return_info=True)

for i in range(120):
    action = env.ik([0, 90])
    action[-1] = 0.1

    print(i, env._eef_())
    observation, reward, done, info = env.step(action)
    print(observation)
    img = env.render()
    # print(img[0].shape)

    if done:
        observation, info = env.reset(return_info=True)


env.close()