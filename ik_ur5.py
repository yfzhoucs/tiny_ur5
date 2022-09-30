from tiny_ur5 import TinyUR5Env
import imageio


def pick_orange(env):
    # env = TinyUR5Env(render_mode='human')
    for i in range(100):
        action = env.ik([-360, 300, 1])
        print(action)
        action[-1] = -0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action, 80)
        # print(observation)
        img = env.render()
        img = env.render('rgb_array')
        # print(img.shape)
        # exit()

        # if i % 2 == 0:
        #     images.append(img)
        # print(img[0].shape)

        if done:
            observation, info = env.reset(return_info=True)

    for i in range(20):
        action = env.ik([-360, 300, 1])
        print(action)
        action[-1] = 0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')

        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)

    for i in range(120):
        action = env.ik([0, 300, 1])
        print(action)
        action[-1] = 0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)



def rotate_orange(env):
    # env = TinyUR5Env(render_mode='human')
    for i in range(100):
        action = env.ik([-360, 300, 1])
        print(action)
        action[-1] = -0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action, 80)
        # print(observation)
        img = env.render()
        img = env.render('rgb_array')
        # print(img.shape)
        # exit()

        # if i % 2 == 0:
        #     images.append(img)
        # print(img[0].shape)

        if done:
            observation, info = env.reset(return_info=True)

    for i in range(20):
        action = env.ik([-360, 300, 1])
        print(action)
        action[-1] = 0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')

        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)

    for i in range(60):
        action = env.ik([-360, 300, -1])
        print(action)
        action[-1] = 0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)

    for i in range(60):
        action = env.ik([-360, 300, 1])
        print(action)
        action[-1] = 0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)



def pick_orange_and_apple(env):
    # env = TinyUR5Env(render_mode='human')
    for i in range(100):
        action = env.ik([-360, 300, 1])
        print(action)
        action[-1] = -0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action, 80)
        # print(observation)
        img = env.render()
        img = env.render('rgb_array')
        # print(img.shape)
        # exit()

        # if i % 2 == 0:
        #     images.append(img)
        # print(img[0].shape)

        if done:
            observation, info = env.reset(return_info=True)

    for i in range(20):
        action = env.ik([-360, 300, 1])
        print(action)
        action[-1] = 0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')

        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)

    for i in range(120):
        action = env.ik([0, 300, 1])
        print(action)
        action[-1] = 0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)

    for i in range(20):
        action = env.ik([0, 300, 1])
        print(action)
        action[-1] = -0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)

    for i in range(120):
        action = env.ik([-360, 300, -1])
        print(action)
        action[-1] = -0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)

    for i in range(20):
        action = env.ik([-360, 300, -1])
        print(action)
        action[-1] = 0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)


    for i in range(120):
        action = env.ik([0, 300, 1])
        print(action)
        action[-1] = 0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)



# observation, info = env.reset(seed=42, return_info=True)
env = TinyUR5Env(render_mode='human')

images = []
observation, info = env.reset(seed=42, return_info=True)
pick_orange_and_apple(env)



# imageio.mimsave('ik3.gif', images)
env.close()

