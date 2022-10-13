from tiny_ur5 import TinyUR5Env
import imageio
import yaml


def pick_orange(env):
    # env = TinyUR5Env(render_mode='human')
    for i in range(100):
        action = env.ik([-360, 300, 1])
        # print(action)
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
        # print(action)
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
        # print(action)
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
        # print(action)
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
        # print(action)
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
    while True:
        for i in range(60):
            action = env.ik([-360, 300, -1])
            # print(action)
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
            # print(action)
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
        # print(action)
        action[-1] = -0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action, 80)
        # print(observation)
        img = env.render()
        img = env.render('rgb_array')
        print(env.robot_joints)
        # print(img.shape)
        # exit()

        # if i % 2 == 0:
        #     images.append(img)
        # print(img[0].shape)

        if done:
            observation, info = env.reset(return_info=True)

    for i in range(20):
        action = env.ik([-360, 300, 1])
        # print(action)
        action[-1] = 0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        print(env.robot_joints)

        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)

    for i in range(120):
        action = env.ik([0, 300, 1])
        # print(action)
        action[-1] = 0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        print(env.robot_joints)
        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)

    for i in range(20):
        action = env.ik([0, 300, 1])
        # print(action)
        action[-1] = -0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        print(env.robot_joints)
        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)

    for i in range(120):
        action = env.ik([-360, 300, -1])
        # print(action)
        action[-1] = -0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        print(env.robot_joints)
        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)

    for i in range(20):
        action = env.ik([-360, 300, -1])
        # print(action)
        action[-1] = 0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        print(env.robot_joints)
        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)


    for i in range(120):
        action = env.ik([0, 300, 1])
        # print(action)
        action[-1] = 0.1

        # print(i, env._eef_())
        observation, reward, done, info = env.step(action)
        # print(observation)
        img = env.render()
        # print(img[0].shape)
        img = env.render('rgb_array')
        print(env.robot_joints)
        # if i % 2 == 0:
        #     images.append(img)
        if done:
            observation, info = env.reset(return_info=True)



# observation, info = env.reset(seed=42, return_info=True)
with open('config.yaml', "r") as stream:
    try:
        config = yaml.safe_load(stream)
        # print(config, type(config))
    except yaml.YAMLError as exc:
        print(exc)
env = TinyUR5Env(config=config, render_mode='human')

images = []
observation, info = env.reset(seed=42, return_info=True)
rotate_orange(env)



# imageio.mimsave('ik3.gif', images)
env.close()

