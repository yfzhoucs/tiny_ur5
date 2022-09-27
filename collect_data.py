from tiny_ur5 import TinyUR5Env
import skimage

env = TinyUR5Env(render_mode='human')

observation, info = env.reset(seed=42, return_info=True)

for i in range(12):
    action = env.ik([0, 0])
    # action = env.ik([-40, 30])
    action[-1] = -0.1

    print(i, env._eef_())
    observation, reward, done, info = env.step(action)
    print(observation)
    img = env.render('rgb_array')
    # print(img[0].shape)
    # exit()
    skimage.io.imsave(f'collected/{i}.png', img)

    if done:
        observation, info = env.reset(return_info=True)