from mimetypes import init
from tiny_ur5 import TinyUR5Env
import os
import skimage
import skimage.transform
from skimage import img_as_ubyte
import json
import numpy as np


# Serialize numpy arrays
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Recorder:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        os.mkdir(data_folder)
        self.states = []

    def record_step(self, step, img, state, sentence, action, task):
        img = skimage.transform.resize(img, (224, 224))
        skimage.io.imsave(os.path.join(self.data_folder,  f'{step}.png'), img_as_ubyte(img))
        state['sentence'] = sentence
        state['action'] = action
        state['task'] = task
        self.states.append(state)
    
    def finish_recording(self):
        with open(os.path.join(self.data_folder, 'states.json'), 'w') as f:
            json.dump(self.states, f, cls=NumpyEncoder, indent=4)



def rotate_orange(env, recorder):
    step = 0
    for i in range(10):
        action = env.ik([-360, 300, 1])
        action[-1] = -0.1

        observation, reward, done, info = env.step(action, 80)
        # print(observation)
        # exit()
        img = env.render()
        img = env.render('rgb_array')
        recorder.record_step(step, img, observation)
        step += 1

        if done:
            observation, info = env.reset(return_info=True)

    for i in range(20):
        action = env.ik([-360, 300, 1])
        action[-1] = 0.1

        observation, reward, done, info = env.step(action)
        img = env.render()
        img = env.render('rgb_array')

        recorder.record_step(step, img, observation)
        step += 1
        if done:
            observation, info = env.reset(return_info=True)
    recorder.finish_recording()


if __name__ == '__main__':
    env = TinyUR5Env(render_mode='human')
    recorder = Recorder('0/')

    rotate_orange(env, recorder)