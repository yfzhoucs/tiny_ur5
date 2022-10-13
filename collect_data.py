from initializer import Initializer
from tiny_ur5 import TinyUR5Env
from planner import Planner
from recorder import Recorder
import skimage
import skimage.transform
import yaml
import os
import numpy as np
import shutil

# env = TinyUR5Env(render_mode='human', yaml_file='config_2.yaml')

# observation, info = env.reset(seed=42, return_info=True)

# for i in range(12):
#     action = env.ik([0, 0])
#     # action = env.ik([-40, 30])
#     action[-1] = -0.1

#     print(i, env._eef_())
#     observation, reward, done, info = env.step(action)
#     print(observation)
#     img = env.render('rgb_array')
#     # print(img[0].shape)
#     # exit()
#     img = skimage.transform.resize(img, (224, 224))
#     skimage.io.imsave(f'collected/{i}.png', img)

#     if done:
#         observation, info = env.reset(return_info=True)


class DataCollector:
    def __init__(self, yaml_file, data_folder, data_id_start, data_id_end):
        self.yaml_file = yaml_file
        self.data_folder = data_folder
        self.data_id_start = data_id_start
        self.data_id_end = data_id_end
        
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

    def _close_(self, arr1, arr2):
        assert arr1.shape == arr2.shape
        for i in range(arr1.shape[-1]):
            if abs(arr1[i] % (np.pi * 2) - arr2[i] % (np.pi * 2)) > 0.2:
                return False
        return True

    def collect_1_rollout(self, data_id):
        with open(self.yaml_file, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                # print(config, type(config))
            except yaml.YAMLError as exc:
                print(exc)
        
        initializer = Initializer(config)

        config, task = initializer.get_config_and_task()
        sentence = initializer.get_sentence()

        env = TinyUR5Env(render_mode='human', config=config)
        planner = Planner(task, env)
        recorder = Recorder(os.path.join(self.data_folder, f'{data_id}/'))

        step = 0
        while not planner.ends():
            action = planner.generate_action()
            # print('action', action, type(action))
            if(np.isnan(action).any()):
                return False

            current_action_close = False
            while not current_action_close:
                # print(env.robot_joints, action)
                observation, reward, done, info = env.step(action, eef_z=50)
                # img = env.render()
                img = env.render('rgb_array')
                recorder.record_step(step, img, observation, sentence, action, task)

                current_action_close = self._close_(env.robot_joints[:3], action[:3]) and env.robot_joints[-1] * action[-1] > 0
                step += 1
                if step > 500:
                    return False
        recorder.finish_recording()
        env.close()

        del planner
        del recorder
        del env
        del initializer
        return True


    def delete_rollout(self, data_id):
        folder = os.path.join(self.data_folder, f'{data_id}/')
        print('romove folder:', folder)
        if os.path.isdir(folder):
            shutil.rmtree(folder)


    def collect_rollouts(self):
        for i in range(self.data_id_start, self.data_id_end):
            success = False
            while not success:
                self.delete_rollout(i)
                success = self.collect_1_rollout(i)
            


if __name__ == '__main__':
    data_collector = DataCollector(
        yaml_file='config.yaml', 
        data_folder='collected7_random_init_angles', 
        data_id_start=3338, 
        data_id_end=3500)
    data_collector.collect_rollouts()