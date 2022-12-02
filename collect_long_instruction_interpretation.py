import os
import json
import numpy as np
import shutil


# Serialize numpy arrays
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def interprete_sentence(action, target):
    sentences = []
    if action == 'push_both':
        sentences.append(f'push {target[0]} forward')
        sentences.append(f'push {target[1]} forward')
    elif action == 'place_both':
        sentences.append(f'place {target[0]} on {target[2]}')
        sentences.append(f'place {target[1]} on {target[2]}')
    elif action == 'place_row':
        sentences.append(f'drag {target[0]} to the left of center')
        sentences.append(f'drag {target[1]} to the center')
        sentences.append(f'drag {target[2]} to the right of center')
    elif action == 'pick_food':
        for i in range(len(target)):
            sentences.append(f'push {target[i]} forward')
    elif action == 'push_rotate':
        sentences.append(f'push {target[0]} forward')
        sentences.append(f'rotate {target[0]} clockwise')
    return sentences


def interprete_trial(states):
    action = states[0]['task']['action']
    target = states[0]['task']['target']
    sentences = interprete_sentence(action, target)
    for i in range(len(states)):
        states[i]['detailed_sentence'] = sentences
    return states



def interprete_folder(data_folder):
    trials = [os.path.join(data_folder, x) for x in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, x))]
    for idx, trial in enumerate(trials):
        states_fd = os.path.join(trial, 'states.json')
        states_fd = open(states_fd, 'r')
        states = json.load(states_fd)
        # print(states[0]['task']['action'])
        # print(states[0]['task']['target'])

        states = interprete_trial(states)
        # print(states[0]['detailed_sentence'])
        # print(states[-1]['detailed_sentence'])


        with open(os.path.join(trial, 'states_detailed_sentence.json'), 'w') as f:
            json.dump(states, f, cls=NumpyEncoder, indent=4)
        print(f'{idx} in {len(trials)} done')


def transfer_location_folder(data_location, old_folder_name, new_folder_name):
    old_data_folder = os.path.join(data_location, old_folder_name)
    new_data_folder = os.path.join(data_location, new_folder_name)

    old_trials = [os.path.join(old_data_folder, x) for x in os.listdir(old_data_folder) if os.path.isdir(os.path.join(old_data_folder, x))]
    new_trials = [os.path.join(new_data_folder, x) for x in os.listdir(old_data_folder) if os.path.isdir(os.path.join(old_data_folder, x))]

    os.mkdir(new_data_folder)
    for idx in range(len(old_trials)):
        os.mkdir(new_trials[idx])
        shutil.copyfile(
            os.path.join(old_trials[idx], 'states_detailed_sentence.json'),
            os.path.join(new_trials[idx], 'states_detailed_sentence.json')
        )
        shutil.copyfile(
            os.path.join(old_trials[idx], '0.png'),
            os.path.join(new_trials[idx], '0.png')
        )
        print(f'{idx} in {len(old_trials)}')



if __name__ == '__main__':
    # data_folder = '/share/yzhou298/dataset/tinyur5/collected_long_inst_val'
    # interprete_folder(data_folder)

    data_location = '/share/yzhou298/dataset/tinyur5/'
    old_folder_name = 'collected_long_inst_val'
    new_folder_name = 'collected_long_inst_val_detailed_sentence'
    transfer_location_folder(data_location, old_folder_name, new_folder_name)
