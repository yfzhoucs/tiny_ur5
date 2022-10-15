import random
import copy
import numpy as np
import yaml


class Initializer:
    def __init__(self, config, obj_num_low=3, obj_num_high=5):
        self.config = config
        self.obj_num_low = obj_num_low
        self.obj_num_high = obj_num_high
        self.available_actions = [
            'push_both', 
            'place_both', 
            'place_row', 
            'pick_food',]

        self.action_properties = {
            'push_both': {
                'tar_obj_num': 2,
            }, 
            'place_both': {
                'tar_obj_num': 3,
            },
            'place_row': {
                'tar_obj_num': 3,
            }, 
            'pick_food': {
                'tar_obj_num': -1, # -1 indicates a uncertain number
            },}

        self.verb_template = {
            'push_forward': [
                'push',
                'drag',
                'move',
                'get',
            ], 
            'push_backward': [
                'push',
                'drag',
                'move',
                'get',
            ], 
            'push_left': [
                'push',
                'drag',
                'move',
                'get',
            ],  
            'push_right': [
                'push',
                'drag',
                'move',
                'get',
            ],  
            'rotate_clock': [
                'rotate',
                'revolve',
                'turn',
                'spin',
            ],  
            'rotate_counterclock': [
                'rotate',
                'revolve',
                'turn',
                'spin',
            ], 
        }
        self.adv_template = {
            'push_forward': [
                'forward',
                'to the front',
                'ahead'
            ], 
            'push_backward': [
                'backward',
                'back',
                'to the back',
            ], 
            'push_left': [
                'to the left',
                'left',
                'to the left hand side',
            ],  
            'push_right': [
                'to the right',
                'right',
                'to the right hand side',
            ],  
            'rotate_clock': [
                'clockwise',
                'clock wise',
                'right'
            ],  
            'rotate_counterclock': [
                'counterclockwise',
                'anticlockwise',
                'anti clock wise',
                'counter clock wise',
                'left',
            ], 
        }
        self.np_template = {
            'orange': [
                'orange',
                'citrus',
                'sweet orange',
                'lime'
            ],
            'apple': [
                'apple',
                'red apple',
                'red delicious apple',
                'gala'
            ],
            'tomato':[
                'tomato'
            ],
            'strawberry':[
                'strawberry'
            ],
            'watermelon':[
                'watermelon'
            ],
            'banana':[
                'banana'
            ],
            'milk_bottle':[
                'bottle',
                'glass bottle',
                'milk bottle'
            ],
            'clock':[
                'clock',
                'timer',
                'watch'
            ],
            'camera':[
                'camera',
                'DSLR',
                'nikon',
                'canon'
            ]
        }
        return
    
    def _random_place_(self, positions):
        
        def l2(x1, y1, x2, y2):
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1/2)

        def not_too_close(positions, x, y):
            for pos in positions:
                if l2(x, y, pos['x'], pos['y']) <= 100:
                    return False
            return True

        done = False

        while not done:
            # https://programming.guide/random-point-within-circle.html
            a = random.random() * np.pi
            r = 450 * np.sqrt(random.random())
            x = r * np.cos(a) + self.config['desk_width'] * self.config['scale']
            y = r * np.sin(a) + 100

            if not_too_close(positions, x, y) and r > 250:
                done = True
        
        return {
            'x': x,
            'y': y,
            'z': 0
        }

    
    def get_config_and_task(self):

        # Pre-process the config
        config = {}
        for key in self.config:
            if not key == 'objects':
                config[key] = copy.deepcopy(self.config[key])

        available_objs = []
        for obj in self.config['objects']:
            if 'position' in self.config['objects'][obj]:
                available_objs.append(obj)

        # Choose an action
        action = random.sample(self.available_actions, 1)[0]

        # Choose objects to place on the table
        num_obj = random.randint(self.obj_num_low, self.obj_num_high)
        
        if action in [
            'push_both', 
            'place_both', 
            'place_row', ]:
            sampled_objs = random.sample(available_objs, num_obj)
        elif action == 'pick_food':
            sampled_objs = random.sample(
                ['orange', 'apple', 'tomato', 'strawberry', 'watermelon', 'banana', 'milk_bottle'], 1)
            available_objs.remove(sampled_objs[0])
            sampled_objs = sampled_objs + random.sample(available_objs, num_obj-1)

        # Random place these objects and add to config
        config['objects'] = {}
        positions = []
        for obj in self.config['objects']:
            if 'position' not in self.config['objects'][obj]:
                config['objects'][obj] = copy.deepcopy(self.config['objects'][obj])
            elif obj in sampled_objs:
                obj_dict = copy.deepcopy(self.config['objects'][obj])
                obj_dict['position'] = self._random_place_(positions)
                positions.append(obj_dict['position'])
                config['objects'][obj] = obj_dict

        # Get the list of manipulated objects
        if action in [
            'push_both', 
            'place_both', 
            'place_row', ]:
            targets = random.sample(sampled_objs, self.action_properties[action]['tar_obj_num'])
        elif action == 'pick_food':
            targets = [x for x in sampled_objs if x in \
                ['orange', 'apple', 'tomato', 'strawberry', 'watermelon', 'banana', 'milk_bottle']]
        
        task = {
            'action': action,
            'target': targets,
        }

        config['init_joints'] = np.random.uniform(-np.pi / 2, np.pi / 2, size=(4,))
        config['init_joints'][-1] *= 0.2

        self.new_config = config
        self.task = task
        print(config)
        print(task)
        return config, task
    
    def get_verb(self, verb):
        return random.sample(self.verb_template[verb], 1)[0]
    
    def get_adv(self, verb):
        return random.sample(self.adv_template[verb], 1)[0]

    def get_np(self, noun):
        return random.sample(self.np_template[noun], 1)[0]

    def get_sentence(self):
        if self.task['action'] == 'push_both':
            sentence = random.sample(['push', 'drag', 'move'], 1)[0]
            sentence = sentence + ' '
            sentence = sentence + random.sample(['forward ', ''], 1)[0]
            sentence = sentence + random.sample(['both ', ''], 1)[0]
            sentence = sentence + self.get_np(self.task['target'][0])
            sentence = sentence + ' and '
            sentence = sentence + self.get_np(self.task['target'][1])
        elif self.task['action'] == 'place_both':
            sentence = random.sample(['put', 'place', 'drop'], 1)[0]
            sentence = sentence + ' '
            sentence = sentence + random.sample(['both ', ''], 1)[0]
            sentence = sentence + self.get_np(self.task['target'][0])
            sentence = sentence + ' and '
            sentence = sentence + self.get_np(self.task['target'][1])
            sentence = sentence + ' '
            sentence = sentence + random.sample(['on', 'to', 'above'], 1)[0]
            sentence = sentence + ' '
            sentence = sentence + self.get_np(self.task['target'][2])
        elif self.task['action'] == 'place_row':
            sentence = random.sample(['put', 'place'], 1)[0]
            sentence = sentence + ' '
            sentence = sentence + self.get_np(self.task['target'][0])
            sentence = sentence + ', '
            sentence = sentence + self.get_np(self.task['target'][1])
            sentence = sentence + ' and '
            sentence = sentence + self.get_np(self.task['target'][2])
            sentence = sentence + ' in a '
            sentence = sentence + random.sample(['line', 'row'], 1)[0]
        elif self.task['action'] == 'pick_food':
            sentence = random.sample(['push', 'drag', 'move'], 1)[0]
            sentence = sentence + ' all food'
            sentence = sentence + random.sample([' forward', ''], 1)[0]
        else:
            sentence = 'not implemented'

        return sentence


if __name__ == '__main__':
    with open('config.yaml', "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # print(config, type(config))
        except yaml.YAMLError as exc:
            print(exc)
    initializer = Initializer(config)
    initializer.get_config_and_task()
    print(initializer.get_sentence())
