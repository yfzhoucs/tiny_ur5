import random
import copy
import numpy as np


class Initializer:
    def __init__(self, config, obj_num_low=3, obj_num_high=5):
        self.config = config
        self.obj_num_low = obj_num_low
        self.obj_num_high = obj_num_high
        self.available_actions = [
            'push_forward',
            'push_backward',
            'push_left',
            'push_right',
            'rotate_clock',
            'rotate_counterclock',
            'place_forward',
            'place_backward',
            'place_left',
            'place_right',
            'place_forward',
            'place_backward',
            'place_left',
            'place_right']
        # self.available_actions = [
        #     'place_forward',
        #     'place_backward',
        #     'place_left',
        #     'place_right']

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
            'place_forward': [
                'place',
                'put',
                'keep',
            ],
            'place_backward': [
                'place',
                'put',
                'keep',
            ],
            'place_left': [
                'place',
                'put',
                'keep',
            ],
            'place_right': [
                'place',
                'put',
                'keep',
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
            'place_forward': [
                'forward',
                'to the front',
                'ahead'
            ],
            'place_backward': [
                'backward',
                'back',
                'to the back',
            ],
            'place_left': [
                'to the left',
                'left',
                'to the left hand side',
            ],
            'place_right': [
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
            ],
            'mouse': [
                'mouse'
            ],
            'basketball': [
                'basketball'

            ],
            'book': [
                'book'
            ],
            'bulb': [
                'bulb'
            ],
            'rubikscube': [
                'rubikscube',
                'rubik'
            ],
            'A': [
                'A',
                'a',
                'letter a'
            ],
            'B': [
                'B',
                'b',
                'letter b'
            ],
            'C': [
                'C',
                'c',
                'letter c'
            ],
            'D': [
                'D',
                'd',
                'letter d'
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
        available_objs = []
        for obj in self.config['objects']:
            if 'position' in self.config['objects'][obj]:
                available_objs.append(obj)

        num_obj = random.randint(self.obj_num_low, self.obj_num_high)
        sampled_objs = random.sample(available_objs, num_obj)

        config = {}
        for key in self.config:
            if not key == 'objects':
                config[key] = copy.deepcopy(self.config[key])
        
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

        task = {
            'action': random.sample(self.available_actions, 1)[0],
            # 'action': 'place_right',
            'target': random.sample(sampled_objs, 2),
            # 'target2': random.sample(sampled_objs, 1)[1],
        }
        # task = {'action': 'place_backward', 'target': 'clock', 'target2': 'watermelon'}

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
        v = self.get_verb(self.task['action'])
        adv = self.get_adv(self.task['action'])
        np = self.get_np(self.task['target'][0])
        sentence = v + ' ' + np + ' ' + adv
        return sentence