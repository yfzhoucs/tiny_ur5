import numpy as np


def l2(xy1, xy2):
    x1 = xy1[0]
    y1 = xy1[1]
    x2 = xy2[0]
    y2 = xy2[1]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1/2)

class Reach:
    def __init__(self, env, obj, offset=None):
        self.env = env
        self.obj = obj
        self.offset = offset

    def generate_action(self):
        base_xy = self.env.get_base_xy()
        obj_xy = self.env.get_pos_xy(self.obj)
        # action = self.env.ik(
        #     [(obj_xy[0] - base_xy[0]) / self.env.scale, 
        #     (obj_xy[1] - base_xy[1]) / self.env.scale, 
        #     0])
        action = self.env.ik(
            [(obj_xy[0] - base_xy[0]) / self.env.scale, 
            (obj_xy[1] - base_xy[1]) / self.env.scale, 
            np.arctan((obj_xy[0] - base_xy[0]) / (obj_xy[1] - base_xy[1]) + np.random.uniform(-0.1, 0.1))])
        action[-1] = -0.1
        # print(f'Reach: {self.obj} {obj_xy[0] - base_xy[0]}, {obj_xy[1] - base_xy[1]}')
        print(f'Reach: {obj_xy[0]}, {obj_xy[1]}')
        print(f'Currently at: {self.env._eef_()}')
        return action
    
    def done(self):
        return (l2(
            self.env.get_pos_xy(self.obj),
            self.env._eef_()
            ) < 15 * self.env.scale)


class Grip:
    def __init__(self, env):
        self.env = env
    
    def generate_action(self):
        print('grip')
        actions = self.env.get_joint_angles()
        actions[-1] = 0.1
        return actions
    
    def done(self):
        if self.env.get_joint_angles()[-1] > 0:
            return True
        else:
            return False


class Release:
    def __init__(self, env):
        self.env = env
    
    def generate_action(self):
        print('release')
        actions = self.env.get_joint_angles()
        actions[-1] = -0.1
        return actions
    
    def done(self):
        if self.env.get_joint_angles()[-1] > 0:
            return False
        else:
            return True


class Rotate:
    def __init__(self, env, angle):
        self.env = env
        self.angle = angle

        self.eef = None
        self.base_xy = self.env.get_base_xy()
        self.eef_orientation = None
        self.gripper = self.env.get_joint_angles()[-1]
        self.angle = angle
    
    def generate_action(self):
        print('rotate', self.gripper)
        if self.eef is None:
            self.eef = self.env._eef_()
        if self.eef_orientation is None:
            self.eef_orientation = self.env._eef_orientation_()
        self.target_pos = [
            (self.eef[0] - self.base_xy[0]) / self.env.scale,
            (self.eef[1] - self.base_xy[1]) / self.env.scale,
            self.eef_orientation + self.angle,
        ]
        action = self.env.ik(self.target_pos)
        action[-1] = 0.1
        return action
    
    def done(self):
        return abs(self.env._eef_orientation_() - self.target_pos[-1]) < 0.1


class Move:
    def __init__(self, env, pos):
        self.env = env
        # eef = self.env._eef_()
        self.eef_orientation = None
        self.gripper = None
        self.eef = None
        self.pos = pos
        self.base_xy = self.env.get_base_xy()
        # self.target_pos = [
        #     (eef[0] + pos[0]) / self.env.scale ,
        #     (eef[1] + pos[1]) / self.env.scale,
        #     eef_orientation,
        # ]
    
    def generate_action(self):
        print('move')
        if self.eef is None:
            self.eef = self.env._eef_()
        if self.eef_orientation is None:
            self.eef_orientation = self.env._eef_orientation_()
        if self.gripper is None:
            self.gripper = self.env.get_joint_angles()[-1]
        self.target_pos = [
            (self.eef[0] + self.pos[0] * self.env.scale - self.base_xy[0]) / self.env.scale,
            (self.eef[1] + self.pos[1] * self.env.scale - self.base_xy[1]) / self.env.scale,
            self.eef_orientation,
        ]
        action = self.env.ik(self.target_pos)
        action[-1] = self.gripper
        return action
    
    def done(self):
        # print((self.target_pos[0] * self.env.scale + self.base_xy[0], self.target_pos[1] * self.env.scale + self.base_xy[1]), self.env._eef_())
        return (l2(
            (self.target_pos[0] * self.env.scale + self.base_xy[0], self.target_pos[1] * self.env.scale + self.base_xy[1]),
            self.env._eef_()
            ) < 15 * self.env.scale)


class Planner:
    def __init__(self, task, env):
        self.task = task
        self.env = env

        self.actions = self._generate_plan_()
        self.current_action = 0
    

    def _generate_plan_(self):
        if self.task['action'] == 'push_forward':
            return [
                Reach(self.env, self.task['target']),
                Grip(self.env),
                Move(self.env, [0, 100]),]
        elif self.task['action'] == 'push_backward':
            return [
                Reach(self.env, self.task['target']),
                Grip(self.env),
                Move(self.env, [0, -100]),]
        elif self.task['action'] == 'push_left':
            return [
                Reach(self.env, self.task['target']),
                Grip(self.env),
                Move(self.env, [-100, 0]),]
        elif self.task['action'] == 'push_right':
            return [
                Reach(self.env, self.task['target']),
                Grip(self.env),
                Move(self.env, [100, 0]),]
        elif self.task['action'] == 'rotate_clock':
            return [
                Reach(self.env, self.task['target']),
                Grip(self.env),
                Rotate(self.env, -1),]
        elif self.task['action'] == 'rotate_counterclock':
            return [
                Reach(self.env, self.task['target']),
                Grip(self.env),
                Rotate(self.env, 1),]


    def generate_action(self):
        assert self.current_action < len(self.actions)

        action = self.actions[self.current_action].generate_action()

        if self.actions[self.current_action].done():
            self.current_action += 1
            
        return action
    

    def ends(self):
        if self.current_action < len(self.actions):
            return False
        else:
            return True
