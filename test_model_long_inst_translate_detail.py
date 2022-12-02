from tiny_ur5 import TinyUR5Env
import yaml
from initializer_long_inst import Initializer
import torch
import numpy as np
import skimage
import clip
from skimage import img_as_ubyte
import json
from recorder import NumpyEncoder
import cv2
from models.translate_model import Encoder, Decoder, Untokenizer
from PIL import Image
import imageio


class TaskSuccess:
    def __init__(self, env: TinyUR5Env, task) -> None:
        self.env = env
        self.task = task

        self.targets = task['target']
        self.target_init_pos = []
        for i in range(len(self.targets)):
            self.target_init_pos.append(env.get_pos_xy(self.targets[i]))
        self.target_init_orientation = []    
        for i in range(len(self.targets)):
            self.target_init_orientation.append(env.get_pos_orientation(self.targets[i]))

        self.task = task['action']
    
    def success(self):
        if self.task == 'push_both':
            success = True
            if self.env.get_pos_xy(self.targets[0])[1] - self.target_init_pos[0][1] < 20:
                success = False
            if self.env.get_pos_xy(self.targets[1])[1] - self.target_init_pos[1][1] < 20:
                success = False
            return success
        elif self.task == 'place_both':
            success = True
            if abs(self.env.get_pos_xy(self.targets[0])[1] - self.env.get_pos_xy(self.targets[2])[1]) > 10:
                success = False
            if abs(self.env.get_pos_xy(self.targets[0])[0] - self.env.get_pos_xy(self.targets[2])[0]) > 10:
                success = False
            if abs(self.env.get_pos_xy(self.targets[1])[1] - self.env.get_pos_xy(self.targets[2])[1]) > 10:
                success = False
            if abs(self.env.get_pos_xy(self.targets[1])[0] - self.env.get_pos_xy(self.targets[2])[0]) > 10:
                success = False
            return success
        elif self.task == 'place_row':
            success = True
            if abs(self.env.get_pos_xy(self.targets[0])[1] - self.env.get_pos_xy(self.targets[2])[1]) > 10:
                success = False
            if abs(self.env.get_pos_xy(self.targets[1])[1] - self.env.get_pos_xy(self.targets[2])[1]) > 10:
                success = False
            return success
        elif self.task == 'pick_food':
            success = True
            for i in range(len(self.targets)):
                if self.env.get_pos_xy(self.targets[i])[1] - self.target_init_pos[i][1] < 20:
                    success = False
            return success
        
        return False




class ModelTester:
    def __init__(self, yaml_file, model, encoder, decoder, 
            model_forward_fn, translater_forward_fn, untokenizer,
            device, method, show_cv2, show_human, time_upper_bound=500) -> None:
        self.yaml_file = yaml_file
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.model_forward_fn = model_forward_fn
        self.translater_forward_fn = translater_forward_fn
        self.untokenizer = untokenizer
        self.device = device
        self.time_upper_bound = time_upper_bound
        self.method = method
        self.show_cv2 = show_cv2
        self.show_human = show_human
    

    def test_1_rollout(self, test_id):
        images = []
        with open(self.yaml_file, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                # print(config, type(config))
            except yaml.YAMLError as exc:
                print(exc)
        
        initializer = Initializer(config)

        config, task = initializer.get_config_and_task()
        sentence = initializer.get_sentence()
        print(sentence)
        the_original_sentence = sentence

        env = TinyUR5Env(render_mode='human', config=config)
        if self.show_human:
            env.render()
        img = env.render('rgb_array')
        images.append(img)
        if self.show_cv2:
            cv2.imshow('tiny_ur5', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        task_success_judge = TaskSuccess(env, task)

        sentences = translater_forward_fn(env, self.encoder, self.decoder, sentence, self.untokenizer, self.device)

        for sentence in sentences:
            time_step = 0
            while time_step < self.time_upper_bound:
                actions = self.model_forward_fn(env, self.model, sentence, self.method, self.device)
                # for i in range(actions.shape[-1]):
                for i in range(60):
                    action = actions[:, i]
                    observation, reward, done, info = env.step(action, eef_z=80)
                    # env.render()
                    img = env.render('rgb_array')

                    if i % 5 == 0:
                        images.append(img)
                    if self.show_human:
                        env.render()
                    if self.show_cv2:
                        cv2.imshow('tiny_ur5', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    time_step += 1

                    success = task_success_judge.success()
                    if success:
                        env.close()
                        return True, task

                print(time_step)
        env.close()
        imageio.mimsave(f'{"_".join(the_original_sentence.split())}.gif', images)
        return False, task
    
    def test(self, test_num:int, name):
        tasks_states = []
        for i in range(test_num):
            success, task = self.test_1_rollout(i)
            task['success'] = success
            tasks_states.append(task)
        
        with open(f'results_{name}_{test_num}.json', 'w') as f:
            json.dump(tasks_states, f, cls=NumpyEncoder, indent=4)


def translater_forward_fn(env, encoder, decoder, sentence, untokenizer, device):
    img = env.render('rgb_array')
    # img = img[::-1, :, :3]
    # img = skimage.transform.resize(img, (224, 224))
    # img = img_as_ubyte(img) / 255
    # # skimage.io.imsave('tmp.png', img_as_ubyte(img))
    # # img = skimage.io.imread('tmp.png')[::-1,:,:3] / 255

    img = Image.fromarray(img[:, :, :3])
    # img = Image.open('tmp.png')
    img = encoder.preprocess(img).unsqueeze(0)
    sentence = clip.tokenize([sentence]).to(device)

    print(img.shape, sentence.shape)

    features = encoder((img, sentence))

    outs = []
    for i in range(50):
        if i == 0:
            hidden1, hidden2, hidden3, out = decoder(torch.ones(1,dtype=torch.int64), features, features, features)
        else:
            hidden1, hidden2, hidden3, out = decoder(out_next_in, hidden1, hidden2, hidden3)
        topv, topi = out.topk(1)
        out_next_in = topi.squeeze().detach().unsqueeze(0)
        outs.append(out_next_in.item())
        # print(out_next_in.item(), i)
        if out_next_in.item() == 3:
            break
    outs_pred = untokenizer.untokenize(outs)
    print(outs_pred)
    outs_pred = outs_pred.replace('<SOS> ', '')
    outs_pred = outs_pred.replace(' <EOS>', '')
    print(outs_pred)
    outs_pred = outs_pred.split('<SEP>')
    for i in range(len(outs_pred)):
        outs_pred[i] = outs_pred[i].strip()
    print(outs_pred)
    return outs_pred


    



def model_forward_fn(env, model, sentence, method, device):
    img = env.render('rgb_array')
    img = img[::-1, :, :3]
    img = skimage.transform.resize(img, (224, 224))
    img = img_as_ubyte(img) / 255
    # skimage.io.imsave('tmp.png', img_as_ubyte(img))
    # img = skimage.io.imread('tmp.png')[::-1,:,:3] / 255
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
    sentence = clip.tokenize([sentence]).to(device)

    def _joints_to_sin_cos_(joints):
        sin_cos_joints = [0] * 8
        for i in range(len(joints)):
            sin_cos_joints[i * 2] = np.sin(joints[i])
            sin_cos_joints[i * 2 + 1] = np.cos(joints[i])
        return sin_cos_joints
    
    def _sin_cos_to_joint_(sin, cos):
        angle = np.arctan(sin / cos)
        if cos < 0:
            if sin > 0:
                angle = angle + np.pi
            else:
                angle = angle - np.pi
        return angle

    def _sin_cos_to_joints_(sin_cos):
        joints = [0] * 4
        for i in range(len(joints)):
            joints[i] = _sin_cos_to_joint_(sin_cos[i * 2], sin_cos[i * 2 + 1])
        return joints
    
    def _seq_sin_cos_to_joint_(sin_cos_seq):
        joints = []
        for i in range(sin_cos_seq.shape[-1]):
            action = _sin_cos_to_joints_(sin_cos_seq[:, i])
            joints.append(action)
        joints = np.transpose(np.array(joints))
        return joints

    if method == 'bcz':
        phis = torch.tensor(np.linspace(0.0, 1.0, 60, dtype=np.float32)) \
            .unsqueeze(0).unsqueeze(0).repeat(1, 8, 1).to(device)
        action = model(img, sentence, phis)
        # return joints_trajectory_pred[0].detach().cpu().numpy()
    elif method == 'ours':

        
        joint_angles = torch.tensor(_joints_to_sin_cos_(env.robot_joints)).unsqueeze(0).to(device)
        phis = torch.tensor(np.linspace(0.0, 1.0, 60, dtype=np.float32)) \
            .unsqueeze(0).unsqueeze(0).repeat(1, 8, 1).to(device)
        stage = 3
        target_position_pred, ee_pos_pred, \
            displacement_pred, attn_map, attn_map2, \
            attn_map3, attn_map4, action = \
            model(img, joint_angles, sentence, phis, stage)
        
    action = action.detach().cpu().numpy()[0]
    action = _seq_sin_cos_to_joint_(action)
    return action


def load_model(ckpt, method, device):
    if method == 'bcz':
        from models.film_model import Backbone
        # model = Backbone(img_size=224, num_traces_out=4, embedding_size=256, num_weight_points=10, input_nc=3, device=device)
        model = Backbone(img_size=224, num_traces_out=8, embedding_size=256, num_weight_points=12, input_nc=3, device=device)
        model.load_state_dict(torch.load(ckpt, map_location=device)['model'], strict=True)
        # model = model.cpu()
        model = model.to(device)

        encoder = Encoder(device=device).to(device)
        decoder = Decoder(1024, device).to(device) 
        decoder.load_state_dict(
            torch.load(
                '/share/yzhou298/ckpts/tinyur5/inst_translater/vanilla_clip/99.pth', 
                map_location=device)['decoder'], strict=True) 
        return model, encoder, decoder
    elif method == 'ours':
        from models.backbone_rgbd_sub_attn_tinyur5 import Backbone
        model = Backbone(img_size=224, embedding_size=256, num_traces_out=2, num_joints=8, num_weight_points=12, input_nc=3, device=device)
        model.load_state_dict(torch.load(ckpt, map_location=device)['model'], strict=True)
        model = model.to(device)

        encoder = Encoder(device=device).to(device)
        decoder = Decoder(1024, device).to(device) 
        decoder.load_state_dict(
            torch.load(
                '/share/yzhou298/ckpts/tinyur5/inst_translater/vanilla_clip/99.pth', 
                map_location=device)['decoder'], strict=True) 
        return model, encoder, decoder


def calculate_success_rate(filename):
    results = json.load(open(filename))

    success = 0
    for i in range(len(results)):
        if results[i]['success'] == True:
            success += 1
    print(success / len(results))

if __name__ == '__main__':
    device = torch.device('cpu')

    # # # BCZ
    # method = 'bcz'
    # # ckpt = '/share/yzhou298/ckpts/tinyur5/train-baseline-bcz-film-resnet-huberloss/200000.pth'
    # ckpt = '/share/yzhou298/ckpts/tinyur5/train-baseline-bcz-film-resnet-huberloss-2-larger-dataset-corrected-rotation/200000.pth'
    # # ckpt = '/share/yzhou298/ckpts/tinyur5/train-baseline-bcz-film-resnet-huberloss-long-inst/60000.pth'
    
    # Ours
    method = 'ours'
    # ckpt = '/share/yzhou298/ckpts/tinyur5/train-tinyur5-rgb-sub-attn-range/90000.pth'
    # ckpt = '/share/yzhou298/ckpts/tinyur5/train-tinyur5-rgb-sub-attn-range-larger-dataset/120000.pth'
    # ckpt = '/share/yzhou298/ckpts/tinyur5/train-tinyur5-rgb-sub-attn-range-larger-dataset/340000.pth'
    ckpt = '/share/yzhou298/ckpts/tinyur5/train-tinyur5-rgb-sub-attn-range-larger-dataset-corrected-rotation-on-original-dataset-finetune/160000.pth'
    
    
    model, encoder, decoder = load_model(ckpt, method, device)
    untokenizer = Untokenizer('dict_list.json')
    modeltester = ModelTester('config.yaml', model, encoder, decoder, 
        model_forward_fn, translater_forward_fn, untokenizer, 
        device, method=method, show_cv2=True, show_human=False, time_upper_bound=200)
    # modeltester.test_1_rollout(0)
    modeltester.test(1, method+'_long_inst_translate_tmp')

    # calculate_success_rate('results_ours_100.json')
    # calculate_success_rate('results_bcz_100.json')
    # calculate_success_rate('results_bcz_correct_rotation_100.json')
    # calculate_success_rate('results_bcz_long_inst_translate_100.json')
    # calculate_success_rate('results_ours_long_inst_translate_100.json')