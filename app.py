import gradio as gr
import torch
from tiny_ur5 import TinyUR5Env
import yaml
from initializer import Initializer
import random
import string
import imageio
from skimage import img_as_ubyte
from test_model import model_forward_fn


def load_model(ckpt, method, device):
    if method == 'bcz':
        from models.film_model import Backbone
        # model = Backbone(img_size=224, num_traces_out=4, embedding_size=256, num_weight_points=10, input_nc=3, device=device)
        model = Backbone(img_size=224, num_traces_out=8, embedding_size=256, num_weight_points=12, input_nc=3, device=device)
        model.load_state_dict(torch.load(ckpt, map_location=device)['model'], strict=True)
        # model = model.cpu()
        model = model.to(device)
        return model
    elif method == 'ours':
        # import tinyur5.models.backbone_rgbd_sub_attn_tinyur5.Backbone as Backbone
        # import tinyur5
        # import models.backbone_rgbd_sub_attn_tinyur5.Backbone
        from models.backbone_rgbd_sub_attn_tinyur5 import Backbone
        # from tinyur5.models.backbone_rgbd_sub_attn_tinyur5 import Backbone
        model = Backbone(img_size=224, embedding_size=256, num_traces_out=2, num_joints=8, num_weight_points=12, input_nc=3, device=device)
        model.load_state_dict(torch.load(ckpt, map_location=device)['model'], strict=True)
        model = model.to(device)
        return model

device = torch.device('cpu')
ckpt = '340000.pth'
print('start loading model')
model = load_model(ckpt, 'ours', device)
print('model loaded')



with gr.Blocks() as demo:


    state = gr.State()

    # with open('config.yaml', "r") as stream:
    #     try:
    #         config = yaml.safe_load(stream)
    #         # print(config, type(config))
    #     except yaml.YAMLError as exc:
    #         print(exc)
        
    #     initializer = Initializer(config)

    #     config, task = initializer.get_config_and_task()
    #     sentence = initializer.get_sentence()
    #     env = TinyUR5Env(config)
    
        
    def init():
        with open('config.yaml', "r") as stream:
            try:
                config = yaml.safe_load(stream)
                # print(config, type(config))
            except yaml.YAMLError as exc:
                print(exc)
            
            initializer = Initializer(config)

            config, task = initializer.get_config_and_task()
            sentence = initializer.get_sentence()
            env = TinyUR5Env(config)
            init_img = env.render('rgb_array')
            current_state = {
                'env': env,
                'id': ''.join(random.choice(string.ascii_lowercase + string.ascii_uppercase + string.ascii_letters) for i in range(20))
            }
            return init_img, current_state


    def exec(sentence, current_state):
        env = current_state['env']
        img = env.render('rgb_array')



        imgs = []
        time_step = 0
        while time_step < 200:
            actions = model_forward_fn(env, model, sentence, 'ours', device)
            # for i in range(actions.shape[-1]):
            for i in range(60):
                action = actions[:, i]
                observation, reward, done, info = env.step(action, eef_z=80)
                img = env.render('rgb_array')
                # imgs.append(Image.fromarray(img))
                if time_step % 6 == 0:
                    imgs.append(img)
                time_step += 1
            print(time_step)
        env.close()

        # context = {}
        # is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # img_buffer = BytesIO()
        # imgs[0].save(img_buffer, save_all=True, append_images=imgs[1:], duration=100, loop=0)
        # img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        # imageio.mimsave(os.path.join('tinyur5/static/', request.session['id'] + '.gif') , [img_as_ubyte(frame) for frame in imgs], 'GIF', fps=20)
        # with open(os.path.join('tinyur5/static/', request.session['id'] + '.gif'), "rb") as gif_file:
        #     img = format(base64.b64encode(gif_file.read()).decode())
        img_id = ''.join(random.choice(string.ascii_lowercase + string.ascii_uppercase + string.ascii_letters) for i in range(20))
        imageio.mimsave(img_id+'.gif', [img_as_ubyte(frame) for frame in imgs], 'GIF', fps=20)





        img = img_id+'.gif'
        next_state = {
            'id': current_state['id'],
            'env': env
        }
        return img, next_state


    with gr.Row():
        load_env = gr.Button(value='load env', )
    with gr.Row():
        instruction = gr.Text(label='Instruction')
    with gr.Row():
        action = gr.Button(value='Action!')
    with gr.Row():
        init_img_placeholder = gr.Image()
        gif_img_placeholder = gr.Image()

    load_env.click(
        init,
        inputs=None,
        outputs=[init_img_placeholder, state],
        show_progress=True
        )
    
    action.click(
        exec,
        inputs=[instruction, state],
        outputs=[gif_img_placeholder, state],
        show_progress=True
    )
    demo.load(
        init,
        inputs=None,
        outputs=[init_img_placeholder, state],
        show_progress=True)
    


demo.launch()
