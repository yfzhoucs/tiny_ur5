import clip
import torch
from PIL import Image
import torch.nn as nn
# from utils import *
import random
import json


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Encoder(torch.nn.Module):
    def __init__(self, clip_model='ViT-B/32', device='cuda'):
        super(Encoder, self).__init__()
        self.device = device
        model, self.preprocess = clip.load(clip_model, device=self.device)
        self.clipModel = model
        print('Encoder Loaded!')
    
    def forward(self, x):
        '''
            Passing x as a tuple (Image, instruction).
            Input shape to LSTM: (1, 1024).
            Output shape from Softmax: (1, 196)
        '''
        image = x[0].to(self.device)
        text = x[1].to(self.device)

        with torch.no_grad():
            image_features = self.clipModel.encode_image(image)
            text_features = self.clipModel.encode_text(text)

        concat = torch.cat((image_features, text_features), dim=1).float()
        return concat

    # def init_hidden(self):
    #     weight = next(self.parameters()).data
    #     if self.device == 'cuda':
    #         w1 = weight.new(self.n_layers, self.n_hidden).zero_().cuda()
    #         w2 = weight.new(self.n_layers, self.n_hidden).zero_().cuda()
    #     else:
    #         w1 = weight.new(self.n_layers, self.n_hidden).zero_()
    #         w2 = weight.new(self.n_layers, self.n_hidden).zero_()       
    #     hidden = (w1, w2)
    #     return hidden



class Decoder(torch.nn.Module):
    def __init__(self, n_hidden, device='cuda'):
        super(Decoder, self).__init__()
        self.device = device

        self.n_hidden = n_hidden
        self.embed = torch.nn.Embedding(27, self.n_hidden)
        self.gru1 = torch.nn.GRUCell(self.n_hidden, self.n_hidden)
        self.gru2 = torch.nn.GRUCell(self.n_hidden, self.n_hidden)
        self.gru3 = torch.nn.GRUCell(self.n_hidden, self.n_hidden)
        self.classifier = torch.nn.Linear(self.n_hidden, 27)
        print('Decoder Loaded!')
    
    def forward(self, input_token, features1, features2, features3):
        input_token = self.embed(input_token)
        hidden1 = self.gru1(input_token, features1)
        hidden2 = self.gru2(hidden1, features2)
        hidden3 = self.gru3(hidden2, features3)
        out = self.classifier(hidden3)
        return hidden1, hidden2, hidden3, out


class Untokenizer:
    def __init__(self, dict_list):
        with open(dict_list) as json_file:
            self.dict_list = json.load(json_file)
            json_file.close()
        self.inv_dict_list = [''] * len(self.dict_list)
        for token in self.dict_list:
            idx = self.dict_list[token]
            self.inv_dict_list[idx] = token

    def untokenize(self, indices):
        # print(indices)
        tokens = []
        for index in indices:
            token = self.inv_dict_list[index]
            tokens.append(token)
        return ' '.join(tokens)


def train_epoch(encoder, decoder, dataloader, optimizer, criterion):
    for idx, (img, lang_in, lang_out) in enumerate(dataloader):
        img = img.to(device)
        lang_in = lang_in.to(device)
        lang_out = lang_out.to(device)
        features = encoder((img, lang_in))
        # print(lang_out.shape)
        outs = []
        loss = 0
        teacher_forcing = True if random.random() < 0.5 else False
        optimizer.zero_grad()
        for i in range(lang_out.shape[1] - 1):
            if i == 0:
                hidden1, hidden2, hidden3, out = decoder(lang_out[:, 0], features, features, features)
            else:
                if teacher_forcing:
                    hidden1, hidden2, hidden3, out = decoder(lang_out[:, i], hidden1, hidden2, hidden3)
                else:
                    hidden1, hidden2, hidden3, out = decoder(out_next_in, hidden1, hidden2, hidden3)
            topv, topi = out.topk(1)
            out_next_in = topi.squeeze().detach()
            # mask = (lang_out[:, i+1] > 0.5).float()
            # print(mask)
            # print(mask.shape, out.shape)
            # current_loss = criterion(out * mask, lang_out[:, i+1] * mask)
            # exit()
            # print(mask.shape, out.shape)
            current_loss = criterion(out, lang_out[:, i+1])
            loss = loss + current_loss
        loss.backward()
        optimizer.step()
        print(f'{idx} in {len(dataloader)}', loss.item())

    return encoder, decoder, optimizer


def val_epoch(encoder, decoder, dataloader_val, untokenizer, file, num_test):
    for idx, (img, lang_in, lang_out) in enumerate(dataloader_val):
        if idx >= num_test:
            break

        img = img.to(device)
        lang_in = lang_in.to(device)
        lang_out = lang_out.to(device)
        features = encoder((img, lang_in))

        outs = []
        for i in range(lang_out.shape[1] + 5):
            if i == 0:
                hidden1, hidden2, hidden3, out = decoder(lang_out[:, 0], features, features, features)
            else:
                hidden1, hidden2, hidden3, out = decoder(out_next_in, hidden1, hidden2, hidden3)
            topv, topi = out.topk(1)
            out_next_in = topi.squeeze().detach().unsqueeze(0)
            outs.append(out_next_in.item())
            # print(out_next_in.item(), i)
            if out_next_in.item() == 3:
                break
        outs_pred = untokenizer.untokenize(outs)
        outs_gt = untokenizer.untokenize(lang_out[0])
        file.write(outs_gt + '\n')
        file.write(outs_pred + '\n')
        file.write('\n')






if __name__ == '__main__':
    encoder = Encoder(device=device).to(device)
    decoder = Decoder(1024, device).to(device)  
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    data_dirs_train = [
        '/data/Documents/yzhou298/dataset/tinyur5/collected_long_inst_detailed_sentence',
        # '/data/Documents/yzhou298/dataset/tinyur5/collected_long_inst_val_detailed_sentence',
        '/data/Documents/yzhou298/dataset/tinyur5/collected_long_inst_push_rotate_detailed_sentence',
        # '/data/Documents/yzhou298/dataset/tinyur5/collected_long_inst_push_rotate_val_detailed_sentence'
    ]
    dataset_train = TranslateDataset(data_dirs_train, 'dict_list.json')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16,
                                          shuffle=True, num_workers=8,
                                          collate_fn=pad_collate_xy_lang)

    data_dirs_val = [
        # '/data/Documents/yzhou298/dataset/tinyur5/collected_long_inst_detailed_sentence',
        '/data/Documents/yzhou298/dataset/tinyur5/collected_long_inst_val_detailed_sentence',
        # '/data/Documents/yzhou298/dataset/tinyur5/collected_long_inst_push_rotate_detailed_sentence',
        '/data/Documents/yzhou298/dataset/tinyur5/collected_long_inst_push_rotate_val_detailed_sentence'
    ]
    dataset_val = TranslateDataset(data_dirs_val, 'dict_list.json')
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                          shuffle=True, num_workers=1,
                                          collate_fn=pad_collate_xy_lang)

    ckpt_path = '/data/Documents/yzhou298/ckpts/tinyur5/inst_translater/vanilla_clip/'
    untokenizer = Untokenizer('dict_list.json')

    for i in range(100):
        encoder, decoder, optimizer = train_epoch(encoder, decoder, dataloader_train, optimizer, criterion)

        checkpoint = {
            # 'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(ckpt_path, f'{i}.pth'))

        with open(f'pred/vanilla_clip/{i}.txt', 'w') as f:
            val_epoch(encoder, decoder, dataloader_val, untokenizer, f, 50)
            f.close()
