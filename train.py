import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from tokenizers import Tokenizer
import math
from tqdm import tqdm
from model import Model,EMB_DIM,FFN_DIM,NUM_LAYERS,SEQ_LEN
import numpy as np

tok = Tokenizer.from_file('tokenizer.json')
model = Model(EMB_DIM,tok.get_vocab_size(),NUM_LAYERS,SEQ_LEN,FFN_DIM).cuda()

with open('datasets/cleaned.txt','r', encoding='utf-8') as f:
    data = f.read()

vocab = tok.encode(data)
tokens = np.array(vocab.ids)

context = 128
stride = 16
x = []
y = []
for i in range(0,len(tokens) - context,stride):
    x.append(tokens[i:context+i])
    y.append(tokens[i+1:i+context+1])

opt = torch.optim.Adam(model.parameters(),lr=0.0003)
loss = nn.CrossEntropyLoss(ignore_index=0)
scaler = torch.amp.Gradscaler('cuda')

try:
    checkpoint = torch.load('model.pth',weights_only=True)
    opt.load_state_dict(checkpoint['opt'])
    model.load_state_dict(checkpoint['model'])
except Exception as e:
    print(f'Error In CheckPoint Loading Training From **SCRATCH** Error:{e}')

uniform_loss = math.log(tok.get_vocab_size())
print(f"Uniform Loss: {uniform_loss}")
x = torch.from_numpy(np.array(x)).long()
y = torch.from_numpy(np.array(y)).long()

dataset = TensorDataset(x, y)
loader = DataLoader(dataset,batch_size=170,shuffle=True,pin_memory=True)

EPOCHS = 15
grads = []
l_mean = []
acc = []
for epoch in range(EPOCHS):
    loop = tqdm(loader,desc=f'Epoch {epoch+1}/{EPOCHS}')
    for xb,yb in loop:
        xb = xb.cuda(non_blocking=True)
        yb = yb.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)

        preds = model(xb)
        l = loss(preds.reshape(-1, tok.get_vocab_size()),yb.reshape(-1))

        pred_ids = preds.argmax(dim=-1)
        accuracy = (pred_ids == yb).float().mean().item()
        top5_ids = preds.topk(5, dim=-1).indices
        top5_accuracy = ((top5_ids == yb.unsqueeze(-1)).any(dim=-1).float().mean().item())

        scaler.scale(l).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.2)
        
        
        uniform_percentage = (uniform_loss - l.item()) / uniform_loss * 100
        l_mean.append(l.item())
        grads.append(grad_norm.item())
        acc.append(accuracy)

        loop.set_postfix({'loss':l.item(),
                          'loss_avg': f'{np.mean(l_mean[-500:]):.2f}',
                          'accuracy_avg': f'{np.mean(acc[-500:]):.2f}',
                          'Uniform':f'{uniform_percentage:.2f}%',
                          'Grad':     f'{np.mean(grads[-800:]):.2f}',
                          'accuracy': f'{accuracy * 100:.2f}%',
                          'top5_acc': f'{top5_accuracy * 100:.2f}%'})
        
        scaler.step(opt)
        scaler.update()

torch.save({
    'model':model.state_dict(),
    'opt': opt.state_dict()
},'model.pth')