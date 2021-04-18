import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

"""
モデルを学習させる関数
"""

def train_model(model, tr_dl, val_dl, num_epochs, output_folder):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('使用デバイス：', device)
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    scheduler = ReduceLROnPlateau(
        optimizer = optimizer,
        mode = 'min',
        patience = 5,
        verbose = True,
        factor = 0.5
    )
    
    train_loss_list = []
    val_loss_list = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        val_loss = 0.0
        
        model.train()
        
        for images, targets, image_ids in tr_dl:
            images = list(torch.tensor(image, device = device, dtype = torch.float) for image in images)
            targets = [{k : torch.tensor(v, device = device, dtype = torch.int64) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
                
        finish_time = time.time()
            
        print('Epochs：{:03} | Train Loss：{:.4f} | Learning Time：{:.4f}'
                .format(epoch + 1, train_loss, finish_time - start_time))
        
    return model