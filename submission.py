import pandas as pd
import numpy as np

import warnings

import torch

from data_processing import create_dataloader
from processing_model import Model
from seed_everything import seed_everything
from train import train_model

import argparse

parser = argparse.ArgumentParser(
    description = "parameter for training"
)

parser.add_argument('--data_folder', default = '/kaggle/input/global-wheat-detection', type = str,
                    help = 'データの入っているフォルダ')
parser.add_argument('--output_folder', default = '/kaggle/working', type = str,
                    help = "提出用ファイルを出力するフォルダ")
parser.add_argument('--epochs', default = 15, type = int,
                    help = "何エポック学習するか")

args = parser.parse_args()

def submission(model, test_dl):
    results, images, outputs = prediction(model, test_dl)
    sub = pd.DataFrame(results, columns = ['image_id', 'PredictionString'])
    sub.to_csv(args.output_folder + '/submission.csv', index = False)


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
        
    return " ".join(pred_strings)

def prediction(model, dl):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    detection_threshold = 0.5
    results = []
    
    model.to(device)
    
    model.eval()
    with torch.no_grad():
    
        for images, image_ids in test_dl:
            images = list(image.to(device) for image in images)
            outputs = model(images)
        
        for i, image in enumerate(images):
            boxes = outputs[i]['boxes'].detach().cpu().numpy()
            scores = outputs[i]['scores'].detach().cpu().numpy()
            
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            image_id = image_ids[i]
            
            boxes[:,2] = boxes[:,2] - boxes[:,0]
            boxes[:,3] = boxes[:,3] - boxes[:,1]
            
            result = {
                'image_id' : image_id,
                'PredictionString' : format_prediction_string(boxes, scores)
            }
            
            results.append(result)
            
    return results, images, outputs

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    seed_everything(71)
    tr_dl, val_dl, test_dl = create_dataloader(args.data_folder)
    model = train_model(
        model = Model(),
        tr_dl = tr_dl,
        val_dl = val_dl,
        num_epochs = args.epochs,
        output_folder = args.output_folder
    )
    submission(model, test_dl)