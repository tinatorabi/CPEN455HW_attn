import torch
import csv
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse

NUM_CLASSES = 4

def get_label_and_logits(model, model_input, device):
    num_classes = NUM_CLASSES
    batch_size = model_input.size(0)
    logits = torch.zeros(batch_size, num_classes, device=device)
    
    for c in range(num_classes):
        labels = torch.full((batch_size,), c, dtype=torch.long, device=device)
        model_output = model(model_input, labels)
        logits[:, c] = model_output.squeeze()
    
    _, predicted_labels = logits.max(1)
    return predicted_labels, logits

def save_predictions_to_csv(image_numbers, predictions, file_path='predictions.csv'):
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image_Number', 'Predicted_Class'])
        for img_num, pred in zip(image_numbers, predictions):
            csvwriter.writerow([img_num, pred])

def classifier_and_save_predictions_and_logits(model, data_loader, device, predictions_file_path, logits_file_path):
    model.eval()
    all_image_numbers = []
    all_predictions = []
    all_logits = []
    
    for batch_idx, (model_input, categories) in enumerate(tqdm(data_loader)):
        model_input = model_input.to(device)
        predictions, logits = get_label_and_logits(model, model_input, device)
        all_predictions.extend(predictions.cpu().numpy())
        all_logits.append(logits.cpu())
    
    # Concatenate all logits and save to a .pt file
    all_logits = torch.cat(all_logits, dim=0)
    torch.save(all_logits, logits_file_path)

    # Save predictions to a CSV file
    save_predictions_to_csv(all_image_numbers, all_predictions, predictions_file_path)

    return f"Predictions saved to {predictions_file_path}", f"Logits saved to {logits_file_path}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str, default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str, default='test', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}  # Important for ensuring no data is dropped in test mode

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(
        CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms),
        batch_size=args.batch_size,
        shuffle=False,  # Important for consistent order in test mode
        **kwargs
    )

    model = PixelCNN(nr_resnet=1, nr_filters=40, nr_logistic_mix=5, input_channels=3, num_classes=4)
    model.load_state_dict(torch.load('conditional_pixelcnn.pth'))
    model = model.to(device)

    print('Model parameters loaded.')

    # Run classifier on test data and save predictions and logits
    predictions_file_path = 'test_predictions.csv'
    logits_file_path = 'logits.pt'
    classifier_and_save_predictions_and_logits(model, dataloader, device, predictions_file_path, logits_file_path)
    print(f"Predictions saved to {predictions_file_path}")
    print(f"Logits saved to {logits_file_path}")
