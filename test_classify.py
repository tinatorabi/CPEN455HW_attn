import torch
import numpy as np
import csv
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
from utils import *
from model import *
from dataset import *

NUM_CLASSES = 4

def get_label_and_log_likelihood(model, model_input, device):
    num_classes = NUM_CLASSES
    batch_size = model_input.size(0)
    log_likelihood = torch.zeros(batch_size, num_classes, device=device)
    
    for c in range(num_classes):
        labels = torch.full((batch_size,), c, dtype=torch.long, device=device)
        model_output = model(model_input, labels)
        # Assuming model_output is the parameters for a logistic mixture distribution
        nll = discretized_mix_logistic_classify(model_input, model_output)  # Compute NLL
        log_likelihood[:, c] = -nll  # Store the negative log likelihood
    
    _, predicted_labels = log_likelihood.max(1)
    return predicted_labels, log_likelihood.cpu().detach().numpy()

def save_predictions_to_csv(image_numbers, predictions, file_path='predictions.csv'):
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image_Number', 'Predicted_Class'])
        for img_num, pred in zip(image_numbers, predictions):
            csvwriter.writerow([img_num, pred])

def classifier_and_save_data(model, data_loader, device, predictions_file_path, log_likelihood_file_path):
    model.eval()
    all_predictions = []
    all_image_numbers = []
    all_log_likelihoods = []
    
    for batch_idx, (model_input, _) in enumerate(tqdm(data_loader)):
        model_input = model_input.to(device)
        predicted_labels, log_likelihood = get_label_and_log_likelihood(model, model_input, device)
        batch_size = model_input.size(0)
        image_numbers = batch_idx * data_loader.batch_size + torch.arange(batch_size)

        all_predictions.extend(predicted_labels.cpu().numpy())
        all_image_numbers.extend(image_numbers.numpy())
        all_log_likelihoods.append(log_likelihood)

    # Save predictions and log likelihoods
    save_predictions_to_csv(all_image_numbers, all_predictions, predictions_file_path)
    all_log_likelihoods = np.concatenate(all_log_likelihoods, axis=0)
    np.save(log_likelihood_file_path, all_log_likelihoods)
    print(f"Predictions saved to {predictions_file_path}")
    print(f"Log likelihood matrix saved to {log_likelihood_file_path}.npy")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str, default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str, default='test', help='Mode for the dataset')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(
        CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms),
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )

    model = PixelCNN(nr_resnet=1, nr_filters=40, nr_logistic_mix=5, input_channels=3, num_classes=4)
    model.load_state_dict(torch.load('conditional_pixelcnn.pth'))
    model = model.to(device)

    predictions_file_path = 'test_predictions.csv'
    log_likelihood_file_path = 'test_log_likelihoods'
    classifier_and_save_data(model, dataloader, device, predictions_file_path, log_likelihood_file_path)
