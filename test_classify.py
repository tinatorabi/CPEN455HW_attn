import torch
import numpy as np
import csv
from tqdm import tqdm
from torchvision import datasets, transforms
from utils import *
from model import *
from dataset import *

NUM_CLASSES = 4

def get_logits(model, model_input, device):
    num_classes = NUM_CLASSES  # Define the number of classes
    batch_size = model_input.size(0)
    logits = torch.zeros(batch_size, num_classes, device=device)
    
    for c in range(num_classes):
        labels = torch.full((batch_size,), c, dtype=torch.long, device=device)
        model_output = model(model_input, labels)
        logits[:, c] = model_output.squeeze()  # Ensure this matches your model's output shape

    return logits

def save_logits(logits, filename):
    torch.save(logits, f"{filename}.pt")  # Save as a PyTorch tensor (.pt file)
    np.save(f"{filename}.npy", logits.cpu().numpy())  # Save as a NumPy array (.npy file)

def save_predictions_to_csv(image_numbers, predictions, file_path='predictions.csv'):
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image_Number', 'Predicted_Class'])
        for img_num, pred in zip(image_numbers, predictions):
            csvwriter.writerow([img_num, pred])

def classifier_and_save_predictions_and_logits(model, data_loader, device, predictions_file_path, logits_file_path):
    model.eval()
    all_predictions = []
    all_image_numbers = []
    all_logits = []
    
    for batch_idx, (model_input, categories) in enumerate(tqdm(data_loader)):
        model_input = model_input.to(device)
        logits = get_logits(model, model_input, device)
        predictions = logits.argmax(dim=1).cpu().numpy()  # Assuming logits are used directly for prediction
        batch_size = model_input.size(0)
        image_numbers = batch_idx * data_loader.batch_size + torch.arange(batch_size)

        all_logits.append(logits.cpu())
        all_image_numbers.extend(image_numbers.numpy())
        all_predictions.extend(predictions)

    # Save predictions and logits
    save_predictions_to_csv(all_image_numbers, all_predictions, predictions_file_path)
    all_logits = torch.cat(all_logits, dim=0)
    save_logits(all_logits, logits_file_path)
    print(f"Predictions saved to {predictions_file_path}")
    print(f"Logits saved to {logits_file_path}.pt and {logits_file_path}.npy")

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str, default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str, default='test', help='Mode for the dataset')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}  # Important for testing mode

    # Dataset and DataLoader setup
    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(
        CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms),
        batch_size=args.batch_size,
        shuffle=False,  # Important for consistent order in test mode
        **kwargs
    )

    # Model setup
    model = PixelCNN(nr_resnet=1, nr_filters=40, nr_logistic_mix=5, input_channels=3, num_classes=4)
    model.load_state_dict(torch.load('conditional_pixelcnn.pth'))
    model = model.to(device)

    # Execute classification and save results
    predictions_file_path = 'test_predictions.csv'
    logits_file_path = 'test_logits'
    classifier_and_save_predictions_and_logits(model, dataloader, device, predictions_file_path, logits_file_path)
