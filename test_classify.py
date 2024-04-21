from classification_evaluation import *
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = 4



def save_predictions_to_csv(image_numbers, predictions, file_path='predictions.csv'):
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image_Number', 'Predicted_Class'])
        for img_num, pred in zip(image_numbers, predictions):
            csvwriter.writerow([img_num, pred])




def classifier_and_save_predictions(model, data_loader, device, file_path):
    model.eval()
    all_predictions = []
    all_image_numbers = []

    for batch_idx, (model_input, categories) in enumerate(tqdm(data_loader)):
        model_input = model_input.to(device)
        predictions = get_label(model, model_input, device).cpu().numpy()
        batch_size = model_input.size(0)

        # Generate image numbers based on batch index and batch size
        image_numbers = batch_idx * data_loader.batch_size + torch.arange(batch_size)
        all_image_numbers.extend(image_numbers.numpy())
        all_predictions.extend(predictions)
    
    # Save predictions to a CSV file
    save_predictions_to_csv(all_image_numbers, all_predictions, file_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str, default='test', help='Mode for the dataset')

    
 
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}  # Set drop_last to False for testing

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(
        CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms),
        batch_size=args.batch_size,
        shuffle=False,  # Set shuffle to False for testing
        **kwargs
    )

    model = PixelCNN(nr_resnet=1, nr_filters=40, nr_logistic_mix=5, input_channels=3, num_classes=4)
    model.load_state_dict(torch.load('conditional_pixelcnn.pth'))
    model = model.to(device)

    print('Model parameters loaded.')

    # Run classifier on test data and save predictions
    predictions_file_path = 'test_predictions.csv'
    classifier_and_save_predictions(model, dataloader, device, predictions_file_path)
    
    print(f"Predictions saved to {predictions_file_path}")
