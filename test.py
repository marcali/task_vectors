import torch
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments
import sys

if 'ipykernel' in sys.argv[0]:
    sys.argv = [sys.argv[0]]

args = parse_arguments()

# Config
dataset = 'MNIST'
model = 'ViT-B-16'
args = parse_arguments()
args.data_location = '/path/to/data'
args.model = model
args.save = f'src/checkpoints/{model}'
pretrained_checkpoint = f'src/checkpoints/{model}/zeroshot.pt'
finetuned_checkpoint = f'src/checkpoints/{model}/{dataset}/finetuned.pt'


# Create the task vector
task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint)
# Negate the task vector
neg_task_vector = -task_vector
# Apply the task vector
image_encoder = neg_task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.5)
# Evaluate
eval_single_dataset(image_encoder, dataset, args)
eval_single_dataset(image_encoder, 'ImageNet', args)