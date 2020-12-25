import argparse
import os
import numpy as np
import glob
from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import *
from torchvision.utils import make_grid, save_image


parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

parser.add_argument('--in-path', type=str, required=True, help='image to test')
parser.add_argument('--out-path', type=str, required=True, help='mask image to save')
parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception', 'drn', 'mobilenet'],
                    help='backbone name (default: resnet)')
parser.add_argument('--class_number', type=int, default=2, help='number of classes')
parser.add_argument('--ckpt', type=str, default='./models/deeplab-resnet.pth', help='saved model')
parser.add_argument('--out-stride', type=int, default=16, help='network output stride (default: 8)')
parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
parser.add_argument('--gpu-ids', type=str, default='0',
                    help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
parser.add_argument('--dataset', type=str, default='pascal', choices=['pascal', 'coco', 'cityscapes'],
                    help='dataset name (default: pascal)')
parser.add_argument('--crop-size', type=int, default=513, help='crop image size')
parser.add_argument('--sync-bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
parser.add_argument('--freeze-bn', type=bool, default=False,
                    help='whether to freeze bn parameters (default: False)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    try:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    except ValueError:
        raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

if args.sync_bn is None:
    if args.cuda and len(args.gpu_ids) > 1:
        args.sync_bn = True
    else:
        args.sync_bn = False

model = DeepLab(num_classes=args.class_number, backbone=args.backbone, output_stride=args.out_stride,
                sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
ckpt = torch.load(args.ckpt, map_location='cpu')
model.load_state_dict(ckpt['state_dict'])
composed_transforms = transforms.Compose([
    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    tr.ToTensor()])
img_names = sorted(glob.glob(args.in_path + "/*.png"))
print(img_names)
image = [Image.open(img_names[0]).convert('RGB'),
        Image.open(img_names[1]).convert('RGB'),
        Image.open(img_names[2]).convert('RGB')]
target = Image.open(img_names[0]).convert('L')
sample = {'image': image, 'label': target}
tensor_in = composed_transforms(sample)['image'].unsqueeze(0)
model.eval()
if args.cuda:
    image = image.cuda()
with torch.no_grad():
    output = model(tensor_in)  # np.set_printoptions(threshold=np.inf)

print(output.shape)
grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 3,
                        normalize=False, range=(0, 255))
print("type(grid) is: ", type(grid_image))
print("grid_image.shape is: ", grid_image.shape)
pred = output[0][1].numpy()
print(pred)
img_prob = []
print(pred.max())
pred -= pred.min()
pred *= 255.0 / pred.max()
pred = np.where(pred < 220, 0, 255)
print(pred.shape)
im = Image.fromarray(pred.astype('uint8'), 'L')
im.save(args.out_path)
# save_image(grid_image, args.out_path)
