import argparse
import numpy as np
import glob
import pathlib

from PIL import Image
from torchvision import transforms
from tqdm.contrib import tzip

from .modeling.deeplab import *
from .modeling import custom_transforms as tr


def infer_dl(in_path, out_path, ckpt='DLv3+torch.pth.tar'):
    model = DeepLab(num_classes=2, backbone='resnet', output_stride=16, sync_bn=False, freeze_bn=False)
    ckpt = pathlib.Path(__file__).parent.absolute() / ckpt
    ckpt = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.cuda()
    model.eval()
    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    img_names = sorted(glob.glob(in_path + "/*.png"))

    for i, triple in enumerate(tzip(img_names, img_names[1:], img_names[2:])):
        images = [Image.open(triple[0]).convert('RGB'),
                  Image.open(triple[1]).convert('RGB'),
                  Image.open(triple[2]).convert('RGB')]
        tensor_in = composed_transforms(images).unsqueeze(0)

        if torch.cuda.is_available():
            tensor_in = tensor_in.cuda()
        with torch.no_grad():
            output = model(tensor_in)

        # probability map or threshold
        pred = output[0][1].cpu().numpy()
        pred -= pred.min()
        pred *= 255.0 / pred.max()
        pred = np.where(pred < 128, 0, 255)

        # # max class prob
        # preds = output.cpu()[0].numpy()
        # pred0 = (preds[0] - preds[0].min()) * 255.0 / preds[0].max()
        # pred1 = (preds[1] - preds[1].min()) * 255.0 / preds[1].max()
        # pred = np.where(pred0 > pred1, 0, 255)

        im = Image.fromarray(pred.astype('uint8'), 'L')
        im.save(out_path + '/' + str(i + 1).zfill(6) + ".png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

    parser.add_argument('--in-path', type=str, help='image to test')
    parser.add_argument('--out-path', type=str, help='mask image to save')
    parser.add_argument('--ckpt', type=str, default='DLv3+torch.pth.tar', help='saved model')
    parser.add_argument('--out-stride', type=int, default=16, help='network output stride (default: 8)')
    args = parser.parse_args()
    infer_dl(args.in_path, args.out_path)
