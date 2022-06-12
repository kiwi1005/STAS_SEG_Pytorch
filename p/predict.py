import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
import segmentation_models_pytorch as smp

def predict_img(net,
                full_img,
                device,
                scale_factor=1,#Df:1
                out_threshold=0.8):
    net.eval()
    img = torch.from_numpy(BasicDataset.predict(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        n_classes=2
        if n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((full_img.size[1], full_img.size[0])),#  ,interpolation=transforms.InterpolationMode.BICUBIC
            transforms.Resize((942, 1716)),#  ,interpolation=transforms.InterpolationMode.BICUBIC
            # transforms.Pad(padding=6,fill=0),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',fromfile_prefix_chars='@')
    
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    # parsee.add_argument('--folder', action='store_true', default=False, help='input all folder imgs' )
    # parser.parse_args('@predictALL.txt')
    return parser.parse_args(['@predictALL.txt'])

# def folder_to_file(args):
    # file = os.  
    # for i in 


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        img_np=(np.argmax(mask, axis=0) * 255 / (mask.shape[0]-1)).astype(np.uint8)
        print(img_np.shape)
        print(np.max(img_np))
        
        return Image.fromarray(img_np)
        


if __name__ == '__main__':
    # args = get_args() #default
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = smp.UnetPlusPlus(
       # encoder_name="efficientnet-b6",#  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
       # encoder_name="resnext50_32x4d",#V3   
       # encoder_name="se_resnext101_32x4d",#V4  
       # encoder_name="inceptionv4",#Incepv4
       encoder_name="timm-efficientnet-b7",#labV3n
       # encoder_name="timm-efficientnet-b5",#sheep100
       # encoder_weights='imagenet',  # use `imagenet` pretreined weights for encoder initialization
       encoder_weights='noisy-student',  # use `imagenet` pretreined weights for encoder initialization
       in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
       classes=2,  # model output channels (number of classes in your dataset)
    )

    # net =smp.DeepLabV3Plus(
       # encoder_name='timm-efficientnet-b4',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
       
       # encoder_weights='noisy-student',  # use `imagenet` pretreined weights for encoder initialization
       
       # in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
       # classes=2,  # model output channels (number of classes in your dataset)
    # )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        # Default
        # if not args.no_save:
            # out_filename = out_files[i]
            # result = mask_to_image(mask)
            # result.save(out_filename)
            # logging.info(f'Mask saved to {out_filename}')

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')            

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
