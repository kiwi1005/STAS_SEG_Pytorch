import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import torchvision
# from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations import transforms

 


class BasicDataset(Dataset):

    
    
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        
        newW, newH = int(scale * w), int(scale * h)
        # newW, newH = int(1700), int(928)## for predict!!
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        
        #變成PIL
        
        
        
        # print("A:",type(pil_img))
        transform1 = transforms.Compose([
            # transforms.RandomAdjustSharpness(1.2, p=0.5),
            # transforms.RandomRotation(30, expand=False, fill=255, resample=None),
            transforms.CenterCrop((600,600)),
            
            transforms.ToTensor(),
            # transforms.Normalize((0.8263105, 0.6212347, 0.7698729), (0.15582965, 0.268009, 0.1718304)),
            transforms.ToPILImage(),
            ])
        # scripted_transforms = torch.jit.script(transform)
        transform2 = transforms.Compose([
            transforms.CenterCrop((600,600)),
            ])
        
        # if not is_mask:
            # pil_img = transform1(pil_img)
        
        # if is_mask:
            # pil_img = transform2(pil_img)
        
        
        
        # pil_img = transform1(pil_img)
        # print("B",type(pil_img))
        pil_img = np.asarray(pil_img) 
        # print("C",type(pil_img))
        # pil_img = transforms.ToPILImage(pil_img)
        img_ndarray = np.asarray(pil_img)
        # print("DDD",type(img_ndarray))
        
        
        
        

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)
    @staticmethod
    def dataAug(img,mask):
        transform = A.Compose([
            # A.RandomCrop(width=700, height=700),
            # A.RandomSizedCrop(min_max_height=(512-100, 512+100), height=512, width=512, p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.CropNonEmptyMaskIfExists (512, 512, ignore_values=[255], ignore_channels=[1], always_apply=True, p=1.0),
            # A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
            
        ])
        
        image = np.asarray(img)
        mask = np.asarray(mask)
        transformed = transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        transformed_image = Image.fromarray(np.uint8(transformed_image))
        transformed_mask = Image.fromarray(np.uint8(transformed_mask))

        return transformed_image, transformed_mask
    @staticmethod
    def predict(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        sW, sH = int(newW/32)+1, int(newH/32)+1
        transform = A.Compose([
            # A.Resize(newH,newW,interpolation=3,always_apply=True),
            # A.Sharpen(always_apply=True,p=1),
            # A.GaussianBlur(blur_limit=(1, 3), sigma_limit=0, always_apply=True),
            # A.Sharpen(alpha=(0.5, 0.6), lightness=(0.3, 0.5), always_apply=True),
            A.Resize(sH*32,sW*32,interpolation=3,always_apply=True),
            
            
            # A.CLAHE(clip_limit=2.0, tile_grid_size=(32, 32), always_apply=True),
            
            # A.FancyPCA (alpha=0.1, always_apply=True),
            # A.PadIfNeeded(min_height=sH*32, min_width=sW*32, always_apply=True, border_mode=2,mask_value=255),
         ])
        
        pil_img = np.asarray(pil_img)
        transformed = transform(image=pil_img)
        transformed_image = transformed["image"]
        
        pil_img = Image.fromarray(np.uint8(transformed_image))
        img_ndarray = np.asarray(pil_img)
        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray 
        
    
    
    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
            
        # transform = torch.nn.Sequential(
            # transforms.RandomRotation(30, expand=False, fill=0, resample=None),
            # transforms.CenterCrop((512,512)),
            # transforms.Normalize((0.8263105, 0.6212347, 0.7698729), (0.15582965, 0.268009, 0.1718304)),
            # transforms.RandomAdjustSharpness(1.2, p=0.5),
            # transforms.ToTensor(),
            # )
        
        # scripted_transforms = torch.jit.script(transform)
        img,mask = self.dataAug(img,mask)
        
        
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        
        
        

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }





class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
