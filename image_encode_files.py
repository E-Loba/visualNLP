#import cv2
import torch
import torchvision

import numpy as np

import kornia

from torch import nn
from PIL import Image
from os import listdir
from os.path import join
from random import randrange
from functools import partial
from numpy.random import randint
#from timm.utils import ModelEma
#from SiT.engine import distortImages
from timm.models import create_model
from timm.models.registry import register_model
from SiT.vision_transformer_SiT import VisionTransformer_SiT
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def drop_rand_patches(X, X_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    #######################
    # X_rep: replace X with patches from X_rep. If X_rep is None, replace the patches with Noise
    # max_drop: percentage of image to be dropped
    # max_block_sz: percentage of the maximum block to be dropped
    # tolr: minimum size of the block in terms of percentage of the image size
    #######################
    
    C, H, W = X.size()
    n_drop_pix = np.random.uniform(0, max_drop)*H*W
    mx_blk_height = int(H*max_block_sz)
    mx_blk_width = int(W*max_block_sz)
    
    tolr = (int(tolr*H), int(tolr*W))
    
    total_pix = 0
    while total_pix < n_drop_pix:
        
        # get a random block by selecting a random row, column, width, height
        rnd_r = randint(0, H-tolr[0])
        rnd_c = randint(0, W-tolr[1])
        rnd_h = min(randint(tolr[0], mx_blk_height)+rnd_r, H) #rnd_r is alread added - this is not height anymore
        rnd_w = min(randint(tolr[1], mx_blk_width)+rnd_c, W)
        
        if X_rep is None:
            X[:, rnd_r:rnd_h, rnd_c:rnd_w] = torch.empty((C, rnd_h-rnd_r, rnd_w-rnd_c), dtype=X.dtype, device='cuda').normal_()
        else:
            X[:, rnd_r:rnd_h, rnd_c:rnd_w] = X_rep[:, rnd_r:rnd_h, rnd_c:rnd_w]    
         
        total_pix = total_pix + (rnd_h-rnd_r)*(rnd_w-rnd_c)

    return X

def rgb2gray_patch(X, tolr=0.05):

    C, H, W = X.size()
    tolr = (int(tolr*H), int(tolr*W))
     
    # get a random block by selecting a random row, column, width, height
    rnd_r = randint(0, H-tolr[0])
    rnd_c = randint(0, W-tolr[1])
    rnd_h = min(randint(tolr[0], H)+rnd_r, H) #rnd_r is alread added - this is not height anymore
    rnd_w = min(randint(tolr[1], W)+rnd_c, W)
    
    X[:, rnd_r:rnd_h, rnd_c:rnd_w] = torch.mean(X[:, rnd_r:rnd_h, rnd_c:rnd_w], dim=0).unsqueeze(0).repeat(C, 1, 1)

    return X
    

def smooth_patch(X, max_kernSz=15, gauss=5, tolr=0.05):

    #get a random kernel size (odd number)
    kernSz = 2*(randint(3, max_kernSz+1)//2)+1
    gausFct = np.random.rand()*gauss + 0.1 # generate a real number between 0.1 and gauss+0.1
    
    C, H, W = X.size()
    tolr = (int(tolr*H), int(tolr*W))
     
    # get a random block by selecting a random row, column, width, height
    rnd_r = randint(0, H-tolr[0])
    rnd_c = randint(0, W-tolr[1])
    rnd_h = min(randint(tolr[0], H)+rnd_r, H) #rnd_r is alread added - this is not height anymore
    rnd_w = min(randint(tolr[1], W)+rnd_c, W)
    
    
    gauss = kornia.filters.GaussianBlur2d((kernSz, kernSz), (gausFct, gausFct))
    X[:, rnd_r:rnd_h, rnd_c:rnd_w] = gauss(X[:, rnd_r:rnd_h, rnd_c:rnd_w].unsqueeze(0))
    
    return X


def distortImages(samples):
    n_imgs = samples.size()[0] #this is batch size, but in case bad inistance happened while loading
    samples_aug = samples.detach().clone()
    for i in range(n_imgs):

        samples_aug[i] = rgb2gray_patch(samples_aug[i])

        samples_aug[i] = smooth_patch(samples_aug[i])

        samples_aug[i] = drop_rand_patches(samples_aug[i])

        idx_rnd = randint(0, n_imgs)
        if idx_rnd != i:
            samples_aug[i] = drop_rand_patches(samples_aug[i], samples_aug[idx_rnd])
      
    return samples_aug


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


@register_model
def SiT_base(pretrained=False, **kwargs):
    if "patch_size" not in kwargs:
        kwargs["patch_size"] = 16
    if "img_size" not in kwargs:
        kwargs["img_size"] = 224
    model = VisionTransformer_SiT(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model



def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop


@torch.no_grad()
def do_things():
    data_path = "/home/egor/Documents/python_codes/visual7w/visual7w_images/images"
    # "/home/egor/Pictures/visual7w"
    # "/home/egor/Documents/python_codes/visual7w/visual7w_images/images"

    checkpoint_path = "/home/egor/Documents/python_codes/visual7w/SiT/checkpoints/finetune/v7w/checkpoint.pth"  # finetuned on v7w
    # "/home/egor/Documents/python_codes/visual7w/SiT/checkpoints/finetune/CIFAR10_LE/checkpoint.pth"  # raw
    # "/home/egor/Documents/python_codes/visual7w/checkpoints/checkpoint4.pth"  # pretrained
    img_size_ = 448
    patch_size_ = 16
    model = create_model('SiT_base',
        pretrained=False,
        img_size=img_size_, patch_size=patch_size_, num_classes=0, 
        drop_rate=0.0, drop_path_rate=0.1,
        drop_block_rate=None, training_mode="SSL", representation_size=768)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    # for k in ['rot_head.weight', 'rot_head.bias', 'contrastive_head.weight', 'contrastive_head.bias']:
    #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #         print(f"Removing key {k} from pretrained checkpoint")
    #         del checkpoint_model[k]
    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed
    
    model.load_state_dict(checkpoint_model, strict=False)
    model.to("cpu")
    # model_ema = ModelEma(model, decay=0.99996,device='cpu', resume='')
    model.eval()
    requires_grad(model, False)
    model.rot_head.weight.requires_grad = True
    model.rot_head.bias.requires_grad = True
    model.contrastive_head.weight.requires_grad = True
    model.contrastive_head.bias.requires_grad = True
    model.pre_logits_rot.fc.weight.requires_grad = True
    model.pre_logits_rot.fc.bias.requires_grad = True
    model.pre_logits_contrastive.fc.weight.requires_grad = True
    model.pre_logits_contrastive.fc.bias.requires_grad = True

    t = []
    # size = int((256 / img_size_) * img_size_)
    t.append(
        torchvision.transforms.Resize(size=(img_size_, img_size_)),  # to maintain same ratio w.r.t. 224 images
    )
    #t.append(torchvision.transforms.CenterCrop(img_size_))
    t.append(torchvision.transforms.ToTensor())
    t.append(torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    t_ = torchvision.transforms.Compose(t)

    images = listdir(data_path)
    with torch.no_grad():
        for image_ in images[:20]:
            image = Image.open(join(data_path, image_))
            input_ = t_(image)
            input_ = input_[None, :, :, :]
            #input_ = distortImages(input_)

            out_raw, out_attn = model(input_, attn=True)
            print(out_attn.shape)

            print_out = join("/home/egor/Documents/python_codes/visual7w/testing_images", image_)
            torchvision.utils.save_image(out_raw, print_out, nrow=1, normalize=True, range=(-1, 1))
            print_out = join("/home/egor/Documents/python_codes/visual7w/testing_images", "distort_"+image_)
            torchvision.utils.save_image(input_, print_out, nrow=1, normalize=True, range=(-1, 1))
    return


if __name__ == "__main__":
    do_things()
