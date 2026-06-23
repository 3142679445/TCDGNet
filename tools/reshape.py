import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

def dilate_cracks(batch_gt_imgs, kernel_size=3, device='cpu'):

    dilation_kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device)


    padding = kernel_size // 2
    batch_gt_imgs = batch_gt_imgs.float().to(device)
    dilated = F.conv2d(batch_gt_imgs, dilation_kernel, padding=padding)
    dilated = (dilated > 0).float()

    return dilated


def load_binary_image(path):
    image = Image.open(path).convert('L')  
    transform = transforms.Compose([
        transforms.ToTensor(),       
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    return transform(image).unsqueeze(0) 


img1 = load_binary_image("/mnt/dataset/ldw/datasets/crack260/val_m/DSCN6432.png")
img2 = load_binary_image("/mnt/dataset/ldw/datasets/crack260/val_m/DSCN6434.png")
batch = torch.cat([img1, img2], dim=0)  # [2, 1, H, W]

dilated_batch = dilate_cracks(batch, kernel_size=3)


to_pil = transforms.ToPILImage()
for i, dilated_img in enumerate(dilated_batch):
    to_pil(dilated_img.squeeze(0)).save(f"/home/jj/cg/dilated_crack_{i+1}.png")
