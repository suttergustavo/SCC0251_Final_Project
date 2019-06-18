import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CelebADataset(Dataset):
  def __init__(self, root_dir, factor, n_samples, interpolation=2):
    self.root_dir = root_dir
    self.n_samples = n_samples
    self.factor = factor
    self.interpolation = interpolation
    
  def __len__(self):
    return self.n_samples
  
  def __getitem__(self, idx):
    img_name = os.path.join(self.root_dir, f'{idx:06d}.jpg')
    
    original_image = Image.open(img_name)
    
    img_shape = list(original_image.size)
    img_shape[0], img_shape[1] = img_shape[1], img_shape[0]
    
    tsfrm = transforms.Compose([
        transforms.Resize((int(img_shape[0] / self.factor), int(img_shape[1] / self.factor)), interpolation=self.interpolation),
        transforms.Resize(img_shape, interpolation=2),
        transforms.ToTensor()
    ])
    
    degrated_image = tsfrm(original_image)
    original_image = transforms.ToTensor()(original_image)
    
    return degrated_image, original_image