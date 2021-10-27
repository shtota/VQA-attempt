import h5py
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models

import torch.utils.data as data
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
from parameters import DEVICE, DATA_FOLDER, image_path, PREPROCESSED_FILE_PATH

class CocoImages(data.Dataset):
    """ Dataset for MSCOCO images located in a folder on the filesystem """
    def __init__(self, path, transform=None):
        super(CocoImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)

class Composite(data.Dataset):
    """ Dataset that is a composite of several Dataset objects. Useful for combining splits of a dataset. """
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        current = self.datasets[0]
        for d in self.datasets:
            if item < len(d):
                return d[item]
            item -= len(d)
        else:
            raise IndexError('Index too large for composite dataset')

    def __len__(self):
        return sum(map(len, self.datasets))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer


def create_coco_loader(*paths):
    transform = transforms.Compose([
                    transforms.Resize(448),
                    transforms.CenterCrop(448),
                    transforms.ToTensor(),
                    transforms.Normalize(  # [5]
                        mean=[0.485, 0.456, 0.406],  # [6]
                        std=[0.229, 0.224, 0.225]  # [7]
                    )])

    datasets = [CocoImages(path, transform=transform) for path in paths]
    dataset = Composite(*datasets)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader


def preprocess_images():
    cudnn.benchmark = True
    if os.path.exists(PREPROCESSED_FILE_PATH):
        os.remove(PREPROCESSED_FILE_PATH)
    net = Net().to(DEVICE)
    net.eval()

    loader = create_coco_loader(image_path(None, True), image_path(None, False))
    features_shape = (
        len(loader.dataset),
        2048,
        14,
        14
    )

    with h5py.File(PREPROCESSED_FILE_PATH, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        coco_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')

        i = 0
        j = 0
        for ids, imgs in tqdm(loader):
            imgs = imgs.to(DEVICE)
            out = net(imgs)

            j = i + imgs.size(0)
            features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
            coco_ids[i:j] = ids.numpy().astype('int32')
            i = j


if __name__ == '__main__':
    preprocess_images()
