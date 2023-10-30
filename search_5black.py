import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import cv2
import os
from torch.utils.data import DataLoader
import torch.utils.data as data
import math
from tqdm import tqdm
import shutil
import glob
from PIL import Image
import torchvision.transforms as transforms
# from utils_patch_full_pic import PatchApplier, ParticleToPatch
import random


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    
    def ret(input):
        losses = []
        for i in range(0, input.shape[0], chunk):
            loss = fn(input[i:i+chunk])
            losses.append(loss)
        return np.concatenate(losses, axis=0)

    return ret



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.Bmp'
]


def is_image_file(filename, key):
    return any(filename.endswith(extension) for extension in key)


def make_dataset(dir, key):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname, key):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):
    def __init__(self, root, background_root, return_path=False, aug_bounds=[0.1, 0.2]):
        self.imgs = make_dataset(root, ['.npy'])
        self.bkgs = make_dataset(background_root, IMG_EXTENSIONS)

        self.background_root = background_root
        if len(self.imgs) == 0 :
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        
        self.root = root
        self.return_path = return_path
        self.tran = transforms.ToTensor()
        self.aug_bounds = aug_bounds

    def __getitem__(self, index):

        path = self.imgs[index]
        bkg_index = random.randint(0, len(self.bkgs) - 1)

        mask = torch.tensor(np.load(self.imgs[index]))
        bkg = self.tran(Image.open(self.bkgs[bkg_index]).convert('L')).squeeze()

        turb = self.aug_bounds[0] + random.random() * (self.aug_bounds[1] - self.aug_bounds[0])
        bkg = bkg * turb
        
        w, h = mask.shape
        W, H = bkg.shape

        # print(mask.shape)
        # print(bkg.shape)
        # assert False

        w_start, h_start = int((W - w) * random.random()), int((H - h) * random.random())
        bkg = bkg[w_start:(w_start+w), h_start:(h_start+h)]

        # print(mask.shape)
        # print(bkg.shape)
        # assert False
        bkg[mask!=0]=0   

        # im = transforms.ToPILImage()(bkg)
        # im.save("patch1.png")
        # assert False

        if self.return_path:
            return mask, bkg, path
        else:
            return mask, bkg

    def __len__(self):
        return len(self.imgs)


def get_data_loaders(root, background_root, batch_size, aug_upper, aug_bottom):
    dataset = ImageFolder(root, background_root, aug_bounds=[aug_upper, aug_bottom])
    print('The length of the dataset is:', len(dataset))
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, pin_memory=False, shuffle=True
    )
    return loader


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class GA:
    def __init__(self, shape_dim, size, iter_num, save_every_iter, p_cross, p_mutation, train_loader, save_root):
        self.shape_dim = shape_dim
        self.size = size                
        self.iter_num = iter_num        
        self.save_every_iter = save_every_iter
        self.p_cross = p_cross          
        self.p_mutation = p_mutation

        self.mapping = [3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

        assert self.shape_dim == len(self.mapping)

        self.lst = []
        self.best_shape = []
        self.best_confidence = []

        self.train_loader = train_loader

        self.shape = np.stack([np.random.randint(0, 2, self.shape_dim) for _ in range(self.size)]).astype(np.float32)

        self.fitness = np.zeros(self.size)

        # Model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cuda:2')  # or yolov5n - yolov5x6, custom
        self.save_root = save_root
        self.log_dir = os.path.join(save_root, 'log.txt')

        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)

    def one_point_crossover(self):
        for _ in range(self.shape_dim):
            if random.random() < self.p_cross:
                chichi_index, haha_index = random.sample([i for i in range(self.size)], 2)
                chichi = self.shape[chichi_index]
                haha = self.shape[haha_index]
                index = random.randint(0, self.shape_dim - 1)
                aijinn = chichi[index:].copy()
                chichi[index:] = haha[index:]
                haha[index:] = aijinn

    def mutation(self):
        for i in range(self.size):
            for j in range(self.shape_dim):
                if random.random() < self.p_mutation:
                    self.shape[i][j] = 1 - self.shape[i][j]

    def roulette_selection(self):
        selected_index = np.random.choice(
            np.arange(self.size), size=self.size, replace=True, p=self.fitness)
        self.shape = self.shape[selected_index]
    
    def calculate_yolo(self, imgs):
        """
            imgs: [number, batch_size, w, h]
        """
        imgs = [transforms.ToPILImage()(imgs[i]) for i in range(imgs.shape[0])]
        with torch.no_grad():
            results = self.model(imgs)
        results = results.xyxy
        losses = []
        for i in range(len(results)):
            confidence = torch.tensor(0.).cuda()
            temp_result = results[i]
            for j in range(temp_result.shape[0]):
                if temp_result[j][-1] == 0:
                    confidence = temp_result[j][-2]
                    break
            losses.append(confidence.cpu().numpy())
        losses = np.stack(losses)
        return losses
    

    def f(self, masks, bkgs):
        """
            masks: [batch_size, 640, 320]
            bkgs: [batch_size, 3, 516, 692]
        """

        b, w, h = masks.shape
        numbers, dim = self.shape.shape

        losses = []

        for i in tqdm(range(numbers)):
            temp_shape = self.shape[i]
            temp_bkg = bkgs
            for j in range(self.shape_dim):
                temp_bkg = torch.where(masks==self.mapping[j], torch.tensor(temp_shape[j]).cuda(), temp_bkg)
            temp_bkg = temp_bkg.unsqueeze(1)
            loss = self.calculate_yolo(temp_bkg)
            losses.append(np.mean(loss))
        losses = np.array(losses)
        return losses

    def get_fitness(self, masks, bkgs):
        val = self.f(masks, bkgs)
        fitness = (np.max(val) - val) / (np.max(val) - np.min(val) + 1e-3) + 1e-3
        fitness = np.power(fitness, 10)
        fitness = fitness / sum(fitness)
        self.fitness = fitness
        
        return fitness, val
    
    def main(self):
        for iter in tqdm(range(self.iter_num)):
            masks, bkgs = next(self.train_loader)
            masks, bkgs = masks.cuda(), bkgs.cuda()
            self.one_point_crossover()
            self.mutation()
            temp_val, loss = self.get_fitness(masks, bkgs)
            self.lst.append(temp_val)
            max_index = np.argmax(temp_val)
            self.best_shape.append(self.shape[max_index])
            self.best_confidence.append(loss[max_index])

            sample_masks, sample_bkgs = masks[0], bkgs[0]
            temp_best_shape = self.best_shape[-1]

            for j in range(self.shape_dim):
                sample_bkgs = torch.where(sample_masks==self.mapping[j], torch.tensor(temp_best_shape[j]).cuda(), sample_bkgs)
            
            sample_board = torch.zeros(3, 640, 320)
            for i in range(self.shape_dim):
                sample_board[:, (640//self.shape_dim)*i:(640//self.shape_dim)*(i+1)] = torch.tensor(temp_best_shape[i])

            save_to = os.path.join(self.save_root, 'pics', f'{iter}_pic.png')
            save_board_to = os.path.join(self.save_root, 'pics', f'{iter}_board.png')
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            transforms.ToPILImage()(sample_bkgs).save(save_to)
            transforms.ToPILImage()(sample_board).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(save_board_to)

            self.roulette_selection()
            
            with open(self.log_dir, 'a+') as fp:
                fp.write(f'{self.best_confidence[-1]}\n')

            if iter % self.save_every_iter == 0:
                os.makedirs(os.path.join(self.save_root, f'best'), exist_ok=True)
                np.save(os.path.join(self.save_root, f'best/{iter}_shape'), self.best_shape)

        return self.lst, self.best_shape
    


if __name__ == '__main__':

    root = './rendered_2d_maps/train'
    background_root = './backgrounds'

    aug_bottom = 1
    aug_upper = 1
    Size = 1000             # 個体数
    batch_size = 300
    Iter_num = 200         # 世代数
    save_every_iter = 1
    P_cross = 0.5         # 交叉確率
    P_mutation = 0.01      # 突然変異確率
    Shape_dim = 26
    save_root = f'EXP/yolos_origin/5black_dim_{Shape_dim}_size{Size}_Pcross{P_cross}_Pmutation{P_mutation}_bs{batch_size}_aug{aug_bottom}_{aug_upper}'
    scripts_to_save = glob.glob('*.py')
    for script in scripts_to_save:
        dst_file = os.path.join(save_root, 'scripts', os.path.basename(script))
        if os.path.exists(dst_file):
            os.remove(dst_file)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copyfile(script, dst_file)

    random.seed(233)
    np.random.seed(233)

    train_loader = sample_data(get_data_loaders(
        root, background_root, batch_size, aug_upper, aug_bottom
    ))

    ga_min = GA(
        shape_dim=Shape_dim, size=Size, iter_num=Iter_num, save_every_iter=save_every_iter, p_cross=P_cross, p_mutation=P_mutation, 
        train_loader=train_loader, save_root=save_root
    )

    lst, best_shape = ga_min.main()

    # 可視化
    # plt.plot(np.linspace(0, Iter_num, Iter_num), val_lst_min, c="r", alpha=0.5)
    # plt.savefig('./GA_min.jpg')
    # ####### 最小化が終わり ####### #
