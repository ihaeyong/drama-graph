from torchvision.datasets.vision import VisionDataset # For custom usage # YDK
# from .vision import VisionDataset

from PIL import Image

import os, sys
import os.path

import torch # YDK
import json # YDK

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    ###########################################################################
    # YDK
    for target in sorted(os.listdir(directory)):
        d = os.path.join(directory, target)
        if not os.path.isdir(d):
            continue
        for root in sorted(os.listdir(d)):
            instances.append(os.path.join(d, root))
    ###########################################################################
    # for target_class in sorted(class_to_idx.keys()):
    #     class_index = class_to_idx[target_class]
    #     target_dir = os.path.join(directory, target_class)
    #     if not os.path.isdir(target_dir):
    #         continue
    #     for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
    #         for fname in sorted(fnames):
    #             path = os.path.join(root, fname)
    #             if is_valid_file(path):
    #                 item = path, class_index
    #                 instances.append(item)
    return instances

def dataset_len(train=True):
    read = 0
    path_temp = []
    label_temp = []
    for target in sorted(os.listdir('reader/')):
        
        temp = os.path.join('reader', target)
        read += 1
        if train:
            if read < 17:
                tx = open(temp, "r")
                while True:
                    line = tx.readline()
                    if not line:
                        break
                    path_ = line.split(' ')[0]; label_ = line.split(' ')[1]
                    path_temp.append(path_)
                    label_temp.append(label_)
        else:
            if read >= 17:
                tx = open(temp, "r")
                while True:
                    line = tx.readline()
                    if not line:
                        break
                    path_ = line.split(' ')[0]; label_ = line.split(' ')[1]
                    path_temp.append(path_)
                    label_temp.append(label_)
    return len(path_temp)



class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, json_label_file=None, file_length=10, train=True): # YDK
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx, id_class = self._find_classes(self.root, json_label_file) # YDK
        self.train = train

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.id_class = id_class # YDK
        self.file_length = file_length

        self._create_reader(root) # HSE

        self.data_len = dataset_len(train=self.train)
        print(self.data_len)
        

    def _find_classes(self, dir, json_label_file): # YDK
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        #################################################
        # YDK
        classes = []
        instances = []
        with open(json_label_file) as f:
            d = json.load(f)
            for file in d:
                for c in d[file]: 
                    item = c['frame_id'], c['place']
                    classes.append(c['place'])
                    instances.append(item)
        classes = sorted(list(set(classes)))
        class_to_idx = {cls_name: i-1 for i, cls_name in enumerate(classes)}
        class_to_idx[''] = 9
        id_class = {i: class_to_idx[j] for i, j in sorted(instances)} # dict # YDK
        # list vs dict: index is either integer or string
        # list: list[0] = ...
        # dict: dict['first'] = ...
        #################################################
        # classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        # classes.sort()
        # class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx, id_class

    def _create_reader(self, directory):
        if os.path.exists('reader/'):
            return

        os.makedirs('reader/')
        for target in sorted(os.listdir(directory)):
            print("Json file Processing on ", target)
            temp = "reader/" + target + ".txt"
            txt_file = open(temp, "w")
            d = os.path.join(directory, target)  #AnotherMissOh01
            for fold1 in sorted(os.listdir(d)):
                d2 = os.path.join(d, fold1) #001
                for fold2 in sorted(os.listdir(d2)):
                    d3 = os.path.join(d2, fold2) #0078 
                    for im in sorted(os.listdir(d3)):
                        final = os.path.join(d3, im) #Image root
                        if has_file_allowed_extension(final, IMG_EXTENSIONS):
                            # print(final[-45:-4])
                            j = final[-45:-4].replace('/', '_') # HSE
                        
                        try:
                            labeling = self.id_class[j]
                            data = str(final) + ' ' + str(labeling) + '\n'
                            txt_file.write(data)
                        except:
                            continue

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        #################################################
        # YDK
        read = 0
        path_fnames = []
        label_list = []
        for target in sorted(os.listdir('reader/')):
            path_temp = []
            label_temp = []
            temp = os.path.join('reader', target)
            read += 1
            if self.train:
                if read < 17:
                    tx = open(temp, "r")
                    while True:
                        line = tx.readline()
                        if not line:
                            break
                        path_ = line.split(' ')[0]; label_ = int(line.split(' ')[1])
                        path_fnames.append(path_)
                        label_list.append(label_)
            else:
                if read >= 17:
                    tx = open(temp, "r")
                    while True:
                        line = tx.readline()
                        if not line:
                            break
                        path_ = line.split(' ')[0]; label_ = int(line.split(' ')[1])
                        path_fnames.append(path_)
                        label_list.append(label_)

        sample_grp = []
        target_grp = []
        fname_grp  = []
        for i in range(index, index + self.file_length):
            try:
                pf = path_fnames[i]
            except:
                pf = path_fnames[-1]
            sample = self.loader(pf)
            if self.transform is not None:
                sample = self.transform(sample)
            sample_grp.append(sample.unsqueeze(0))
            try:
                target_grp.append(label_list[i])
            except:
                target_grp.append(label_list[-1])

        sample = torch.cat(sample_grp, dim=0)
        target = torch.Tensor(target_grp)
        # print(sample.size()) # For debug
        # print(target) # For debug
        # for i in range(len(fname_grp)): print(fname_grp[i]) # For debug
        #################################################
        # path, target = self.samples[index]
        # sample = self.loader(path)
        # if self.transform not None:
        #     sample = self.transform(sample)
        # if self.target_transform not None:
        #     target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return self.data_len


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, json_label_file=None, file_length=10, train=True): # YDK
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, json_label_file=json_label_file, file_length=file_length, train=True)
