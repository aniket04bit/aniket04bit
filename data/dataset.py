import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import random
from utils.transform import transform
from utils.postprocess import BaseRecLabelDecode

try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False


class SimpleDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(SimpleDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", [1.0])
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        self.seed = seed
        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines,
                                        round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_idx_order_list)
        return

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            outs = transform(data, self.ops)
        except Exception as e:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, e))
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = np.random.randint(self.__len__()) if self.mode == "train" else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)


class LMDBDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(LMDBDataSet, self).__init__()
        if not LMDB_AVAILABLE:
            raise ImportError("LMDB is required for LMDBDataSet. Install with: pip install lmdb")
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        batch_size = loader_config['batch_size_per_card']
        data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir)
        logger.info("Initialize indexs of datasets:%s" % data_dir)
        self.data_idx_order_list = self.dataset_traversal()
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()
        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 2)

        ratio_list = dataset_config.get("ratio_list", 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * len(self.lmdb_sets)

        assert len(ratio_list) == len(self.lmdb_sets)

        self.need_reset = True in [x < 1.0 for x in ratio_list]

    def load_hierarchical_lmdb_dataset(self, data_dir):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, dirnames, filenames in os.walk(data_dir + '/'):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)
                txn = env.begin(write=False)
                num_samples = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {"dirpath":dirpath, "env":env, 
                                         "txn":txn, "num_samples":num_samples}
                dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = end_idx
        return data_idx_order_list

    def get_img_data(self, value):
        """get_img_data"""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype='uint8')
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        load_data_ops = self.ops[:self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            file_idx = self.data_idx_order_list[np.random.randint(self.__len__())]
            lmdb_idx, file_idx = int(file_idx[0]), int(file_idx[1])
            lmdb_idx = file_idx % len(self.lmdb_sets)
            txn = self.lmdb_sets[lmdb_idx]['txn']
            label_key = 'label-%09d'.encode() % file_idx
            label = txn.get(label_key)
            if label is None:
                continue
            label = label.decode('utf-8')
            img_key = 'image-%09d'.encode() % file_idx
            imgbuf = txn.get(img_key)
            img = self.get_img_data(imgbuf)
            if img is None:
                continue

            data = {'image': img, 'label': label}
            outs = transform(data, load_data_ops)
            if outs is None:
                continue
            ext_data.append(outs)
        return ext_data

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        lmdb_idx, file_idx = int(file_idx[0]), int(file_idx[1])
        txn = self.lmdb_sets[lmdb_idx]['txn']
        label_key = 'label-%09d'.encode() % file_idx
        label = txn.get(label_key)
        if label is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % file_idx
        imgbuf = txn.get(img_key)
        img = self.get_img_data(imgbuf)
        if img is None:
            return self.__getitem__(np.random.randint(self.__len__()))

        data = {'image': img, 'label': label}
        outs = transform(data, self.ops)
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]


class RecDataset(Dataset):
    """
    Simple text recognition dataset for PyTorch.
    """
    def __init__(self, data_dir, label_file, img_size=(48, 320), 
                 character_dict_path=None, transforms=None, mode='train'):
        self.data_dir = data_dir
        self.img_size = img_size
        self.mode = mode
        
        # Load character dictionary
        self.label_decoder = BaseRecLabelDecode(character_dict_path)
        
        # Load data list
        self.data_list = []
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    img_path, label = line.split('\t', 1)
                else:
                    # For simple format: image_name.jpg label
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        img_path, label = parts
                    else:
                        continue
                
                full_img_path = os.path.join(data_dir, img_path)
                if os.path.exists(full_img_path):
                    self.data_list.append((full_img_path, label))
        
        self.transforms = transforms
        print(f"Loaded {len(self.data_list)} samples for {mode}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        
        # Load image
        try:
            image = cv2.imread(img_path)
            if image is None:
                image = np.array(Image.open(img_path).convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy sample
            image = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            label = ""
        
        # Encode label
        encoded_label = self.label_decoder.encode(label)
        if encoded_label is None:
            encoded_label = []
        
        # Prepare data
        data = {
            'image': image,
            'label': encoded_label,
            'label_str': label
        }
        
        # Apply transforms
        if self.transforms:
            data = transform(data, self.transforms)
            if data is None:
                # Return dummy data if transform fails
                return self.__getitem__((idx + 1) % len(self.data_list))
        
        return data


def create_dataset(config, mode='train'):
    """Create dataset based on config."""
    dataset_config = config[mode]['dataset']
    dataset_name = dataset_config['name']
    
    if dataset_name == 'SimpleDataSet':
        dataset = SimpleDataSet(config, mode, None)
    elif dataset_name == 'LMDBDataSet':
        dataset = LMDBDataSet(config, mode, None)
    elif dataset_name == 'RecDataset':
        dataset = RecDataset(**dataset_config)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    
    return dataset


def collate_fn(batch):
    """Custom collate function for text recognition."""
    images = []
    labels = []
    label_lengths = []
    
    for sample in batch:
        if isinstance(sample, (list, tuple)):
            image, label = sample[0], sample[1]
        else:
            image, label = sample['image'], sample['label']
        
        images.append(torch.tensor(image, dtype=torch.float32))
        if isinstance(label, (list, np.ndarray)):
            labels.extend(label)
            label_lengths.append(len(label))
        else:
            # Handle case where label is already processed
            labels.append(label)
            label_lengths.append(1)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Convert labels to tensor
    if len(labels) > 0 and isinstance(labels[0], (int, np.integer)):
        labels = torch.tensor(labels, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    
    return images, labels, label_lengths