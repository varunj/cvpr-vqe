import pandas as pd
import torch
from torch.utils import data
import os
import numpy as np
from hydra.utils import instantiate
import cv2
import logging
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class ImageVQEDataset(data.Dataset):
    def __init__(self, dataframe, root_folder, augmentation_policy=None, h=0, w=0):
        super(ImageVQEDataset, self).__init__()
        self._df = dataframe
        self._root = root_folder
        self._transform = None
        if augmentation_policy is not None:
            self._transform = instantiate(augmentation_policy)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        # read images
        row = self._df.loc[index]
        img_x = cv2.imread(os.path.join(self._root, row['img_source']), cv2.IMREAD_COLOR)
        assert img_x is not None, 'could not load img_x [%s]' % row['img_source']
        img_y = None
        if not pd.isnull(row['img_target']):
            img_y = cv2.imread(os.path.join(self._root, row['img_target']), cv2.IMREAD_COLOR)

        # read tags
        subset = row['subset']
        video_id = row['video_id']

        if self._transform is not None and self._transform._tfm:
            # albumentations.augmentations.transforms expects RGB
            img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
            img_x = self._transform(img_x)
            img_x = cv2.cvtColor(img_x, cv2.COLOR_RGB2BGR)

        return {
            'index': index,
            'img_x': torch.from_numpy(img_x).permute(2, 0, 1),
            'img_y': torch.from_numpy(img_y).permute(2, 0, 1) if img_y is not None else torch.tensor([]),
            'subset': subset,
            'video_id': video_id,
        }


def make_splits(length, max_size, num_groups=None, drop_last=False):
    if drop_last:
        assert num_groups is None, 'number of groups is fixed, when drop_last is True'
        return (max_size,) * (length // max_size)
    else:
        if num_groups is not None:
            assert length >= num_groups, 'can not have less than 1 element in a batch'
            assert max_size * num_groups >= length, 'can not split into groups, max_size * num_groups < length'
            g = [max_size] * num_groups
            extra = max_size * num_groups - length
            while extra > 0:
                g[extra % len(g)] -= 1
                extra -= 1
            return tuple(g)
        else:
            g = [max_size] * (length // max_size)
            if length % max_size > 0:
                g.append(length % max_size)
            return tuple(g)


class DDPGroupedBatchSampler(data.Sampler):
    def __init__(
            self,
            dataframe: pd.DataFrame,
            group_by=None,
            batch_size=1,
            num_replicas=1,
            rank=0,
            shuffle=False,
            seed=0,
            drop_last=False,
            dropna=True):
        super().__init__(dataframe)
        self.log = logging.getLogger('lightning').getChild('sampler_%d_%d' % (rank, num_replicas))
        if rank >= num_replicas or rank < 0:
            raise ValueError('Invalid rank %d, rank should be in the interval [0, %d)' % (rank, num_replicas))

        self.batch_size = batch_size
        self.log.debug('using batches of %d samples per node' % self.batch_size)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.dropna = dropna
        self.shuffle = shuffle
        self.seed = seed

        # groups list contains indexes tensor for each group
        self.groups = [
            torch.from_numpy(ds.index.to_numpy(dtype=np.int32)) for _, ds in dataframe.groupby(group_by, dropna=self.dropna)
        ]
        self.groups.sort(key=len, reverse=True) #sort the groups by the number of images in descending order

        # with drop last we guarantee fixed batch size plus we may drop last batches,
        # if those are not equally distributable between nodes

        # lengths of batches within groups
        self.splits = []
        # number of samples to drop from each group
        self.drops = []
        # number of full batches to drop
        self.num_batches = 0
        if self.drop_last:
            for g in self.groups:
                group_length = len(g)
                self.drops.append(group_length % self.batch_size)
                self.splits.append(make_splits(group_length, self.batch_size, drop_last=self.drop_last))
                self.log.debug('group length %d, batches %s, dropping %d' % (group_length, self.splits[-1], self.drops[-1]))
            total_batches = sum([len(split) for split in self.splits])
            self.log.debug('total number of batches %d' % total_batches)
            drop_batches = total_batches % self.num_replicas
            if not self.shuffle:
                drops = sum(self.drops)
                total_drops = drop_batches * self.batch_size + drops
                if total_drops > 0:
                    self.log.warning('%d full batch(es) and %d individual samples will be left unused (total %d)' % (
                        drop_batches, drops, total_drops
                    ))
            self.num_batches = total_batches - drop_batches
        else:
            for g in self.groups:
                group_length = len(g)
                self.drops.append(0)
                self.splits.append(make_splits(group_length, self.batch_size, drop_last=self.drop_last))
            total_batches = sum([len(split) for split in self.splits])
            if total_batches % self.num_replicas > 0:
                extra_batches = self.num_replicas - (total_batches % self.num_replicas)
                for idx, split in enumerate(self.splits):
                    if len(split) == sum(split):
                        continue
                    num_groups = min(sum(split), len(split) + extra_batches) # try to add as many extra_batches as possible
                    extra_batches -= (num_groups - len(split)) # remaining extra_batches
                    self.splits[idx] = make_splits(sum(split), self.batch_size, num_groups=num_groups, drop_last=self.drop_last)
                    if extra_batches == 0:
                        break
                assert extra_batches == 0, 'could not split, increase batch size or too few batches for gpus, decrease batch size'
                self.num_batches = sum([len(split) for split in self.splits])
            else:
                self.num_batches = total_batches
            for idx, g in enumerate(self.groups):
                group_length = len(g)
                self.log.debug('group length %d, batches %s, dropping %d' % (group_length, self.splits[idx], self.drops[idx]))

            self.log.debug('number of batches %d' % self.num_batches)
            if self.num_batches == 0:
                raise ValueError('Can not sample with these parameters, try to decrease batch size, or set drop_last '
                                 'to false')

    def __iter__(self):
        batches = []
        g = None
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

        for indexes, split, drop in zip(self.groups, self.splits, self.drops):
            if self.shuffle:
                indexes = indexes[torch.randperm(len(indexes), generator=g)]
            batches.extend((t.tolist() for t in torch.split(indexes[drop:], split)))

        batches = batches[:self.num_batches]

        if self.shuffle:
            new_idx = torch.randperm(len(batches), generator=g)
            return iter([batches[idx] for idx in new_idx[self.rank::self.num_replicas]])
        else:
            return iter(batches[self.rank::self.num_replicas])

    def __len__(self) -> int:
        return self.num_batches // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.log.debug('set epoch %d' % epoch)
        self.epoch = epoch
