from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, data, args, info):
        super(BaseDataset, self).__init__()

        self.data = data
        self.patch_size = args.patch

        self.H = self.data.shape[1] - self.patch_size // 2 * 2
        self.W = self.data.shape[2] - self.patch_size // 2 * 2

    def __len__(self):
        return self.H * self.W
    
    def __getitem__(self, index):
        raise NotImplementedError('Not implemented __getitem__')
    
    def get_patch(self, x, y):
        return self.data[:, x : x+self.patch_size, y : y+self.patch_size]

    def parseLocation(self, location):
        x = location // self.W
        y = location % self.W
        return x, y

class AllDataset(BaseDataset):
    def __init__(self, data, args, info):
        super(AllDataset, self).__init__(data, args, info)

    def __getitem__(self, index):
        x, y = self.parseLocation(index)
        patch = self.get_patch(x, y)
        return patch
    
class LabelDataset(BaseDataset):
    def __init__(self, data, gt, args, info):
        super(LabelDataset, self).__init__(data, args, info)

        assert data.shape[1] == (gt.shape[0] + args.patch // 2 * 2) \
            and data.shape[2] == (gt.shape[1] + args.patch // 2 * 2), 'Error: incorrect dataset shape'

        self.gt = gt

    def __getitem__(self, index):
        x, y = self.parseLocation(index)
        patch = self.get_patch(x, y)
        label = self.gt[x, y]
        return patch, label - 1
