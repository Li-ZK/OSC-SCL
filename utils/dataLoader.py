from torch.utils.data import Subset, DataLoader

from utils.my_dataset import AllDataset, LabelDataset
from utils.split_data import initDataset

def getDataLoader(args, info):
    dataset: dict = initDataset(args, info)

    all_dataset = AllDataset(dataset['data'], args, info)
    label_dataset = LabelDataset(dataset['data'], dataset['gt'], args, info)

    known_train_dataset = Subset(label_dataset, dataset['known_train_index'])
    known_test_dataset = Subset(label_dataset, dataset['known_test_index'])
    unknown_test_dataset = Subset(label_dataset, dataset['unknown_test_index'])
    unknown_unknown_dataset = Subset(label_dataset, dataset['unknown_unknown_index'])

    return {
        'all': DataLoader(all_dataset, batch_size=args.batch, shuffle=False),
        'known': {
            'train': DataLoader(known_train_dataset, batch_size=args.batch, shuffle=True),
            'test': DataLoader(known_test_dataset, batch_size=args.batch, shuffle=True)
        },
        'unknown': {
            'test': DataLoader(unknown_test_dataset, batch_size=args.batch, shuffle=True),
            'unknown': DataLoader(unknown_unknown_dataset, batch_size=args.batch, shuffle=True)
        }
    }