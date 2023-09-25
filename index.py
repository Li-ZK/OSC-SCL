import pytorch_lightning as pl

from utils.dataLoader import getDataLoader
from utils.utils import getDatasetInfo, seed_torch

from model.OSC_SCL import get_model, parse_args

if __name__ == '__main__':

    args = parse_args()
    seed_torch(args.seed)
    data_info = getDatasetInfo(args.dataset)
    data_loader: dict = getDataLoader(args, data_info)

    model = get_model(args, data_info)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=[1,],
        enable_checkpointing=False
    )

    trainer.fit(
        model,
        train_dataloaders=data_loader['known']['train'],
    )

    anchor = model.getMeanAnchor(data_loader['known']['train'])
    model.setAnchor(anchor)

    trainer.test(model, data_loader['unknown']['test'])