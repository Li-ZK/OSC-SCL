import pytorch_lightning as pl

from utils.dataLoader import getDataLoader
from utils.utils import getDatasetInfo

from model.OSC_SCL import get_model, parse_args
from utils.draw import drawPredictionMap

if __name__ == '__main__':

    args = parse_args()
    data_info = getDatasetInfo(args.dataset)
    data_loader: dict = getDataLoader(args, data_info)

    model = get_model(args, data_info)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=[0,], # gpu index
        enable_checkpointing=False
    )

    trainer.fit(
        model,
        train_dataloaders=data_loader['known']['train'],
    )

    anchor = model.getMeanAnchor(data_loader['known']['train'])
    model.setAnchor(anchor)

    trainer.test(model, data_loader['unknown']['test'])

    # draw classification map
    prediction = trainer.predict(model, data_loader['all'])
    drawPredictionMap(prediction, args, data_info, draw_background=False)
