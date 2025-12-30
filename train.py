import logging
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from pytorch_lightning.tuner import Tuner
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    logger.info(f"Experiments are stored in {cfg.output_dir}")
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    datamodule = instantiate(cfg.datamodule.pl_module, logger=logger)

    model = instantiate(cfg.model.pl_module)
    logger.info(model)

    callbacks = instantiate(cfg.callbacks)
    trainer = pl.Trainer(
        callbacks=callbacks,
        **cfg.trainer
    )
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(
    #                     model,
    #                     datamodule=datamodule,
    #                     min_lr=1e-6,
    #                     max_lr=1e-1,
    #                     num_training=1000,  # 只跑 200 个 batch
    #                     mode='exponential',  # 推荐 exponential（默认）
    #                     early_stop_threshold=None  # 防止提前停（轨迹预测 loss 可能波动大）
    #                     )

    # suggested_lr = lr_finder.suggestion()
    # print(f"✅ Suggested LR: {suggested_lr:.2e}")

    # # 可视化
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig("lr_finder.png")
    # fig.show()  # 或 fig.savefig("lr_finder.png")

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.checkpoint)
    trainer.validate(model, datamodule.val_dataloader())


if __name__ == "__main__":
    main()
