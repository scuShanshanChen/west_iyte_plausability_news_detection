import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import os
import argparse
import logging

logging.basicConfig(filename='training_kg_embeddings.log')


def train(kg_path: str, dim: int, train_times: int, alpha: float, use_gpu: bool, saved_dir: str, model_name: str):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=kg_path,
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # dataloader for test
    logging.info('Train set is loaded')
    ent_tot = train_dataloader.get_ent_tot()
    rel_tot = train_dataloader.get_rel_tot()

    # define the model
    transe = TransE(
        ent_tot=ent_tot,
        rel_tot=rel_tot,
        dim=dim,
        p_norm=1,
        norm_flag=True)

    logging.info('Model is defined')

    # define the loss function
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=5.0),
        batch_size=train_dataloader.get_batch_size()
    )

    # train the model
    trainer = Trainer(model=model, data_loader=train_dataloader, train_times=train_times, alpha=alpha, use_gpu=use_gpu)
    trainer.run()

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    checkpoint_name = model_name + '_dim_{dim}_ent_{ent_tot}_rel_{rel_tot}.ckpt'.format(dim=dim, ent_tot=ent_tot,
                                                                                        rel_tot=rel_tot)

    logging.info('Model {} is saving ...'.format(checkpoint_name))

    checkpoint_path = os.path.join(saved_dir, checkpoint_name)
    transe.save_checkpoint(checkpoint_path)


def test(model_path, ent_tot: int, rel_tot: int, dim: int, kg_path: str, use_gpu: bool):
    # test the model
    # define the model
    transe = TransE(
        ent_tot=ent_tot,
        rel_tot=rel_tot,
        dim=dim,
        p_norm=1,
        norm_flag=True)
    transe.load_checkpoint(model_path)

    test_dataloader = TestDataLoader(kg_path, "link")
    logging.info('Dev and test are loaded')
    tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=use_gpu)
    tester.run_link_prediction(type_constrain=False)

#ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A module for converting KGs into OpenKE format')
    parser.add_argument('--kg_path', type=str, help='Directory locating KG')
    parser.add_argument('--saved_dir', type=str, help='Directory for saving OpenKE files')
    parser.add_argument('--dim', type=int, help='Enter embedding size')
    parser.add_argument('--train_times', type=int, help='Enter train times')
    parser.add_argument('--alpha', type=float, help='Learning rate alpha')
    parser.add_argument('--use_gpu', type=str2bool)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()

    if 'train' == args.mode:
        train(args.kg_path, args.dim, args.train_times, args.alpha, args.use_gpu, args.saved_dir, args.model_name)
    elif 'test' == args.mode:
        test(args.model_path, args.ent_tot, args.rel_tot, args.dim, args.kg_path, args.use_gpu)
