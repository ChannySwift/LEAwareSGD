from main import main as train
import argparse
import random

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(args):
    save_path = args.save_path

    EX_args = Namespace(
        seed=random.randint(0, 100000), #random.randint(0, 100000)
        dataset=args.dataset,
        algorithm=args.algorithm,
        model=args.network, #resnext29,WideResNet,alexnet,resnet18,resnet34,resnet50,resnet101,resnet152
        batch_size=args.batch_size,
        num_classes=args.num_class,
        seen_index=args.seen_index,
        train_epochs=args.train_epochs,
        loops_min=100,  # number of batches per epoch
        loops_adv=50,
        lr=0.0001,
        lr_max=5.0,
        momentum=0.9,
        le_weight=0.1,
        weight_decay=0.0005,
        path=save_path,
        deterministic=True,
        k=3,
        gamma=1,
        eta=10,  # parameter for the regularizer in the maximization procedure
        eta_min=0.01,  # parameter for the entropy regularizer in the minimization procedure
        beta=1.0,  # paramter for the contrastive loss regularizer
        num_workers=8,
        train_mode="contrastive",
        tag="",
        gen_freq=1,
        domain_number=100,
        ratio=1.0,  # select how much training data to use
    )


    expr_args = EX_args
    train(expr_args)


if __name__ == "__main__":
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--seed", type=int, default=1, help="")
    train_arg_parser.add_argument(
        "--save_path",
        type=str,
        default="pacs_experiments/pacs_AdvST_test",
        help="path to saved models and results",
    )
    train_arg_parser.add_argument(
        "--seen_index",
        type=int,
        default=3,
        help="seen domain index",
    )
    train_arg_parser.add_argument(
        "--dataset",
        type=str,
        default="pacs",
        help="seen dataset name",
    )
    train_arg_parser.add_argument(
        "--network",
        type=str,
        default="resnet18",
        help="seen network architecture",
    )
    train_arg_parser.add_argument(
        "--num_class",
        type=int,
        default="7",
        help="seen class",
    )
    train_arg_parser.add_argument(
        "--train_epochs",
        type=int,
        default="50",
        help="seen epochs",
    )
    train_arg_parser.add_argument(
        "--algorithm",
        type=str,
        default="LEAware",
        help="ERM ADA LEAware",
    )
    train_arg_parser.add_argument(
        "--batch_size",
        type=int,
        default="32",
        help="seen batch_size",
    )

    args = train_arg_parser.parse_args()
    main(args)






