import argparse
from model import ModelBaseline as ERM, ModelADA as ADA, ModelLEAware as LEAware
import os
import torch
import numpy as np

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def main(args):

    def load_model(path2model):
        model_ckpt = torch.load(path2model)
        args = model_ckpt["args"]
        if args.algorithm == "ERM":
            model_obj = ERM(flags=args)
        elif args.algorithm == "ADA":
            model_obj = ADA(flags=args)
        elif args.algorithm == "LEAware":
            model_obj = LEAware(flags=args)
        else:
            raise RuntimeError
        model_obj.network.load_state_dict(model_ckpt["state"])
        return model_obj

    model_list = []
    best_test_model_path = os.path.join(args.path, "models", "best_test_model.tar")
    best_model_path = os.path.join(args.path, "models", "best_model.tar")
    model_list.append(best_test_model_path)
    model_list.append(best_model_path)

    with open(os.path.join(args.path, "results.txt"), "a") as fout:
        fout.write(args.info + "\n")
        for idx, mpath in enumerate(model_list):

            model_obj = load_model(mpath)
            auc_arr, acc_arr, f1_arr = model_obj.batch_test_workflow()
            mean_auc = np.array(auc_arr).mean()
            mean_acc = np.array(acc_arr).mean()
            mean_f1 = np.array(f1_arr).mean()

            if idx == len(model_list) - 2:
                tag = "best_test"
            elif idx == len(model_list) - 1:
                tag = "best"
            else:
                tag = idx

            names = [n for n, _ in model_obj.test_loaders]

            res = (
                    "{}:{} ".format(tag, model_obj.train_name)
                    + " ".join(
                ["{}:{:.2f}".format(n, a* 100) for n, a in zip(names, auc_arr)]
            )
                    + "   | ACC:{:.2f} AUC:{:.2f} F1:{:.2f}".format(mean_acc * 100, mean_auc * 100, mean_f1 * 100)
            )
            print(res)
            fout.write(res + "\n")
        fout.write(args.time + "\n")

if __name__ == "__main__":
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--gpu", type=str, default="0", help="")
    train_arg_parser.add_argument(
        "--path", type=str, default="/path/to/save/folder", help=""
    )
    train_arg_parser.add_argument("--info", type=str, default="Evaluation", help="")
    train_arg_parser.add_argument("--dataset", type=str, default="mnist", help="")
    args = train_arg_parser.parse_args()
    main(args)
