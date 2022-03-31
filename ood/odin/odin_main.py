from . import cal as c

def main(args, ood_dataset, ckpt_path):
    odin = args.ood.odin
    odin.data = args.data
    odin.model_type = args.model_type
    print(f"id dataset {args.data}, ood dataset {ood_dataset}")
    return c.test(odin, ckpt_path, ood_dataset)