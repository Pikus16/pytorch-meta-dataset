from .standard import __dict__ as standard_dict
from .meta import __dict__ as meta_dict
import argparse
import timm
from .mae import models_vit_mae

def get_model(args: argparse.Namespace,
              num_classes: int):
    if args.timm_name != 'NONE':
        print('Getting TIMM model')
        return timm.create_model(args.timm_name, pretrained=True, num_classes = 0)#num_classes)
    if 'mae_' in args.arch:
        print('Getting MAE model')
        return models_vit_mae.__dict__[args.arch.replace('mae_','')](
            num_classes=num_classes,
            global_pool=False#args.global_pool,
        )
    if 'MAML' in args.method:
        print(f"Meta {args.arch} loaded")
        return meta_dict[args.arch](num_classes=num_classes, use_fc=args.use_fc)
    else:
        print(f"Standard {args.arch} loaded")
        return standard_dict[args.arch](num_classes=num_classes, use_fc=args.use_fc)
