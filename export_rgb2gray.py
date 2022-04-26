# export_rgb2gray.py
#   export rgb model weight (3ch conv0 layer) to gray (single ch in conv0) model
#
# How to use:
#   export_rgb2gray.py [rgb pt file] [output gray pt filename]
#
# date :
#   2022.04.06

import sys,os
import copy
from pathlib import Path
from datetime import datetime

import torch
from models.yolo import Model

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(" * How to use:")
        print("     export.py [rgb pt file] [output gray pt filename] -v")
        print("     -v : verbose, show model and all weights")
        print("          without -v, show only conv0 weight")
        exit(1)
    weights = sys.argv[1]
    out_weights = sys.argv[2]
    verbose = True if len(sys.argv) == 4 and sys.argv[3] == "-v" else False

    assert weights.endswith('.pt')
    assert Path(weights).exists()

    ckpt = torch.load(weights, map_location='cpu')
    csd = ckpt['model'].float().state_dict()
    if verbose:
        print(" ** model **")
        print(ckpt)
        csd_keys = list(csd.keys())
        print(" ** weight name & shape **")
        for ck in csd_keys:
            print("{:40s}{:s}".format(ck, str(list(csd[ck].shape))))
        print(" ** weight values **")
        print(csd)
    else:
        print("csd['model.0.conv.weight'].shape = ", csd['model.0.conv.weight'].shape)
        print("csd['model.0.conv.weight'][0]")
        print(csd['model.0.conv.weight'][0])

    # make new model with gray ch
    new_yaml = copy.deepcopy(ckpt['model'].yaml)
    new_yaml['ch'] = 1
    model = Model(new_yaml)
    md = model.state_dict()

    # load other layers weight only without conv0
    csd_in = {k: v for k, v in csd.items() if k in md and v.shape == md[k].shape} # intersect
    exclude = [x for x in csd.keys() if x not in csd_in.keys()]
    print(" (%d/%d) migrated from ckpt [exclude]=%s" % (len(csd_in.keys()), len(csd.keys()), str(exclude)))
    model.load_state_dict(csd_in, strict=False)

    # load conv0
    bs, ch, k1, k2 = csd['model.0.conv.weight'].shape
    conv0_w = torch.reshape(torch.sum(csd['model.0.conv.weight'],1), (bs, 1, k1, k2))
    print("conv0_w.shape = ", conv0_w.shape)
    print("conv0_w[0]")
    print(conv0_w[0])
    conv0_dict = {'model.0.conv.weight': conv0_w}
    model.load_state_dict(conv0_dict, strict=False)

    # save new ckpt
    new_ckpt = {
        'epoch': -1,
        'best_fitness': ckpt['best_fitness'],
        'model': model.half(),
        'ema': ckpt['ema'],
        'update': ckpt['updates'],
        'optimizer': ckpt['optimizer'],
        'wandb_id': ckpt['wandb_id'],
        'date': datetime.now().isoformat()
    }
    torch.save(new_ckpt, out_weights)
    print("%s save done." % out_weights)
