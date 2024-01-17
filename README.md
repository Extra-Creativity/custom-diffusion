# PKU Multimodal Learning Project

JunLin Hao, Jiaming Liang

## Environment Setup

See `README-original.md` (i.e. `README.md` of the original project). Its environment list is relatively complete and sound; we don't have additional comments because it's not hard to set up. The basic operations of tuning and generating an image can be checked there too.

##  Multi-concept Correction Code

Though the paper mentions the triple-concept or more as failure case, the code only provides dual-concept adjustment, which makes it inconvenient for us to do more experiments on that. It's a pity that the code is designed to be at most two concepts (like `reg_path2` etc.), so we change such part by giving a `train_multi.py`. It utilizes `OmegaConf` as the paper does, which needs data configuration `.yaml`. We give some examples at `configs/data/`, for example:

```yaml
data:
  target: train_multi.DataModuleFromConfig
  params:
    batch_size: 4
    num_workers: 4
    wrap: false
    train_infos:
      mouse: # This name can be random, we will omit it.
        target: src.finetune_data.MaskBase
        params:
          size: 512
          caption: "<new1> mouse" # param1
          datapath: "data/mouse"  # param2
          reg_caption: "real_reg/samples_mouse/caption.txt" # param3
          reg_datapath: "real_reg/samples_mouse/images.txt" # param4
      cat:
        target: src.finetune_data.MaskBase
        params:
          size: 512
          caption: "<new2> cat"
          datapath: "data/cat"
          reg_caption: "real_reg/samples_cat_new/caption.txt"
          reg_datapath: "real_reg/samples_cat_new/images.txt"
```

And run `train_multi.py` like:

```bash
python -u train_multi.py --base <original-config-path> <data-config-path> -t --gpus <Two GPUs ID> --resume-from-checkpoint-custom <sd-ckpt-weights> --modifier_token "<NewTokensConnectedBy+>" --name <name-you-like> --batch_size <integral times of concepts>
```

For example:

```bash
python -u train_multi.py --base configs/custom-diffusion/finetune_joint.yaml configs/data/mouse_cat.yaml -t --gpus 0,1 --resume-from-checkpoint-custom link-data/weights/v1-5-pruned.ckpt --modifier_token "<new1>+<new2>" --name "mouse_cat_joint" --batch_size 2
```

Note that`data` section in the original configuration will be omitted.

## Improvement code

Dense diffusion doesn't fit very well with custom diffusion, so we change some code there too; we refer the changed code as submodules.

Dense diffusion is based on `diffusers`, a convenient diffusion lib. It provides a specific modification to apply layout control on `Stable-Diffusion-v1.5`. To make it compatible with custom Diffusion, we change its attention forward method and it can work with CustomDiffusion Attention now.

To run Dense diffusion on a pretrained custom diffusion, we strongly recommand you to use diffusers as well. See more details about how to train a custom diffusion through diffusers in [REAME-original](./README-original.md), and don't forget to change your base model to `runwayml/stable-diffusion-v1-5`

Once you get the checkpoints in diffusers format (.bin), put them in `CustomDenseDiffusion/ckpt`, and then run `gradio_app.py`, just like the way to start a Dense Diffusion WebUI. Now, you can use your personal concept along with layout condition.