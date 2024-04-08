# Fine-tune ViT with Colossal-AI

This example shows how to fine-tune ViT using [Colossal-AI](https://github.com/hpcaitech/ColossalAI).


## Model

Vision Transformer (ViT) is a class of Transformer model tailored for computer vision tasks. It was first proposed in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) and achieved SOTA results on various tasks at that time.

In this example, we will be using pretrained weights for the model loaded from HuggingFace.

We adapt the ViT training code to ColossalAI by leveraging [Boosting API](https://colossalai.org/docs/basics/booster_api) loaded with a chosen plugin, where each plugin corresponds to a specific kind of training strategy. Plugins supported in this example include TorchDDPPlugin (DDP), LowLevelZeroPlugin (Zero1/Zero2), GeminiPlugin (Gemini) and HybridParallelPlugin (any combination of tensor/pipeline/data parallel).


## Dataset

In this example, we will be finetuning a [ViT-base](https://huggingface.co/google/vit-base-patch16-224) model on this [dataset](https://huggingface.co/datasets/beans), with more than 8000 images of bean leaves. This dataset is for image classification task and there are 3 labels: ['angular_leaf_spot', 'bean_rust', 'healthy'].


## Run fine-tuning

To start fine-tuning, run the following script:
```bash
bash run_finetuning.sh
```
The script can be modified to configure different values of hyperparameters or change to another ViT model with different size.

The demo code refers to this [blog](https://huggingface.co/blog/fine-tune-vit).


## Results

Following are the results of the fine-tuning on different combinations of values of a subset of hyperparameters: ```batch_size in [8, 16, 32]``` and ```learning_rate in [0.0001, 0.0002, 0.0005]```.

Fine-tuning was done for each combination of the hyperparameters, with the average loss and accuracy of the model recorded below:
| Batch Size | Learning Rate | Average Loss | Accuracy   |
| ---------- | ------------- | ------------ | ---------- |
| 8          | 0.0001        | 0.0426       | 0.9844     |
| 16         | 0.0001        | 0.0493       | 0.9844     |
| 32         | 0.0001        | **0.0172**   | **0.9922** |
| 8          | 0.0002        | 0.0864       | 0.9688     |
| 16         | 0.0002        | 0.0369       | 0.9844     |
| 32         | 0.0002        | 0.0263       | 0.9844     |
| 8          | 0.0005        | 0.0850       | 0.9688     |
| 16         | 0.0005        | 0.0389       | 0.9844     |
| 32         | 0.0005        | 0.0219       | **0.9922** |

Other hyperparameters were kept constant as follows:
| PLUGIN | GPUNUM | EPOCH | WEIGHT DECAY | WARMUP RATIO |
| ------ | ------ | ----- | ------------ | ------------ |
| Gemini | 1      | 5     | 0.5          | 0.3          |
