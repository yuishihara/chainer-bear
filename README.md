# chainer-bear

Reproduction codes of Bootstrapping Error Accumulation Reduction (BEAR) with chainer

## Prerequisites

install [chainer](https://github.com/chainer/chainer) and [tensorboardX](https://github.com/lanpa/tensorboardX) in prior of using the code

## How to train

### with gpu

$ python3 main.py --env="Ant-v2" --datafile=\<file to buffer path\> --gpu=\<gpu number\>

### without gpu

$ python3 main.py --env="Ant-v2" --datafile=\<file to buffer path\>

## Results

I tested only with Ant-v2 data and found that laplacian kernel is highly stable compared to gaussian kernel. </br>
However, both kernel succeeded learning similar policy that scores like the behavior policy used for gathering the training data. 

|eval result|lagrange multiplier|mmd loss|vae loss|
|:---:|:---:|:---:|:---:|
| ![eval-result](./trained_results/optimal_data/laplacian/eval_result.svg?s=200)| ![lagrange](./trained_results/optimal_data/laplacian/lagrange_multiplier.svg?s=200)| ![mmd](./trained_results/optimal_data/laplacian/mmd_loss.svg?s=500) | ![vae](./trained_results/optimal_data/laplacian/vae_loss.svg?s=200)|