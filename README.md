# generative-bridges

## Getting Started
Clone the repo with 

git clone --recurse-submodules https://github.com/<yourusername>/your-repo.git

If you already cloned the repo withouth the submodules, update as

git submodule update --init --recursive

## Repo Structure

 - thirdparty contains as sub-modules relevant external repos to use as references
 - sandbox contains folders for everyone in the team where to put half baked experiments
 - jupyter contains notebooks with simple examples or self-contained experiments
 - main contains the main code

 ## Train a model

 ```
 python simple_bridge.py --data_folder /workspace/aim-bridges-data/cifar10/ --imgs_folder /workspace/aim-bridges-img/ --models_folder /workspace/aim-bridges-models/bridge/ --dataset cifar --unet openAI
 ```
  ## Sample imagsges from checkpoint
  ```
  python sample_bridge.py --model /workspace/aim-bridges-models/model_path --imgs_folder /workspace/aim-bridges-img/ --T 0.1 --sigma 0.1 --num_samples 1000
  ```

