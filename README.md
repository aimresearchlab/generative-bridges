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
Open a shell and start tensorboard to monitor loss ect
 ```
 tensorboard --logdir=aim-bridges-models/bridge/runs
 ```
Open another shell and kick of the model calibration
 ```
python simple_bridge.py --data_folder /workspace/aim-bridges-data/mnist/ --imgs_folder /workspace/aim-bridges-img/ --models_folder /workspace/aim-bridges-models/bridge/ --dataset mnist --unet openAI --batch_size 128 --T 0.1
 ```
 model checkpoints are in the folder
  ## Sample imagsges from checkpoint
  ```
  python sample_bridge.py --model /workspace/aim-bridges-models/model_path --imgs_folder /workspace/aim-bridges-img/ --T 0.1 --sigma 0.1 --num_samples 1000
  ```

