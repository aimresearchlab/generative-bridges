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


 ## Train a DDPM model
Open a shell and kick of the model calibration
 ```
python ddpm.py --data_folder /workspace/aim-bridges-data/mnist/ --imgs_folder /workspace/aim-ddpm-img/ --models_folder /workspace/aim-ddpm-models/denoise/ --dataset mnist --unet openAI --batch_size 128 --t_steps 1000
 ```
 model checkpoints are in the folder
 ```
 aim-bridges-models/denoise/checkpoints
 ```

 ## Train a Diffusion Bridge model
Open a shell and start tensorboard to monitor loss ect
 ```
 tensorboard --logdir=aim-bridges-models/bridge/runs
 ```
Open another shell and kick of the model calibration
 ```
python simple_bridge.py --data_folder /workspace/aim-bridges-data/mnist/ --imgs_folder /workspace/aim-bridges-img/ --models_folder /workspace/aim-bridges-models/bridge/ --dataset mnist --unet openAI --batch_size 128 --T 0.1
 ```
 model checkpoints are in the folder
 ```
 aim-bridges-models/bridge/checkpoints
 ```
  ## Sample images from checkpoint
  ```
 python sample_bridges.py 
--model /workspace/aim-bridges-models/bridge/checkpoints/mnist_batch_128_T_0.1_stoch_openAI_50M_checkpoint_n_batch_15000.pth --imgs_folder /workspace/aim-bridges-img/ --T 0.1 --sigma 0.1 --num_samples 100
  ```
The scripts saves images as 
```
aim-bridges-img/sample_*.png
```
and creates at the end 
```
aim-bridges-img/sample.npz
```
that is used as input for the FID calc. Before switching off the machine remember to delete the images as it will slow down startup and tear-down time keeping too many files.
```
rm aim-bridges-img/sample_*.png
```
   ## Compute FID

   ```
 python compute_metrics.py --model_samples_path /workspace/aim-bridges-img/samples.npz --train_dataset_path /workspace/aim-bridges-data/mnist --dataset mnist --max_images 10
   ```

