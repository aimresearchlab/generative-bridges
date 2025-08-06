# generative-bridges

## Some comments

This is a modified version of the original repo, much of the code and file structure has been changed to run on my local machine. I've also tried to directly clone OpenAI's diffusion model into this repo so all 3 models are nicely stored and accessed from one place. It's likely that the files on this branch will only work on my machine unless some changes are made to the scripts and file structures. 

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
Open a shell and kick of the model calibration (curently only supporting ddpm)
 ```
python ddpm.py --data_folder /workspace/aim-bridges-data/mnist/ --imgs_folder /workspace/aim-ddpm-img/ --models_folder /workspace/aim-ddpm-models/denoise/  --unet openAI --batch_size 128 --t_steps 1000
 ```
 model checkpoints are in the folder
 ```
 aim-bridges-models/denoise/checkpoints
 ```

## Sample a DDPM model from checkpoint
```
python sample_ddpm.py --model /workspace/aim-ddpm-models/denoise/checkpoints/batch_128_t_steps_1000_stoch_final_model.pth --imgs_folder /workspace/aim-ddpm-img/ --num_samples 10
```
Saved .npz file: ```/workspace/aim-ddpm-img/mnist_steps_1000_openAI_100.npz```
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
 python sample_bridges.py --model /workspace/aim-bridges-models/bridge/checkpoints/mnist_batch_128_T_0.1_stoch_openAI_50M_checkpoint_n_batch_15000.pth --imgs_folder /workspace/aim-bridges-img/ --T 0.1 --sigma 0.1 --num_samples 100 --n_steps 100
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
Use the appropriate model_samples_path, dataset and max_imageds (50k for mnist)
   ```
 python compute_metrics.py --model_samples_path /workspace/aim-bridges-img/samples.npz --train_dataset_path /workspace/aim-bridges-data/mnist --dataset mnist --max_images 10
   ```

  ## Train and sample Improved Diffusion

```bash
MODEL_FLAGS="--image_size 28 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

python scripts/image_train.py --data_dir datasets/mnist/train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
``` 

The above training script saves checkpoints to `.pt` files in the logging directory. These checkpoints will have names like `ema_0.9999_200000.pt` and `model200000.pt`. You will likely want to sample from the EMA models, since those produce much better samples.

Once you have a path to your model, you can generate a large batch of samples like so:

```
DIFFUSION_FLAGS="--diffusion_steps 2 --noise_schedule cosine"
python scripts/image_sample.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS --num_samples 100
```
this saves to tmp `saving to /tmp/openai-2025-01-18-22-54-48-956751/samples_50000x28x28x3.npz`
