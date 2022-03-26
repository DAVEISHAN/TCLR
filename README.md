# TCLR
Official code repo for TCLR: Temporal Contrastive Learning for Video Representation

##Preparation: Environment and Dataset

First make enviroment from tclr_env.yml using:

  ```
  conda env create -f environment.yml
  ```

Download UCF101   and split in train - testing set (Use UCF101 split-1)
[Download UCF101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)

```
  TODO: Put unzip command here
  ```

Generate a file of full paths of training videos separarted by newlines, name it `train_vids.txt` and `testing_vids.txt`, example:

l1: FULL/PATH/ApplyMakeup/abc.avi
l2: FULL/PATH/ApplyMakeup/xyz.avi

```
  TODO: Put ls commands here
  ```
  
##Self-supervised Pretraining

GPU Memory requirement: **48G**

```
  cd tclr_pretraining/
  ```
  
In config.py file give location to "path_folder" where the above two full `train_vids.txt` and `testing_vids.txt` files are located 

Activate the environment
```
  conda activate tclr_env
  ```
or 
```
  source activate tclr_env
  ```



Run TCLR pretraining code using the following command:
```
  python train_gen_all_step.py --run_id="EXP_NAME"
  ```

Use "--restart" to continue the stopped training

The pretraining will save models at `tclr_pretraining/ss_saved_models` and tensorboard logs in `tclr_pretraining/logs`

##Linear Evaluation (Linear Classification)
```
cd linear_eval
  ```


In config.py file give location to "path_folder" where the above two full `train_vids.txt` and `testing_vids.txt` files are located 

Run the linear evaluation code using the following command:

```
python train.py --saved_model="FULL/PATH/TO/SAVED/PRETRAINED/MODEL" --linear
  ```

The trained linear classifier will be saved at `linear_eval/saved_models` and tensorboard logs in `linear_eval/logs`

