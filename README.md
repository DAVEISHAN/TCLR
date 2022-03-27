# TCLR: Temporal Contrastive Learning for Video Representation [CVIU March, 2022]
Official code repo for TCLR: Temporal Contrastive Learning for Video Representation, Computer Vision and Image Understanding Journal [Paper](https://doi.org/10.1016/j.cviu.2022.103406) and [Arxiv Version](https://arxiv.org/abs/2101.07974)

### Preparation: Environment and Dataset

First make enviroment from tclr_env.yml using:

  ```
  conda env create -f tclr_env.yml
  ```

[Download UCF101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar) and [Splits files](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip) in train - testing set (Use UCF101 split-1)

```
  TODO: Put unzip command here
  ```

Generate a file of full paths of training videos separarted by newlines, name it `train_vids.txt` and `testing_vids.txt`, example:

l1: FULL/PATH/ApplyMakeup/abc.avi
l2: FULL/PATH/ApplyMakeup/xyz.avi

```
  TODO: Put ls commands here
  ```
  
### Self-supervised Pretraining

GPU Memory requirement: **48G**

```
  cd tclr_pretraining/
  ```
  
In config.py file give location to "path_folder" where the above two full `train_vids.txt` and `testing_vids.txt` files are located 

Activate the environment: `conda activate tclr_env` or `source activate tclr_env`


Run TCLR pretraining code using the following command:
```
  python train_gen_all_step.py --run_id="EXP_NAME"
  ```

Use "--restart" to continue the stopped training

The pretraining will save models at `tclr_pretraining/ss_saved_models` and tensorboard logs in `tclr_pretraining/logs`

### Linear Evaluation (Linear Classification)

Change directory to `cd linear_eval`

In `config.py` file give location to "path_folder" where the above two full `train_vids.txt` and `testing_vids.txt` files are located 

Run the linear evaluation code using the following command:

```
python train.py --saved_model="FULL/PATH/TO/SAVED/PRETRAINED/MODEL" --linear
  ```

The trained linear classifier will be saved at `linear_eval/saved_models` and tensorboard logs in `linear_eval/logs`

### TODO: Nearest Neighbour Retrieval

### Pretrained weights
 
R3D-18 with UCF101 pretraining: [Google Drive](https://drive.google.com/file/d/1Y-YmohPPeZKmd8MO_KVYKDNoIbzpjQWV/view?usp=sharing)<br/>R3D-18 with Kinetics400 pretraining: [Google Drive](https://drive.google.com/file/d/1m-u8N18dYFqP9B2JF3dEYOowKg3xDrds/view?usp=sharing)<br/>R2+1D-18 with Kinetics400 pretraining: [Google Drive](https://drive.google.com/file/d/1cuM4vFJA8wDDYmkQeAhwBUDQD0aDGmqD/view?usp=sharing)

 
### Citation
If you find the repo useful for your research, please consider citing our paper: 
```
@article{dave2022tclr,
  title={Tclr: Temporal contrastive learning for video representation},
  author={Dave, Ishan and Gupta, Rohit and Rizve, Mamshad Nayeem and Shah, Mubarak},
  journal={Computer Vision and Image Understanding},
  pages={103406},
  year={2022},
  publisher={Elsevier}
}
```
For any questions, welcome to create an issue or contact Ishan Dave ([ishandave@knights.ucf.edu](mailto:ishandave@knights.ucf.edu)).
