# TCLR: Temporal Contrastive Learning for Video Representation [CVIU March, 2022]
Official code repo for TCLR: Temporal Contrastive Learning for Video Representation, Computer Vision and Image Understanding Journal [Paper](https://doi.org/10.1016/j.cviu.2022.103406) and [Arxiv Version](https://arxiv.org/abs/2101.07974). In the current state, the repository exactly reproduces state-of-the-art results of our paper for UCF101 self-supervised pretraining for R3D-18 model: **69.9\%** linear evaluation, **82\%** on Full-Finetuning, **56.1\%** on NN Retrieval.

### Preparation: Environment and Dataset
```
# Clone the github to your path, expected space: 15G
git clone https://github.com/DAVEISHAN/TCLR.git && cd TCLR

# Create environment
conda env create -f tclr_env.yml

# UCF101 data preparation
mkdir data && cd data
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
unrar x UCF101.rar
rm -rf UCF101.rar
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate
unzip UCF101TrainTestSplits-RecognitionTask.zip
rm -rf UCF101TrainTestSplits-RecognitionTask.zip
```

### Self-supervised Pretraining

GPU Memory requirement: **48G**

```
  cd tclr_pretraining/
  ```
  
Activate the environment: `conda activate tclr_env` or `source activate tclr_env`

Run TCLR pretraining code using the following command:
```
  python train_gen_all_step.py --run_id="EXP_NAME"
  ```

Use "--restart" to continue the stopped training

The pretraining will save models at `tclr_pretraining/ss_saved_models` and tensorboard logs in `tclr_pretraining/logs`

### Linear Evaluation (Linear Classification)

Change directory to `cd linear_eval`

Run the linear evaluation code using the following command:

```
python train.py --saved_model="FULL/PATH/TO/SAVED/PRETRAINED/MODEL" --linear
  ```

The trained linear classifier will be saved at `linear_eval/saved_models` and tensorboard logs in `linear_eval/logs`

### Nearest Neighbour Retrieval
```
  cd nn_retreival
  python complete_retrieval.py --run_id="provide_exp_id_here" --saved_model="provide_complete_path_to_saved_ssl_pretrained_model"
  ```

### Pretrained weights
 
R3D-18 with UCF101 pretraining: [Google Drive](https://drive.google.com/file/d/1wGr6KsC4TcRweAGOFwmOYBPeZe0oGV7r/view?usp=sharing)<br/>R3D-18 with Kinetics400 pretraining: [Google Drive](https://drive.google.com/file/d/1m-u8N18dYFqP9B2JF3dEYOowKg3xDrds/view?usp=sharing)<br/>R2+1D-18 with Kinetics400 pretraining: [Google Drive](https://drive.google.com/file/d/1cuM4vFJA8wDDYmkQeAhwBUDQD0aDGmqD/view?usp=sharing)

Pl, note that all models are trained on BGR video input, for inference dataloading refer to `linear_eval/dl_linear`
 
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
