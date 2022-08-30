# Adversarial patch attack and attack detection and recovery training 

## Objective

> 1. Train an adversarial patch attack method against the EfficientDet object detector.
> 2. Train an attention U-net model to detect patches and recover background info.

## Approach

[Please refer to the wiki](https://ssrc.atlassian.net/wiki/spaces/ML/pages/556859580/Physical+Adversarial+Attack+Detection+on+Object+Detection+Models)
for the theory behind this implementation.

## Key results

1. Adversrial patch reduces object class score by significant values.
2. Attention U-net can neutralise patches and recover some background information.
3. In general, our attack approach is independent of the object detection model used as long as we can define an
adversarial objective function to train on (for example by minimizing the class score).
4. Our defensive approach is also independent of the model being protected, since it uses a self-
supervised learning approach.


---

## Setup

### Required steps to setup

```bash
# Clone the repository
git clone https://github.com/tiiuae/MLAdversarialObjectDetection
# chdir into MLAdversarialObjectDetection
cd MLAdversarialObjectDetection
# Build the dev docker image
docker build -t phyadv_dev .
# Run the build docker image as a detached container
docker run -itdv $PWD:/phyadv --name phyadv_dev --gpus all --network host phyadv_dev
# This also mounts the host repo files onto the docker container under /phyadv folder on the container.
```

The repo already contains the EffficientDet code from the automl repository of google brain project.

All following documents assumes we are working inside this docker environment setup as above unless otherwise indicated.

### Downloading required data

We need to download the MSCOCO dataset which is required for the training. COCO however, is a very large dataset and since 
we are interested in only a person class, it can be done using `python coco_dl_by_category.py`. It only downloads those images which
have the person objects in them, saving download time and space.

```bash
# enter docker environment
docker exec --workdir /phyadv -it phyadv_dev /bin/bash
# start training
python coco_dl_by_category.py
# to leave it running in background we need to escape the terminal environment without exiting the terminal
# to do so press CTRL+p followed by CTRL+q (this the standard docker escape sequence which detaches terminals without
# closing them, which is what we require here)
```

This will download the images in a `downloaded_images` folder and `labels` folder will have corresponding ground truth
bounding box information.

> **NOTE**: Though we have downoaded labels, we will not be using them. They are not necessary for our use case.

## Training

Before starting the training, you need to start a tensorboard instance for monitoring the progress of your training.
To do so, use the following steps in your terminal:

```bash
# enter docker environment
docker exec --workdir /phyadv -it phyadv_dev /bin/bash
# create a directory named log_dir
mkdir log_dir
# start tensorboard to listen on host ips
tensorboard --logdir log_dir --host 0.0.0.0
# to leave it running in background we need to escape the terminal environment without exiting the training
# to do so press CTRL+p followed by CTRL+q
```

### Training the adversarial attack on victim model

Before starting the training open `attacker_train.py` in the editor and change the `output_dirs` variable to a new name.
This will avoid overwriting your old data from earlier runs and create a new directory for the current run.

If training for the first time, we also need to download the weights for the EfficientDet model and this can be done
by setting the `download_model=True`.
```python
# in attacker_train.py, set download_model to True
victim_model = util.get_victim_model(MODEL, download_model=True)
```

We can start the training of the attacker inside the container with the following steps:
```bash
# enter docker environment
docker exec --workdir /phyadv -it phyadv_dev /bin/bash
# start training
python attacker_train.py
# to leave it running in background we need to escape the terminal environment without exiting the training
# to do so press CTRL+p followed by CTRL+q
```

You can then monitor the progress of the training by opening tensorboard in a browser.


### Training the adversarial attack detection and recovery model

The process to train the defender attention U-net model is exactly the same as for the attack step.
Except the file to be used here is `defender_train.py`.
