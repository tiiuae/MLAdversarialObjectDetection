# Adversarial patch attack and attack detection and recovery training 

## Objective

> 1. Train an adversarial patch attack method against the efficientdet object detector.
> 2. Train an attention U-net model to detect patches and recover background info.

## Key results

1. Adversrial patch reduces object class score by significant values
2. Attention U-net can neutralise patches and recover some background information


---

## Universal ML inference pipeline

> **Goal**: Allow custom ML inference pipelines to be run within the ros node.

Overview of the pipeline in relation with the ros node:

![](img/../references/img/inference-pipeline-detail.png)

### Required steps to run custom pipeline

1. Implement a class with the method `run(self, input_array: np.array) -> str`. The `run` method takes a numpy image array, preprocesses according to the model, predicts with the model and postprocesses predictions to return a `string` (this string will be published from the ros node).
2. Implement a function like `init_inference_pipeline(type: str)` that returns the initialized class from above. It serves as a factory method and comes in handy when you have different implementations, such as `ssd_mobilenet_v1`, `yolo_v5` etc.
3. Import `init_inference_pipeline` in the main exetuable `main.py`and call `inference_pipeline = init_inference_pipeline()`  in  `main()`. Plug `inference_pipeline` into the `RosNode` object. The decorator `@ros_node_handler` then handles everything ros-related.

Below is an example:

```python
# main.py

import argparse

from src.ros_node import RosNode, ros_node_handler


@ros_node_handler  # type:ignore
def main(drone_id: str) -> RosNode:
    "Returns ros node. Init inference pipeline and plug into ros node."

    from src.example_inference_pipeline import init_inference_pipeline

    inference_pipeline = init_inference_pipeline(type="ssd_mobilenet_v1")

    node = RosNode(inference_pipeline=inference_pipeline, drone_id=drone_id)

    return node


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection node for ROS2")
    # add drone_id
    parser.add_argument(
        "-d",
        "--drone_id",
        type=str,
        default="sad04",
        help="Drone ID to be used to subscribe to video topic: /<drone_id>/camera/color/video",
    )

    main(**vars(parser.parse_args()))

```

### How to implement custom ML pipeline

To fit your own inference pipeline into the architecture, you need a `run()` function, such as `src/example_inference_pipeline.py:InferenceSsdMobilenetV1Tflite:run`, that takes an image array as input and returns a string with predictions.

Nice to have is a user interface to initialize the pipeline from within the ros node, i.e. a factory method, such as `src/example_inference_pipeline.py:init_inference_pipeline`.

Change the import in `src/main.py` such as  `from example_inference_pipeline import init_inference_pipeline` according to your files.

Lastly, add your requirements to the `Pipfile` under `[dev]` section. This could be done as follows using `pipenv install --dev <package-name>` from within a container that mounts to the project root directory to update the `Pipfile`

```bash
# image for local development and testing
docker build --target dev -t rosdev .
# run container, remove on exit, mount PWD, override entrypoint
docker run --entrypoint "/bin/bash" -it --rm -v ${PWD}:/app --name rosdev rosdev
# then run
pipenv install --dev tensorflow pandas  # with -v ${PWD}:/app pipenv will update the Pipfile
```

#### Summary

1. Implement the prediction pipeline called by `run(self, img_array: np.array) -> np.array` that takes an image array and outputs the predictions as string
2. Implement factory method like `init_inference_pipeline() -> InferencePipeline` that initializes the prediction pipeline, such as loading model or reading labels from `.txt` file
3. Import and call the factory method from (2) in `./main.py`. Pass the pipeline instance to `RosNode(inference_pipeline=inference_pipeline)`.
4. Ensure that `main.py:main` returns an instance of `RosNode` and add the decorator `@ros_node_handler`. If necessary add an inline comment for `mypy` to ignore types: `# type:ignore`
5. Add requirements to `Pipfile` by running `pipenv install --dev <package-name>` from within the container

### Example prediction pipeline

The inference pipeline is defined in `example_inference_pipeline.py`.

It takes an image array and outputs a string with detections. The ROS node defined in `src/ros_node.py` constructs the inference pipeline with a factory method `src/example_inference_pipeline.py:init_inference_pipeline`.

#### Pipeline input

RGB image as an array of any size, like (480, 640, 3) [height, width, channels]

#### Pipeline output

Json string that contains predictions, if any, in the format propoces by [`GCP cloud vision API`](https://cloud.google.com/vision/docs/reference/rest/v1p3beta1/AnnotateImageResponse#localizedobjectannotation).

```json
{
  "localizedObjectAnnotations": [
    {
      "name": "Bicycle wheel",
      "score": 0.94234306,
      "boundingPoly": {
        "normalizedVertices": [
          {
            "x": 0.31524897,
            "y": 0.78658724
          },
          {
            "x": 0.44186485,
            "y": 0.78658724
          },
          {
            "x": 0.44186485,
            "y": 0.9692919
          },
          {
            "x": 0.31524897,
            "y": 0.9692919
          }
        ]
      }
    },
    // ...
  ]
}
```

The format has to be imposed onto the model predictions. Here is an example, how `predictions_formatter:apply_gcp_format()` changes the output format:

Raw output from model prediction:
```python
{
  'boxes': array([], shape=(0, 4), dtype=float32),
  'labels': array([], dtype='<U14'),
  'scores': array([], dtype=float32)
}
```

Formatted output following GCP cloud vision api:

```python
{
  "localizedObjectAnnotations": [
    {
      "name": string,
      "score": float,
      "boundingPoly": {
        object (BoundingPoly)
      }
    },
    # ...
  ]
}
```

 #### How to run

```bash
# build image for local development and testing
docker build --target dev -t rosdev .
# run container, remove on exit, mount PWD, override entrypoint
docker run --entrypoint "/bin/bash" -it --rm -v ${PWD}:/app --name rosdev rosdev
# run main pipeline
pipenv run python main.py --drone-id sad04

# open another shell and shell into same container to play a rosbag
docker exec -it rosdev /bin/bash
# example: play rosbag in infinite loop
ros2 bag play -l data/video/PMK_sad04_2022-03-24T080058.123352166Z_flight.db3

# open a shell in same container to echo publishing to topic
docker exec -it rosdev /bin/bash
ros2 topic echo /sad04/object_detection/video
```

![](reference/../references/img/computer-vision-pipeline-demo.gif)

---

## Development

Ensure to have `pipenv`, `python 3.8` and `docker` installed. If on windows, also ensure to have `wsl`.

```bash
# go to your project root dir
git clone https://github.com/tiiuae/platform-ml-model-deployment.git

# install packages for local dev
pipenv install --dev

# init pre-commit hooks
pre-commit install
pre-commit autoupdate
```
Build `rosdev` or `rosprod` docker image and run a container `dev` or `prod`.

> Note: There is `dev` and `prod` as docker images which come from a multi-stage build. They differ in available python packages installed through `pipenv` with (`dev`) or without (`prod`) `--dev` flag. In `dev` all additional packages list under `[dev]` in the `Pipfile`.

```bash
# image for local development and testing
docker build --target dev -t rosdev .
# image for production
docker build --target prod -t rosprod .

# run container, remove on exit, mount PWD, override entrypoint
docker run --entrypoint "/bin/bash" -it --rm -v ${PWD}:/app --name rosdev rosdev

# open another shell in same container
docker exec -it rosdev /bin/bash

# windows: if xserver display needed, set DISPLAY env variable
$DISPLAY = $((netsh interface ip show address "vEthernet (WSL)" | where { $_ -match "IP Address"} | %{ $_ -replace "^.*IP Address:\W*", ""}).trim())
# add env to docker run: -e DISPLAY=${DISPLAY}:0.0
```

### Interface for machine learning pipeline

The interface should be simple, maintainable & reliable.

Requirements:

* one function like `init_inference_pipeline()` to initialize all static components of the pipeline, such as loaded model artifact and labels list
* one class method like `run()` that preprocesses, predicts and postprocesses a numpy array
* validate functionality quickly

The researchers should be able to easily plug in their pipeline without having to learn too much module-specific lingo. This could be done within a `main.py` script that imports both the ros node from `ros_node.py` and the inference pipeline from `example_inference_pipeline.py`



### VScode devcontainer

If you use `vscode devcontainer`, you can build the `DEV` image and forward the display with the following settings `.devcontainer/devcontainer.json`:

```json
// ...
// build image with build-arg
"initializeCommand": [
  "docker",
  "build",
  "--target",
  "dev",
  "."
],
// win: get ip of your machine with `ipconfig` and set $DISPLAY=<ip>
"runArgs": [
      "-e", "DISPLAY=$DISPLAY:0",
],
// ...
```

### Add remote tag to docker image

Go to [`packages`](https://github.com/tiiuae/platform-ml-model-deployment/pkgs/container/tii-object-detection) to copy the image tag you want to re-tag.

The add version tags to docker image on `ghcr.io` to release:

```bash
docker pull ghcr.io/tiiuae/tii-object-detection:<tag>

# tag with v1.0.0
docker tag ghcr.io/tiiuae/tii-object-detection:<tag> ghcr.io/tiiuae/tii-object-detection:v1.0.0

docker push ghcr.io/tiiuae/tii-object-detection:v1.0.0
```



## Run simulation on cloud

Prerequisites for running a simulation:

* dronsole ([see installation below](#install-dronsole-win-10))
* Access to https://dev.sacplatform.com/
* Access to GCP `TII SAC Platform Development`
* OTA profile with `ghcr.io/tiiuae/tii-object-detection:<container id>` added to `apps` section:

Go to `https://github.com/tiiuae/platform-ml-model-deployment/pkgs/container/tii-object-detection` to get the versions you want to test in the simulation.

Create an OTA profile and add object detection docker image, `ghcr.io/tiiuae/tii-object-detection` with the version you want to test such as `sha-84a725a` to the `apps` section.

```yaml
# ota_ros_node_v11_encoder_settings.yaml
kind: ota-profile
name: ota
spec:
  core:
    - ...
  apps:
    - ...
    - ghcr.io/tiiuae/tii-object-detection:sha-84a725a  # ADD HERE
```

Useful dronsole commands:

```bash
# create simulation with small_forest world
dronsole sim create <sim-name> --standalone=false --world small_forest.world

# go to https://dev.sacplatform.com/fleet-registry~s~sim-ps-1

# add 1 drone with OTA profile (navigate to dir)
dronsole sim drones add <sim-name> <drone-name> -x 1 --ota-profile ota_ros_node_v11_encoder_settings.yaml
# dronsole sim drones add sim-ps-1 ps1 -x 1 --ota-profile ota_ros_node_v11_encoder_settings.yaml

# takeoff
dronsole sim drones command takeoff <sim-name> <drone-name>

# shelling to drones
dronsole sim drones sh <sim-name> <drone-name> tii-px4-sitl
dronsole sim drones sh <sim-name> <drone-name> tii-mission-data-recorder
dronsole sim drones sh <sim-name> <drone-name> tii-object-detection
dronsole sim drones sh <sim-name> <drone-name> tii-cloud-link
ros2 topic list -t
ros2 topic echo <topic>
# dronsole sim drones sh ps-sim-1 ps1 tii-object-detection

# logging
dronsole sim drones log sim-matti-4 <drone-name> tii-object-detection

# drone configuration for showing object detection
# go to GCP > iot core > fleet-registry~s~<sim-name> > devices > <drone-name> > update config
# replace
  - name: '{drone}/camera/color/video'
    rate_limit: 1 Hz
# with
  - name: '{drone}/object_detection/video'
    map_to: '{drone}/camera/color/video'
    rate_limit: 25 Hz

# remove sim
dronsole sim remove <sim-name>
```

### Install dronsole (win 10)

Make sure to have the following installed and available:

* wsl2
* git
* golang
* [tdm-gcc compiler](https://jmeubank.github.io/tdm-gcc/download/)
* github > settings > ssh and gpg keys > new SSH key


```bash
# install dronsole from source
git clone git@github.com:tiiuae/dronsole.git \
  && cd dronsole \
  && go install

# update dronsole to recent version in ./dronsole
git pull && go install
```

Finally, add dronsole as a system environment variable: Edit the system environment variables > Environment Variables > System variables > Path > New

Added for example: `C:\Users\Philipp Schmalen\go\bin`

---

## Demo May 2022

Backups to display object detection instead of UI require a rosbag in `data/video`. Download the rosbag and store in `data/video`.


1. UDP stream from docker container `port/udp` to VLC media player on localhost.

```python
# ros_node.py
ros_node_node = RosNode(
    video_output_type="udp",  # CHANGE HERE
    publish_msg_type="CompressedImage",
    host="172.20.80.1"  # ADD HOST IP
)
```

Create the following file, replace `c=IN IP4 x.x.x.x` with localhost ipv4. Adjust the port `5004` to what you specified in `udpsink` in gstreamer

```bash
# udpvlcconfig.sdp
v=0
m=video 5004 RTP/AVP 96
c=IN IP4 <insert host ip here>
a=rtpmap:96 MP4V-ES/90000
```

Start 2 containers as in [Get started (devs)](#get-started-devs). Then open VLC media player by executing the `.sdp` file.

2. Write to `avi` file. Playback with vlc media player on localhost to see video stream.


```python
# ros_node.py
ros_node_node = RosNode(
    video_output_type="file",  # CHANGE HERE
    publish_msg_type="CompressedImage",
)
```

3. Show video on local machine with xserver display.


```python
# ros_node.py
ros_node_node = RosNode(
    video_output_type="xserver",  # CHANGE HERE
    publish_msg_type="CompressedImage",
)
```


### Download rosbags with `dronsole`

You want to download rosbags of a flight and combine the uploaded batches to one rosbag, also described [here](https://ssrc.atlassian.net/wiki/spaces/DRON/pages/526352827/Record+flight+data+to+cloud+and+download+rosbag+with+dronsole).

```bash
# ensure suitable dronsole config
dronsole config api demo.sacplatform.com

# command
dronsole rosbag download <out-file.bag> <drone-id> <start> <end> <topic> [topic] .. [flags]

# example
dronsole rosbag download demobag-220518.db3 sad06 2022-05-18T11:26:13.00Z 2022-05-18T11:55:41.00Z /sad06/object_detection/video /sad06/camera/video
```

---


## Inference pipeline for object detection

>Goal: Deploy computer vision model w/pre-predict-post processing elements
>	1. on the drone, real-time
>	2. cloud, event-driven

We distinguish between two separate pipelines:

1. Real-time inference on the drone
2. Event-driven inference on the cloud

Each prediction pipeline consists of three elements: preprocessing, prediction and postprocessing. Since each model has their own specifications how preprocessing and postprocessing needs to be done, we do not validate the inference pipeline on a model level. Instead we validate the input and output to and from the pipeline. Anything unexpected that appears before or after the pipeline will be tested for, but not the internals of the inference pipeline. Here is a high-level illustration:

![](references/img/inference-pipeline-tflite.png)

Here is a more detailed view with ROS and video parser:

![](references/img/inference-pipeline-detail.png)


### Event-driven inference

One example: Rosbags arrive on cloud storage as batches. Whenever a new rosbag appears, we could run a model prediction pipeline on the AI platform or through cloud functions that outputs to big table or cloud storage.

![](references/img/on-cloud-inference-pipeline.png)


---


## Miscellaneous

### Testing

We use `pytest` to test the node. By convention, you can find the tests in `./tests`. A `conftest.py` configures the tests, such as initializing [fixtures](https://docs.pytest.org/en/7.1.x/how-to/fixtures.html).

* Node code unit testing ideas:
  * Test video decoding output, data fed to the model (requires identifying keyframes)
  * Test the inference (a couple of test cases with fixed model), bounding boxes and classes
* ROS node testing ideas:
  * Get video topic published and observe does the node publish intended messages from object detection

### Refactoring

We follow the factory design pattern to create the three main elements of the data pipeline: Ingesting data with a ros subscriber, processing & predicting on data with gstreamer and a model and

1. `gstreamer pipeline` with specified type, such as
   1. xserver: open display via xserver
   2. ui: publish compressedimage messages to topic
   3. udp: publish to udp port onto specified host address
   4. file: Store output in output.avi for replay
2. `ros publisher` with message type, such as:
   1. String
   2. CompressedImage
   3. Float32MultiArray
3. `ros subscriber`


### Performance ideas

* Packaging: ROS package vs. docker
* Prediction pipeline: real-time to batch for inference
* Docker image: Ubuntu ~1GB, but can use ros2 base image
* Model: TFLite micro vs. TFLite
* resource monitoring




### Architecture

![](references/img/architecture%20Fleet%20Management%20Stream%20Analytics%20and%20ML.png)


---

## Free notes
### Tflite

Get model input-output details with:

```python
"""
Source: TensorFlow Lite inference
https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
"""
import numpy as np
import tensorflow as tf
from pprint import pprint

model_path = 'model/ssd_mobilenet_v2_coco.tflite'

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
pprint(input_details)
pprint(output_details)
```

#### Security

Guide on **Using TensorFlow Securely**: https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md

### Gstreamer

Useful commands

```bash
# List element properties
gst-inspect-1.0 <element_name>

# run test pipeline with fps display
gst-launch-1.0 videotestsrc pattern=ball ! videoconvert ! fpsdisplaysink

# enable debug messages

```



### Nnstreamer

Useful commands

```bash
# List available plugins
apt-cache search nnstreamer
```

* `tensor_decoder mode=bounding_boxes` options for different models: https://github.com/nnstreamer/nnstreamer/blob/main/ext/nnstreamer/tensor_decoder/tensordec-boundingbox.c

* see details of element, such as `tensor_sink` with `gst-inspect-1.0 tensor_sink`
* example to validate model as part of nnstreamer pipeline https://github.com/nnstreamer/nnstreamer-example/blob/d42431f54507c197148648f3ad150ae32059c712/native/example_image_classification_tflite/nnstreamer_example_image_classification_tflite.py#L178


Test pipeline

```bash
# replay video from file
gst-launch-1.0 filesrc location=.dev/testvid.mp4 ! decodebin name=decode ! \
      videoconvert ! videoscale ! video/x-raw,width=640,height=360,format=RGB \
      ! videoconvert \
      ! ximagesink

# apply object detection
gst-launch-1.0 filesrc location=.dev/testvid.mp4 \
  ! decodebin name=decode \
  ! videoconvert \
  ! videoscale ! video/x-raw,width=640,height=360,format=RGB,framerate=30/1 \
  ! tee name=t \
  t. ! queue leaky=2 max-size-buffers=2 \
    ! videoscale ! video/x-raw,width=300,height=300,format=RGB \
    ! tensor_converter \
    ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 \
    ! tensor_filter framework=tensorflow-lite model=model/ssd_mobilenet_v2_coco.tflite \
    ! tensor_decoder mode=bounding_boxes option1=mobilenet-ssd option2=model/coco_labels_list.txt option3=model/box_priors.txt option4=640:360 option5=300:300 \
  ! compositor name=mix sink_0::zorder=2 sink_1::zorder=1 \
  ! videoconvert \
  ! ximagesink sync=false \
  t. ! queue leaky=2 max-size-buffers=10 ! mix.

# TEST w/ssd_mobilenet_v2_fpn_100_fp32.tflite
gst-launch-1.0 filesrc location=.dev/testvid.mp4 \
  ! decodebin name=decode \
  ! videoconvert \
  ! videoscale ! video/x-raw,width=640,height=360,format=RGB,framerate=30/1 \
  ! tee name=t \
  t. ! queue leaky=2 max-size-buffers=2 \
    ! videoscale ! video/x-raw,width=300,height=300,format=RGB \
    ! tensor_converter \
    ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 \
    ! tensor_filter framework=tensorflow-lite model=model/ssd_mobilenet_v2_fpn_100_fp32.tflite \
    ! tensor_decoder mode=bounding_boxes option1=mobilenet-ssd-postprocess option2=model/coco_labels_list.txt option4=640:360 option5=300:300 \
  ! compositor name=mix sink_0::zorder=2 sink_1::zorder=1 \
  ! videoconvert \
  ! ximagesink sync=false \
  t. ! queue leaky=2 max-size-buffers=10 ! mix.
```




### ROS2

Useful commands

```bash
# Get details about a message type
ros2 interface show sensor_msgs/msg/CompressedImage
> [message specs]

#ros2 bag info <filename>
ros2 bag info PMK_sad04_2022-01-26T13_12_08.391522587Z.db3
> [rosbag specs]

# run rosbag in loop
ros2 bag play --loop <filename>

# list publishing topics
ros2 topic list
```

`ros2 topic list` reveals publisher `sad04/camera/color/video` where `sad04` is the *drone ID*. This ID has to be available as env for docker container.


```bash
# add sourcing for ros2 command, activate pipenv
source /opt/ros/galactic/setup.bash
pipenv shell

# DEBUGGING NOTES
# show DEBUG messages for QT when running python
export QT_DEBUG_PLUGINS=1

## Filenot found error when including nnstreamer-python3 plugin
# try finding file
sudo find / -name libnnstreamer_converter_python3.so
echo $LD_LIBRARY_PATH
sudo ldconfig

# get meaningful error message
LD_PRELOAD=/usr/lib/nnstreamer/converters/libnnstreamer_converter_python3.so python ros_node.py

# check if file exists
python -c "from os import path; print(path.exists('/usr/lib/nnstreamer/converters/libnnstreamer_converter_python3.so'))"

sudo chown -R root:root /root/.local/share/virtualenvs/ROS_model_deployment-DptcJWh6

# inspect LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# make a plugin shared object available
export LD_LIBRARY_PATH=/usr/lib/nnstreamer/converters:$LD_LIBRARY_PATH

```



### Subscriber node with python

We need to ensure the same quality of service (QoS) for a publisher and subscriber. So, we set QoS settings for `ros2 bag play` with a yaml file and add the option `-qos-profile-overrides-path reliability_override.yaml`.

```yaml
# ./data/video/reliability_override.yaml
/talker:
  reliability: best_effort
  history: keep_all
```

ROS2 playback with QoS override to match the same QoS set within subscriber node in `ros_node.py`.

```bash
# ./data/video
ros2 bag play --loop --qos-profile-overrides-path reliability_override.yaml PMK_sad04_2022-01-26T13_12_08.391522587Z.db3
```


Define subscriber node with `sad04` as the drone id.

```python
# ./src/ros_node.py
class RosNode(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')

        self.subscription = self.create_subscription(
                msg_type=CompressedImage,
                topic='sad04/camera/color/video',
                callback=self.listener_callback,
                # history = 2: keep_all, reliability = 2: best_effort
                qos_profile=QoSProfile(history=2, reliability=2)
            )

        self.subscription  # prevent unused variable warning
```


```bash
# Run the subscriber node in another terminal
pipenv run ros_node.py
```
