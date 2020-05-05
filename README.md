# FastEstimator

[![Build Status](http://jenkins.fastestimator.org:8080/buildStatus/icon?subject=PR-test&job=Pull+Requests)](http://jenkins.fastestimator.org:8080/job/Pull%20Requests/)
[![Build Status](http://jenkins.fastestimator.org:8080/buildStatus/icon?subject=nightly-build&job=nightly)](http://jenkins.fastestimator.org:8080/job/nightly/)

FastEstimator is a high-level deep learning library built on tensorflow 2 and pytorch. With the help of FastEstimator, you can easily build a high-performance deep learning model and run it anywhere. :wink:


## Prerequisites:
* Python >= 3.6

## Installation:
### 1. Install Dependencies:
* Windows (CPU):
    ```
    pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    ```

* Linux (CPU/GPU):
    ```
    apt-get install libglib2.0-0 libsm6 libxrender1 libxext6
    ```

* Mac (CPU):
    ```
    echo No dependency needed ":)"
    ```

### 2. Install FastEstimator:
* Stable:
    ```
    pip install fastestimator==1.0.0
    ```
* Most Recent:
    ```
    pip install fastestimator-nightly
    ```


## Docker Hub
Docker container creates isolated virtual environment that shares resources with host machine. Docker provides an easy way to set up FastEstimator environment, users can pull image from [Docker Hub](https://hub.docker.com/r/fastestimator/fastestimator/tags).

* GPU: `docker pull fastestimator/fastestimator:1.0.0-gpu`
* CPU: `docker pull fastestimator/fastestimator:1.0.0-cpu`



## Useful Links:
* [Website](https://www.fastestimator.org): More info about FastEstimator API and news.
* [Tutorial Series](https://github.com/fastestimator/fastestimator/tree/master/tutorial): Everything you need to know about FastEstimator.
* [Application Hub](https://github.com/fastestimator/fastestimator/tree/master/apphub): End-to-end deep learning examples in FastEstimator.



## Citation
Please cite FastEstimator in your publications if it helps your research:
```
@misc{dong2019fastestimator,
    title={FastEstimator: A Deep Learning Library for Fast Prototyping and Productization},
    author={Xiaomeng Dong and Junpyo Hong and Hsi-Ming Chang and Michael Potter and Aritra Chowdhury and
            Purujit Bahl and Vivek Soni and Yun-Chan Tsai and Rajesh Tamada and Gaurav Kumar and Caroline Favart and
            V. Ratna Saripalli and Gopal Avinash},
    year={2019},
    eprint={1910.04875},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License
[Apache License 2.0](https://github.com/fastestimator/fastestimator/blob/master/LICENSE)
