# Multi-Aspect Reconstruction and Multi-Object Tracking

This repository contains all the steps necessary to tackle the multi-view multi-people tracking problem. The pipeline consists of 5 steps: calibration, annotation, training, inference and tracking.

![Image showing the validation of the training.](images/full_pipeline.png)
<!-- <img src="images/full_pipeline.png" alt="drawing" width="100%" height="200"/> -->
## System setup

More information about system setup can be found in the [setup Readme](doc/!-setup.md).

## Running the pipeline

Each step of the pipeline is describe in detail in [documentation](doc).

* [0-calibration](doc/0-calibration.md)
* [1-annotation](doc/1-annotation.md)
* [2-training](doc/2-training.md)
* [3-inference](doc/3-inference.md)
* [4-tracking](doc/4-tracking.md)



## Reference
If you found this code useful, please cite us:

    @misc{MARMOT2023,
    author        = {Engilberge, Martin and Grosche, Wilke and Fua, Pascal},
    year          = {2023},
    title         = {Multi-Aspect Reconstruction and Multi-Object Tracking},
    howpublished = {\url{https://github.com/wgrosche/MARMOT/}}
    }

    @inproceedings{engilber2023multi,
	  title={Multi-view Tracking Using Weakly Supervised Human Motion Prediction},
	  author={Engilberge, Martin and Liu, Weizhe and Fua, Pascal},
	  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
	  year={2023}
	}

    @inproceedings{engilber2023two,
        title={Two-level Data Augmentation for Calibrated Multi-view Detection},
        author={Engilberge, Martin and Shi, Haixin and Wang, Zhiye and Fua, Pascal},
        booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
        year={2023}
    }


The annotation tool is based on the following work:
* [Multicam-Gt](https://github.com/M-Eng/multicam-gt)

## License
By downloading this program, you commit to comply with the license as stated in the LICENSE file.
