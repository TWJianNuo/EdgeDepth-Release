# EdgeDepth

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in

> **The Edge of Depth: Explicit Constraints between Segmentation and Depth**
>
> [Shengjie Zhu](https://scholar.google.com/citations?user=4hHEXZkAAAAJ&hl=en), [Garrick Brazil](https://garrickbrazil.com/) and [Xiaoming Liu](https://www.cse.msu.edu/~liuxm/index2.html)
>
> [CVPR 2020](https://arxiv.org/abs/2004.00171)

## ⚙️ Setup
1. Compile Morphing operaiton: 

	We implement a customized Morphing Operation in our evaluation and training codes. You can still do training and evaluation without it with a sacrifice of performance. To enable it, you can do as follows:

	1. Guranttee your computer's cuda version the same as your pytorch cuda version.

	2. Type:
	```shell
	cd bnmorph
	python setup.py install
	cd ..
	```

	You should be able to successfully compile it if you can compile cuda codes in this [Pytorch Tutorial](https://github.com/pytorch/extension-cpp)

2. Prepare Kitti Data:
	We use Kitti Raw Dataset as well as predicted semantics label from [this Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.pdf).
	1. To download Kitti Raw Data
	```shell
	wget -i splits/kitti_archives_to_download.txt -P kitti_data/
	```

	2. Use thins [Link](https://drive.google.com/file/d/1OWBLheukaMiv_LfAgrSCmGEE9GBQb4yK/view?usp=sharing) to download precomputed semantics Label


## ⏳ Training
Training Code will be released soon.

## 📊 evaluation
1. Pretrained Model is available [here](https://drive.google.com/file/d/1Wu2oyoKqsvNHTMoDZF4wbek2gS9RAF2w/view?usp=sharing)

2. Precompute GroundTruth DepthMap
	```shell
	python export_gt_depth.py --data_path [Your Kitti Raw Data Address] --split eigen
	```

3. To Evaluate without using Morphing, use command:
	```shell
	python evaluate_depth.py --split eigen --dataset kitti --data_path [Your Kitti Raw Data Address] --load_weights_folder [Your Model Address] --eval_stereo \
	 --num_layers 50 --post_process
	```

	To Evaluate using Morphing, use command:
	```shell
	python evaluate_depth.py --split eigen --dataset kitti --data_path [Your Kitti Raw Data Address] --load_weights_folder [Your Model Address] --eval_stereo \
	 --num_layers 50 --post_process --bnMorphLoss --load_semantics --seman_path [Your Predicted Semantic Label Address]
	```

4. You should get performance similar to Entry "Ours" listed in the table:

	| Method Name     | Use Lidar Groundtruth? | Is morphed?  | KITTI abs. rel. error |  delta < 1.25  |
	|-----------------|----------------|--------------|-----------------------|----------------|
	|[BTS](https://arxiv.org/pdf/1907.10326.pdf)              | Yes            | No           | 0.091                | 0.904           |
	|[Depth Hints](http://openaccess.thecvf.com/content_ICCV_2019/papers/Watson_Self-Supervised_Monocular_Depth_Hints_ICCV_2019_paper.pdf)              | No            | No           | 0.096                | 0.890           |
	|Ours              | No            | No           | 0.091                | 0.898           |
	|Ours              | No            | Yes           | 0.090                | 0.899           |

## 🖼 Running on your own images

To run on your own images, run:

```bash
python test_simple.py --image_path <your_image_path>
  --model_path <your_model_path>
  --num_layers <18 or 50>
```

This will save depths as a numpy array (in original resolution), and a colormapped depth and disparity image.

## Acknowledgment
Quite a few our code base come from [Monodepth2](https://github.com/nianticlabs/monodepth2) and [Depth Hints](https://github.com/nianticlabs/depth-hints)
