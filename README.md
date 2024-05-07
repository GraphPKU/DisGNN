# Is Distance Matrix Enough for Geometric Deep Learning?

Pytorch implementation of the paper [Is Distance Matrix Enough for Geometric Deep Learning?](https://arxiv.org/abs/2302.05743)

## Dependencies

python=3.8

pytorch-lightning=2.0.1.post0

pytorch=2.0.0+cu118

torch_geometric=2.3.0


## Train and Test $k$-DisGNN

To train and test k-DisGNN, run 

```bash
python scripts/kDis_script.py [--model <model_name>] [--ds <dataset_name>] [--dname <data_name>] [--devices <device_ids>] [--data_dir <data_dir>] [--version <version>] [--resume] [--only_test] [--ckpt <checkpoint_path>] [--merge <merge hparam list>] [--use_wandb] [--proj_name <project_name>]
```

Arguments description:

+ --model: ["2EDis", "3EDis", "2FDis"], corresponds to the three models. By default, it is "2FDis".
+ --ds: ["md17", "qm9"], corresponds to the two datasets. By default, it is "md17".
+ --dname: In MD17, choices are the name of molecules, which can be found in [md17](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MD17.html?highlight=md17#torch_geometric.datasets.MD17); In QM9, choices are numbers in [0, 18], corresponding to the targets in [qm9](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html?highlight=qm9#torch_geometric.datasets.QM9).
+ --devices: The number of the GPU devices. For example, if you want to use GPU 0 and 1, please specify it as 0 1. If no devices are specified, the code will run on the GPU with minimum memory usage. If it's set to -1, the code will run on CPU.
+ --data_dir: The directory of the dataset. By default, it is "~/datasets/MD17".
+ --version: The version of the log. By default, it is "NO_VERSION".
+ --resume: Whether to resume the training from the latest checkpoint. By default, it is False.
+ --only_test: Whether to only test the model. By default, it is False.
+ --ckpt: The path of the checkpoint. By default, it is None.
+ --merge: The hparams to merge. For example, if you want to modify the hparam model_config.block_num to 4, then use --merge model_config.block_num 4.
+ --use_wandb: Whether to use wandb to log the training process. By default, it is False.
+ --proj_name: The name of the project in wandb. By default, it is None.

Hyperparameters are specified in the dictionary `hparams`.

For example, to run 2FDis on revised MD17 (revised ethanol) using gpu 0:
```bash
python scripts/kDis_script.py --model 2FDis --ds md17 --dname 'revised ethanol' --devices 0 --data_dir <your_data_dir> --version test 
```


## Run Counterexamples Verification Programs


To verify the counterexamples in Appendix A.1 and A.2 (the first two kinds), run

```bash
python counterexamples/verify_ce.py [--reg <regular_polyhedron>] [--verbose] [--visualize]
```

Arguments description:

+ --reg: ["12", "20", "cube_8", "cube_cube"], corresponds to the counterexamples sampled from different regular polyhedrons (or the combination of regular polyhedrons).
+ --verbose: store_true. Whether to show the detailed 1-WL-E process.
+ --visualize: store_true. Whether to visualize the counterexamples.


To verify the counterexamples in Appendix A.3 (the families), run 

```
python counterexamples/verify_families.py [--reg <regular_polyhedron>] [--verbose] [--num_of_layers <num_of_layers>]
```

Arguments description:

+ --reg: ["12", "20"], corresponds to the counterexamples sampled from different regular polyhedrons.
+ --verbose: store_true. Whether to show the detailed 1-WL-E process.
+ --num_of_layers: The number of layers of the counterexamples. By default, it is None, which means that the number of layers is randomly sampled from [1, 10].


To verify the Lemma A.3 (Stable states), run 

```bash
python counterexamples/verify_stable_state.py [--reg <regular_polyhedron>] [--verbose]
```

Arguments description:

+ --reg: ["12", "20"], corresponds to the counterexamples sampled from different regular polyhedrons.
+ --verbose: store_true. Whether to show the detailed 1-WL-E process.

## Cite
If you use the code please cite our paper.

```
@article{li2023distance,
  title={Is Distance Matrix Enough for Geometric Deep Learning?},
  author={Li, Zian and Wang, Xiyuan and Huang, Yinan and Zhang, Muhan},
  journal={arXiv preprint arXiv:2302.05743},
  year={2023}
}
```
