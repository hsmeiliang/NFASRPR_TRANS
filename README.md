## Applying Non-Fourier and AST-Structural Relative Position Representations in Transformer for Source Code Summarization
>[paper link](https://ieeexplore.ieee.org/document/10400421)

### Installing NFASRPR-TRANS

Require `Linux` and `Python 3.6` or higher. It also requires installing `PyTorch` version 1.3 or higher. `CUDA` is strongly recommended for speed, but not necessary.

For generating attention heatmaps, you should also install `matplotlib` and `seaborn`, and checking the `### for draw graph ###` part in the `train.py` and `multi_attn.py` file.

Its other dependencies are listed in `requirements.txt`.

Run the following commands to clone the repository and install NFASRPR-TRANS:

```
git clone https://github.com/hsmeiliang/NFASRPR_TRANS.git
cd NFASRPR_TRANS; 
```

```
conda create --name NFASRPR python=3.6.9
```

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

```
pip install psutil
```

if "AttributeError: 'Meteor' object has no attribute 'meteor_p' "
you need to install [java jdk](https://blog.csdn.net/qq_36468195/article/details/115630818)

```
pip install -r requirements.txt; python setup.py develop
```

### Download Java and Python dataset
Where, choices for DATASET_NAME are ["java", "python"].

```
$ gdown '1-1cmfpYYzggM8Oe0ViHNcXfWo5jFBJmZ' --output data.tar.gz
$ tar -zxf data.tar.gz
$ cd DATASET_NAME
$ tar -zxf adjacency.tar.gz
$ tar -zxf pre_rev_positions.tar.gz
```

### Training/Testing Models

 To perform training and evaluation, first go the scripts directory associated with the target dataset.

```
$ cd  scripts/DATASET_NAME
```


To train/evaluate a model, run:

```
$ bash script_name.sh GPU_ID MODEL_NAME
```

For example, to train/evaluate the transformer model, run:

```
$ bash transformer.sh 0 code2jdoc
```

#### Generated log files

While training and evaluating the models, a list of files are generated inside a `tmp` directory. The files are as follows.

- **MODEL_NAME.mdl**
  - Model file containing the parameters of the best model.
- **MODEL_NAME.mdl.checkpoint**
  - A model checkpoint, in case if we need to restart the training.
- **MODEL_NAME.txt**
  - Log file for training.
- **MODEL_NAME.json**
  - The predictions and gold references are dumped during validation.
- **MODEL_NAME_test.txt**
  - Log file for evaluation (greedy).
- **MODEL_NAME_test.json** 
  - The predictions and gold references are dumped during evaluation (greedy).


**[Structure of the JSON files]** Each line in a JSON file is a JSON object. An example is provided below.

```json 
{
    "id": 0,
    "code": "private int current Depth ( ) { try { Integer one Based = ( ( Integer ) DEPTH FIELD . get ( this ) ) ; return one Based - NUM ; } catch ( Illegal Access Exception e ) { throw new Assertion Error ( e ) ; } }",
    "predictions": [
        "returns a 0 - based depth within the object graph of the current object being serialized ."
    ],
    "references": [
        "returns a 0 - based depth within the object graph of the current object being serialized ."
    ],
    "bleu": 1,
    "rouge_l": 1
}
```


#### Running experiments on CPU/GPU/Multi-GPU

- If GPU_ID is set to -1, CPU will be used.
- If GPU_ID is set to one specific number (e.g., 0), only one GPU will be used.

### Acknowledgement

We borrowed and modified code from [NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum), [SiT](https://github.com/gingasan/sit3). We would like to expresse our gratitdue for the authors of these repositeries.


### Citation

```

```

