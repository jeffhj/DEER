# DEER🦌

The code and data for "[DEER: Descriptive Knowledge Graph for Explaining Entity Relationships](https://arxiv.org/pdf/2205.10479.pdf)" (EMNLP '22)

## Introduction

We propose **DEER**🦌 (**D**escriptive Knowledge Graph for **E**xplaining **E**ntity **R**elationships) - an open and informative form of modeling entity relationships. In DEER, relationships between entities are represented by free-text relation descriptions. For instance, the relationship between entities of machine learning and algorithm is represented as "*Machine learning explores the study and construction of algorithms that can learn from and make predictions on data.*"

![image](https://user-images.githubusercontent.com/44779294/196641224-5f984fca-3fd1-46bb-b4ff-5b00fc6235b7.png)

## Requirements

See `requirements.txt`

## Data

Data are available on this [link](https://osf.io/8chyx/?view_only=629c47293bca4435ae6747211795a96a)

## Relation Description Extraction

1. Suppose you already have downloaded the Wikipedia dump and preprocess it with WikiExtractor, you may extract the candidate relation descriptions by running.

   ```
   python extract_wiki.py preprocess_wikipedia [Wikipedia/folder]
   ```

   This code may run 26 hours to finish and you will get a *digraph.pickle* file under an *extract_wiki* folder.

   You may directly download the *digraph.pickle* file from this [link](https://osf.io/8chyx/?view_only=629c47293bca4435ae6747211795a96a)

2. \[Option\] The DEER is a directed graph. You may find it useful to have a undirected version for later steps. To convert the directed graph into an undirected one, you need to first make sure there exist a file *extract_wiki/digraph.pickle*, then run

   ```
   python extract_wiki.py convert_dir_to_undir
   ```

   and you will get a *graph.pickle* file under an *extract_wiki* folder.

3. To generate the dataset, run

   ```
   python extract_wiki.py collect_sample [source graph directed(true/false)] [target graph directed(true/false)] [context_threshold]
   ```

   To get the dataset used to train the model in the paper, just use the default value by running

   ```
   python extract_wiki.py collect_sample false true 0.75
   ```

   and you will get a dataset file named *dataset_0.50_undir2dir_0.75.json* under the *extract_wiki* folder.

4. To get the *train/dev/test* split, run

   ```
   python extract_wiki.py split_dataset [dataset_file] [prefix]
   ```

   where \[dataset_file\] is the dataset file generated from last step and \[prefix\] is the string prepended to the generated files, which is optional. You will get \[prefix\]_train/dev/test.json files when this command is finished.

## Relation Description Generation

### Train

**Train model**

```
python model/train_reader.py --config config/train3.yaml
```

### Test

Generate relation descriptions for entity pairs in test set

```
python model/test_reader.py --config config/test3.yaml
```

### Evaluation

Use Automatic metrics to evaluate the generated sentences in test set

```
python split_eval.py [model_path/in/test/config]/final_output.tsv
```

A *output.txt* and a *target.txt* file will be generated after running the above command. Then, run

```
bash RM-scorer.sh output.txt target.txt
```

to compare the generation outputs with the targets.

## Citation

The details of this repo are described in the following paper. If you find this repo useful, please kindly cite it:

```
@inproceedings{huang2022deer,
  title={DEER: Descriptive Knowledge Graph for Explaining Entity Relationships},
  author={Huang, Jie and Zhu, Kerui and Chang, Kevin Chen-Chuan and Xiong, Jinjun and Hwu, Wen-mei},
  booktitle={Proceedings of EMNLP},
  year={2022}
}
```
