# <img src="assets/logo.png" width="40" height="40" align="top">  APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding [ICLR 2025]

### [[Paper](https://arxiv.org/abs/2502.05431)] | [[Project](https://infini-ai-lab.github.io/APE-Page)]

## TL;DR

We introduce APE for context-augmented generation with better efficiency and performance.

## Usage

### Environment Setup

```bash
conda create -yn ape python=3.10
conda activate ape

pip install -r requirements.txt
python setup.py install
```

## Run Context-augmented QA system with APE

```bash
CUDA_VISIBLE_DEVICES=0 python demo_APE.py  --model llama3-8b-instruct
```

## Experiments

To reproduce APE results on the retrieval-augmented generation (RAG) and in-context-learning (ICL) tasks, refer to the instruction and code available in the `experiments` directory.


## Citation

```bibtex
@inproceedings{yang2025ape,
  title={APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding},
  author={Yang, Xinyu and Chen, Tianqi and Chen, Beidi},
  booktitle={ICLR 2025},
  year={2025}
}
```
