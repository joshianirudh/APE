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

## Run Context-augmented Question Answering with APE

By default, the temperature and scaling factor are set to 0.9, preserving over 90% performance on few-shot tasks.

```bash
CUDA_VISIBLE_DEVICES=0 python demo_APE.py --model llama3-8b-instruct
```

## Experiments

To reproduce the APE results for retrieval-augmented generation (RAG) and in-context learning (ICL) tasks in Section 5, please follow the instructions and use the code provided in the `experiments` directory.

## TODOs
We will release the code and data in the following order, please stay tuned!

- [x] Release core code of APE, including Llama-3, Llama-3.1, Mistral-v0.3, and Gemma-2.
- [x] Release RAG and ICL evaluation code.
- [x] Release APE context-augmented QA demo
- [ ] Incorporate APE into efficient inference engine

## Citation

If you find APE useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@inproceedings{yang2025ape,
  title={APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding},
  author={Yang, Xinyu and Chen, Tianqi and Chen, Beidi},
  booktitle={ICLR 2025},
  year={2025}
}
```
