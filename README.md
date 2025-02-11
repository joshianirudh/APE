<div style="display: flex; align-items: flex-start;">
  <img src="image/logo.png" width="40" height="40" style="margin-right: 10px;">
  <div>
    <h1 style="margin: 0;">APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding</h1>
    <p style="margin: 0;">[ICLR 2025]</p>
  </div>
</div>





## Installation

### Install APE attention
```bash
python setup.py install
```

### Install experiment dependencies
```bash
pip install -r requirements.txt
```

## Demo

Here, we demonstrate a basic demo of APE in `demo_APE.py`

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
