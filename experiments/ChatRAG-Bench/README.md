---
language:
- en
license:
- other
size_categories:
- 1K<n<10K
tags:
- RAG
- ChatRAG
- conversational QA
- multi-turn QA
- QA with context
- evaluation
configs:
- config_name: coqa
  data_files:
    - split: dev
      path: data/coqa/*
- config_name: inscit
  data_files:
    - split: dev
      path: data/inscit/*
- config_name: inscit
  data_files:
    - split: dev
      path: data/inscit/*
- config_name: topiocqa
  data_files:
    - split: dev
      path: data/topiocqa/*
- config_name: hybridial
  data_files:
    - split: test
      path: data/hybridial/*
- config_name: doc2dial
  data_files:
    - split: test
      path: data/doc2dial/test.json
- config_name: quac
  data_files:
    - split: test
      path: data/quac/test.json
- config_name: qrecc
  data_files:
    - split: test
      path: data/qrecc/test.json
- config_name: doqa_cooking
  data_files:
    - split: test
      path: data/doqa/test_cooking.json
- config_name: doqa_movies
  data_files:
    - split: test
      path: data/doqa/test_movies.json
- config_name: doqa_travel
  data_files:
    - split: test
      path: data/doqa/test_travel.json
- config_name: sqa
  data_files:
    - split: test
      path: data/sqa/test.json
---

## ChatRAG Bench
ChatRAG Bench is a benchmark for evaluating a model's conversational QA capability over documents or retrieved context. ChatRAG Bench are built on and derived from 10 existing datasets: Doc2Dial, QuAC, QReCC, TopioCQA, INSCIT, CoQA, HybriDialogue, DoQA, SQA, ConvFinQA. ChatRAG Bench covers a wide range of documents and question types, which require models to generate responses from long context, comprehend and reason over tables, conduct arithmetic calculations, and indicate when questions cannot be found within the context. The details of this benchmark are described in [here](https://arxiv.org/pdf/2401.10225). **For more information about ChatQA, check the [website](https://chatqa-project.github.io/)!**

## Other Resources
[Llama3-ChatQA-1.5-8B](https://huggingface.co/nvidia/Llama3-ChatQA-1.5-8B) &ensp; [Llama3-ChatQA-1.5-70B](https://huggingface.co/nvidia/Llama3-ChatQA-1.5-70B) &ensp; [Training Data](https://huggingface.co/datasets/nvidia/ChatQA-Training-Data) &ensp; [Retriever](https://huggingface.co/nvidia/dragon-multiturn-query-encoder) &ensp; [Website](https://chatqa-project.github.io/) &ensp; [Paper](https://arxiv.org/pdf/2401.10225)

## Benchmark Results

### Main Results
| | ChatQA-1.0-7B | Command-R-Plus | Llama3-instruct-70b | GPT-4-0613 | GPT-4-Turbo | ChatQA-1.0-70B | ChatQA-1.5-8B | ChatQA-1.5-70B |
| -- |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Doc2Dial | 37.88 | 33.51 | 37.88 | 34.16 | 35.35 | 38.90 | 39.33 | 41.26 |
| QuAC | 29.69 | 34.16 | 36.96 | 40.29 | 40.10 | 41.82 | 39.73 | 38.82 |
| QReCC | 46.97 | 49.77 | 51.34 | 52.01 | 51.46 | 48.05 | 49.03 | 51.40 |
| CoQA | 76.61 | 69.71 | 76.98 | 77.42 | 77.73 | 78.57 | 76.46 | 78.44 |
| DoQA | 41.57 | 40.67 | 41.24 | 43.39 | 41.60 | 51.94 | 49.60 | 50.67 |
| ConvFinQA | 51.61 | 71.21 | 76.6 | 81.28 | 84.16 | 73.69 | 78.46 | 81.88 |
| SQA | 61.87 | 74.07 | 69.61 | 79.21 | 79.98 | 69.14 | 73.28 | 83.82 |
| TopioCQA | 45.45 | 53.77 | 49.72 | 45.09 | 48.32 | 50.98 | 49.96 | 55.63 |
| HybriDial* | 54.51 | 46.7 | 48.59 | 49.81 | 47.86 | 56.44 | 65.76 | 68.27 |
| INSCIT | 30.96 | 35.76 | 36.23 | 36.34 | 33.75 | 31.90 | 30.10 | 32.31 |
| Average (all) | 47.71 | 50.93 | 52.52 | 53.90 | 54.03 | 54.14 | 55.17 | 58.25 |
| Average (exclude HybriDial) | 46.96 | 51.40 | 52.95 | 54.35 | 54.72 | 53.89 | 53.99 | 57.14 |

Note that ChatQA-1.5 is built based on Llama-3 base model, and ChatQA-1.0 is built based on Llama-2 base model. ChatQA-1.5 models use HybriDial training dataset. To ensure fair comparison, we also compare average scores excluding HybriDial.

### Evaluation of Unanswerable Scenario

ChatRAG Bench also includes evaluations for the unanswerable scenario, where we evaluate models' capability to determine whether the answer to the question can be found within the given context. Equipping models with such capability can substantially decrease the likelihood of hallucination.

| | GPT-3.5-turbo-0613 | Command-R-Plus | Llama3-instruct-70b | GPT-4-0613 | GPT-4-Turbo | ChatQA-1.0-70B | ChatQA-1.5-8B | ChatQA-1.5-70B |
| -- |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Avg-Both | 73.27 | 68.11 | 76.42 | 80.73 | 80.47 | 77.25 | 75.57 | 71.86 |
| Avg-QuAC | 78.335 | 69.605 | 81.285 | 87.42 | 88.73 | 80.76 | 79.3 | 72.59 |
| QuAC (no*) | 61.91 | 41.79 | 66.89 | 83.45 | 80.42 | 77.66 | 63.39 | 48.25 |
| QuAC (yes*) | 94.76 | 97.42 | 95.68 | 91.38 | 97.03 | 83.85 | 95.21 | 96.93 |
| Avg-DoQA | 68.21 | 66.62 | 71.555 | 74.05 | 72.21 | 73.74 | 71.84 | 71.125 |
| DoQA (no*) | 51.99 | 46.37 | 60.78 | 74.28 | 72.28 | 68.81 | 62.76 | 52.24 |
| DoQA (yes*) | 84.43 | 86.87 | 82.33 | 73.82 | 72.13 | 78.67 | 80.92 | 90.01 |

We use QuAC and DoQA datasets which have such unanswerable cases to evaluate such capability. We use both answerable and unanswerable samples for this evaluation. Specifically, for unanswerable case, we consider the model indicating that the question cannot be answered as correct, and as for answerable cases, we consider the model not indicating the question is unanswerable as correct (i.e., the model giving an answer). In the end, we calculate the average accuracy score of unanswerable and answerable cases as the final metric.

## Evaluation Scripts
We also open-source the [scripts](https://huggingface.co/datasets/nvidia/ChatRAG-Bench/tree/main/evaluation) for running and evaluating on ChatRAG (including the unanswerable scenario evaluations).

## License
The ChatRAG are built on and derived from existing datasets. We refer users to the original licenses accompanying each dataset.

## Correspondence to
Zihan Liu (zihanl@nvidia.com), Wei Ping (wping@nvidia.com)

## Citation
<pre>
@article{liu2024chatqa,
  title={ChatQA: Surpassing GPT-4 on Conversational QA and RAG},
  author={Liu, Zihan and Ping, Wei and Roy, Rajarshi and Xu, Peng and Lee, Chankyu and Shoeybi, Mohammad and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2401.10225},
  year={2024}}
</pre>


## Acknowledgement
We would like to give credits to all the works constructing the datasets we use for evaluating ChatQA. If you use these resources, please also cite all the datasets you use.
<pre>
@inproceedings{feng2020doc2dial,
  title={doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset},
  author={Feng, Song and Wan, Hui and Gunasekara, Chulaka and Patel, Siva and Joshi, Sachindra and Lastras, Luis},
  booktitle={Proceedings of the 2020 Conference on EMNLP},
  year={2020}
}
@inproceedings{choi2018quac,
  title={QuAC: Question Answering in Context},
  author={Choi, Eunsol and He, He and Iyyer, Mohit and Yatskar, Mark and Yih, Wen-tau and Choi, Yejin and Liang, Percy and Zettlemoyer, Luke},
  booktitle={Proceedings of the 2018 Conference on EMNLP},
  year={2018}
}
@inproceedings{anantha2021open,
  title={Open-Domain Question Answering Goes Conversational via Question Rewriting},
  author={Anantha, Raviteja and Vakulenko, Svitlana and Tu, Zhucheng and Longpre, Shayne and Pulman, Stephen and Chappidi, Srinivas},
  booktitle={Proceedings of the 2021 Conference on NAACL},
  year={2021}
}
@article{reddy2019coqa,
  title={CoQA: A Conversational Question Answering Challenge},
  author={Reddy, Siva and Chen, Danqi and Manning, Christopher D},
  journal={Transactions of the Association for Computational Linguistics},
  year={2019}
}
@inproceedings{campos2020doqa,
  title={DoQA-Accessing Domain-Specific FAQs via Conversational QA},
  author={Campos, Jon Ander and Otegi, Arantxa and Soroa, Aitor and Deriu, Jan Milan and Cieliebak, Mark and Agirre, Eneko},
  booktitle={Proceedings of the 2020 Conference on ACL},
  year={2020}
}
@inproceedings{chen2022convfinqa,
  title={ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering},
  author={Chen, Zhiyu and Li, Shiyang and Smiley, Charese and Ma, Zhiqiang and Shah, Sameena and Wang, William Yang},
  booktitle={Proceedings of the 2022 Conference on EMNLP},
  year={2022}
}
@inproceedings{iyyer2017search,
  title={Search-based neural structured learning for sequential question answering},
  author={Iyyer, Mohit and Yih, Wen-tau and Chang, Ming-Wei},
  booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
  year={2017}
}
@article{adlakha2022topiocqa,
  title={TopiOCQA: Open-domain Conversational Question Answering with Topic Switching},
  author={Adlakha, Vaibhav and Dhuliawala, Shehzaad and Suleman, Kaheer and de Vries, Harm and Reddy, Siva},
  journal={Transactions of the Association for Computational Linguistics},
  year={2022}
}
@inproceedings{nakamura2022hybridialogue,
  title={HybriDialogue: An Information-Seeking Dialogue Dataset Grounded on Tabular and Textual Data},
  author={Nakamura, Kai and Levy, Sharon and Tuan, Yi-Lin and Chen, Wenhu and Wang, William Yang},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2022},
  year={2022}
}
@article{wu2023inscit,
  title={InSCIt: Information-Seeking Conversations with Mixed-Initiative Interactions},
  author={Wu, Zeqiu and Parish, Ryu and Cheng, Hao and Min, Sewon and Ammanabrolu, Prithviraj and Ostendorf, Mari and Hajishirzi, Hannaneh},
  journal={Transactions of the Association for Computational Linguistics},
  year={2023}
}
</pre>