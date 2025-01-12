# Awesome Personalized Large Language Models (PLLMs)

This repository focuses on personalized large language models (LLMs) that leverage user data to generate tailored responses based on individual user preferences.

<table style="width: 100%;">
  <tr>
    <td style="width: 60%;">
      <ul>
        <li><a href="#personalized-prompting">Personalized Prompting</a></li>
        <ul>
          <li><a href="#profile-augmented-prompting-pag">Profile-Augmented</a></li>
          <li><a href="#retrieval-augmented-prompting-rag">Retrieval-Augmented</a></li>
          <li><a href="#soft-fused-prompting">Soft-Fused</a></li>
        </ul>
        <li><a href="#personalized-adaptation">Personalized Adaptation</a></li>
        <li><a href="#personalized-alignment">Personalized Alignment</a></li>
        <li><a href="#benchmark">Benchmark</a></li>
      </ul>
    </td>
    <td>
      <img src="Figures/framework.png" alt="Awesome Personalized LLMs" width="700">
    </td>
  </tr>
</table>

---

## Personalized Prompting

### Profile-Augmented Prompting (PAG)

|  Method   |                                             Paper Title                                              |   Published At    |                                                                                                                                  Link                                                                                                                                  |
| :-------: | :--------------------------------------------------------------------------------------------------: | :---------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| *Cue-CoT* |      Cue-cot:Chain-of-thought prompting for responding to in-depth dialogue questions with llms      | EMNLP'23 Findings |                        [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2305.11792) [![Github](https://img.shields.io/github/stars/ruleGreen/Cue-CoT.svg?style=social&label=Github)](https://github.com/ruleGreen/Cue-CoT)                        |
|  *PAG*  |  Integrating Summarization and Retrieval for Enhanced Personalization via Large Lanuage Models |      CIKM'23       |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2310.20081)               |
|  *ONCE*  | ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models |      WSDM'24       |                     [![Paper](https://img.shields.io/badge/Paper-blue)]([https://arxiv.org/abs/2306.07206](https://arxiv.org/abs/2305.06566)) [![Github](https://img.shields.io/github/stars/Jyonn/ONCE.svg?style=social&label=Github)](https://github.com/Jyonn/ONCE) |
|  *Matryoshka* | Matryoshka: Learning to Drive Black-Box LLMs with LLMs |     Arxiv'24       | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2410.20749) [![Github](https://img.shields.io/github/stars/lichangh20/Matryoshka.svg?style=social&label=Github)](https://github.com/lichangh20/Matryoshka) |
|  *RewriterSlRl* | Learning to Rewrite Prompts for Personalized Text Generation |    WWW '24   | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2310.00152)  |
|  _COS_ | CoS: Enhancing Personalization and Mitigating Bias with Context Steering |   Arxiv'24         |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2405.01768) [![Github](https://img.shields.io/github/stars/sashrikap/context-steering.svg?style=social&label=Github)](https://github.com/sashrikap/context-steering) |


### Retrieval-Augmented Prompting (RAG)

|  Method   |                                             Paper Title                                              |   Published At    |                                                                                                                                  Link                                                                                                                                  |
| :-------: | :--------------------------------------------------------------------------------------------------: | :---------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  _MemPrompt_ | Memory-assisted prompt editing to improve GPT-3 after deployment |     EMNLP'22       | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2201.06009) [![Github](https://img.shields.io/github/stars/madaan/memprompt.svg?style=social&label=Github)](https://github.com/madaan/memprompt) |
| _TeachMe_  | Towards Teachable Reasoning Systems: Using a Dynamic Memory of User Feedback for Continual System Improvement |    EMNLP'22        | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2204.13074) |
|  _MaLP_ | LLM-based Medical Assistant Personalization with Short- and Long-Term Memory Coordination |   NAACL'24         |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2309.11696) [![Github](https://img.shields.io/github/stars/MatthewKKai/MaLP.svg?style=social&label=Github)](https://github.com/MatthewKKai/MaLP) |
|   | Long-Term Memory for Large Language Models Through Topic-Based Vector Database |     IALP'23       |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://ieeexplore.ieee.org/document/10337079) |
| *LD_Agent* | Hello Again! LLM-powered Personalized Agent for Long-term Dialogue |   Arxiv'24         |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2406.05925) [![Github](https://img.shields.io/github/stars/leolee99/LD-Agent.svg?style=social&label=Github)](https://github.com/leolee99/LD-Agent) |
|  _MemoRAG_ | MemoRAG: Moving towards Next-Gen RAG Via Memory-Inspired Knowledge Discovery |   Arxiv'24         |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2409.05591) [![Github](https://img.shields.io/github/stars/qhjqhj00/MemoRAG.svg?style=social&label=Github)](https://github.com/qhjqhj00/MemoRAG) |
| _LaMP_  | LaMP: When Large Language Models Meet Personalization |    ACL'24        | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2304.11406) [![Github](https://img.shields.io/github/stars/LaMP-Benchmark/LaMP.svg?style=social&label=Github)](https://github.com/LaMP-Benchmark/LaMP) |
| _MSP_  | Less is More: Learning to Refine Dialogue History for Personalized Dialogue Generation |    ACL'22        |   [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2204.08128) [![Github](https://img.shields.io/github/stars/bangbangbang12315/MSP.svg?style=social&label=Github)](https://github.com/bangbangbang12315/MSP/tree/release) |
|  _AuthorPred_ | Teach LLMs to Personalize -- An Approach inspired by Writing Education  |     Arxiv'23      |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2308.07968) |
|  _PEARL_ | Pearl: Personalizing Large Language Model Writing Assistants with Generation-Calibrated Retrievers |    CustomNLP4U@  EMNLP'24        |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2311.09180) |
|  *ROPG / RSPG*| Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation |  Arxiv'24          |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2404.05970) |
|  _HYDRA_ | HYDRA: Model Factorization Framework for Black-Box LLM Personalization | NeurIPS'24           |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2406.02888) [![Github](https://img.shields.io/github/stars/night-chen/HYDRA.svg?style=social&label=Github)](https://github.com/night-chen/HYDRA) |
|  *RECAP*  | RECAP: Retrieval-Enhanced Context-Aware Prefix Encoder for Personalized Dialogue Response Generation |      ACL'23       |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2306.07206) [![Github](https://img.shields.io/github/stars/isi-nlp/RECAP.svg?style=social&label=Github)](https://github.com/isi-nlp/RECAP/tree/main)                     |

### Soft-Fused Prompting

|  Method   |                                             Paper Title                                              |   Published At    |                                                                                                                                  Link                                                                                                                                  |
| :-------: | :--------------------------------------------------------------------------------------------------: | :---------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| _UEM_  | User Embedding Model for Personalized Language Prompting |     PERSONALIZE@   ACL'24       |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2401.04858) |
| _PERSOMA_  | PERSOMA: PERsonalized SOft ProMpt Adapter Architecture for Personalized Language Prompting |  GenAIRecP@  KDD'24          |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2408.00960)|
| *REGEN*  | Beyond Retrieval: Generating Narratives in Conversational Recommender Systems |    Arxiv'24        | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2410.16780) |
| *PeaPOD*  | Preference Distillation for Personalized Generative Recommendation |    Arxiv'24        |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2407.05033) [![Github](https://img.shields.io/github/stars/jeromeramos70/peapod.svg?style=social&label=Github)](https://github.com/jeromeramos70/peapod) |
|  *User-LLM* | User-LLM: Efficient LLM Contextualization with User Embeddings |    Arxiv'24        |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2402.13598) |
|  *RECAP*  | RECAP: Retrieval-Enhanced Context-Aware Prefix Encoder for Personalized Dialogue Response Generation |      ACL'23       |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2306.07206) [![Github](https://img.shields.io/github/stars/isi-nlp/RECAP.svg?style=social&label=Github)](https://github.com/isi-nlp/RECAP/tree/main)                     |
| GSMN  | Personalized Response Generation via Generative Split Memory Network |     NAACL'21       |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://aclanthology.org/2021.naacl-main.157/) |

## Personalized Adaptation

|  Method   |                                             Paper Title                                              |   Published At    |                                                                                                                                  Link                                                                                                                                  |
| :-------: | :--------------------------------------------------------------------------------------------------: | :---------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      |                   Personalized Dialogue Generation with Persona-Adaptive Attention                   |      AAAI'23      | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2210.15088) [![Github](https://img.shields.io/github/stars/hqsiswiliam/persona-adaptive-attention.svg?style=social&label=Github)](https://github.com/hqsiswiliam/persona-adaptive-attention) |
|   | |            |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2306.07206) [![Github](https://img.shields.io/github/stars/isi-nlp/RECAP.svg?style=social&label=Github)](https://github.com/isi-nlp/RECAP/tree/main) |

## Personalized Alignment

## Benchmark
|                      Paper Title                      |                     Affiliation                      | Published At |                                                                                                                                                         Link                                                                                                                                                         |
| :---------------------------------------------------: | :--------------------------------------------------: | :----------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LaMP: When Large Language Models Meet Personalization | University of Massachusetts Amherst; Google Research |   ACL'24   | [![Home](https://img.shields.io/badge/Home-red)](https://lamp-benchmark.github.io/) [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2304.11406) [![Github](https://img.shields.io/github/stars/LaMP-Benchmark/LaMP.svg?style=social&label=Github)](https://github.com/LaMP-Benchmark/LaMP) |
| REGEN  | Beyond Retrieval: Generating Narratives in Conversational Recommender Systems |    Arxiv'24        | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2410.16780) |

