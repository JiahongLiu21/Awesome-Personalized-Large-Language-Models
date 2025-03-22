# Awesome Personalized Large Language Models (PLLMs)
[![](https://img.shields.io/badge/📑-Survey_Paper-blue)](https://arxiv.org/abs/2502.11528)
[![Awesome](https://awesome.re/badge.svg)](https://github.com/JiahongLiu21/Awesome-Personalized-Large-Language-Models)
![](https://img.shields.io/github/last-commit/JiahongLiu21/Awesome-Personalized-Large-Language-Models) 
![](https://img.shields.io/badge/PRs-Welcome-red)
<img src="https://badges.pufler.dev/visits/jiahongliu21/Awesome-Personalized-Large-Language-Models?style=flat-square&logo=github">
![](https://img.shields.io/github/stars/JiahongLiu21/Awesome-Personalized-Large-Language-Models?color=yellow)
![](https://img.shields.io/github/forks/JiahongLiu21/Awesome-Personalized-Large-Language-Models?color=lightblue)

This repository focuses on personalized large language models (LLMs) that leverage user data to generate tailored responses based on individual user preferences. Our survey paper can be accessed on arXiv and we welcome any discussion and feedback! 

* **A Survey of Personalized Large Language Models: Progress and Future Directions** [https://arxiv.org/abs/2502.11528]

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
        <ul>
          <li><a href="#one-peft-all-users-one4all">One4All</a></li>
          <li><a href="#one-peft-per-users-one4one">One4One</a></li>
        </ul>
        <li><a href="#personalized-alignment">Personalized Alignment</a></li>
        <ul>
          <li><a href="#data-construction">Data</a></li>
          <li><a href="#optimization">Optimization</a></li>
        </ul>
        <li><a href="#analysis">Analysis</a></li>
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
|  _DPL_ | Measuring What Makes You Unique: Difference-Aware User Modeling for Enhancing LLM Personalization |   Arxiv'25         |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2503.02450) [![Github](https://img.shields.io/github/stars/SnowCharmQ/DPL.svg?style=social&label=Github)](https://github.com/SnowCharmQ/DPL) |
|  _COS_ | CoS: Enhancing Personalization and Mitigating Bias with Context Steering |   Arxiv'24         |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2405.01768) [![Github](https://img.shields.io/github/stars/sashrikap/context-steering.svg?style=social&label=Github)](https://github.com/sashrikap/context-steering) |


### Retrieval-Augmented Prompting (RAG)

|  Method   |                                             Paper Title                                              |   Published At    |                                                                                                                                  Link                                                                                                                                  |
| :-------: | :--------------------------------------------------------------------------------------------------: | :---------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  _MemPrompt_ | Memory-assisted prompt editing to improve GPT-3 after deployment |     EMNLP'22       | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2201.06009) [![Github](https://img.shields.io/github/stars/madaan/memprompt.svg?style=social&label=Github)](https://github.com/madaan/memprompt) |
| _TeachMe_  | Towards Teachable Reasoning Systems: Using a Dynamic Memory of User Feedback for Continual System Improvement |    EMNLP'22        | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2204.13074) |
|  _MaLP_ | LLM-based Medical Assistant Personalization with Short- and Long-Term Memory Coordination |   NAACL'24         |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2309.11696) [![Github](https://img.shields.io/github/stars/MatthewKKai/MaLP.svg?style=social&label=Github)](https://github.com/MatthewKKai/MaLP) |
|   | Long-Term Memory for Large Language Models Through Topic-Based Vector Database |     IALP'23       |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://ieeexplore.ieee.org/document/10337079) |
| *TACITREE*| Toward Multi-Session Personalized Conversation: A Large-Scale Dataset and Hierarchical Tree Framework for Implicit Reasoning | Arxiv'25 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/pdf/2503.07018) |
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
| *PPlug*  | LLMs + Persona-Plug = Personalized LLMs |     Arxiv'24       |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/pdf/2409.11901/) |
| *ComMer*  | ComMer: a Framework for Compressing and Merging User Data for Personalization  |     Arxiv'25       |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2501.03276) |
|  *User-LLM* | User-LLM: Efficient LLM Contextualization with User Embeddings |    Arxiv'24        |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2402.13598) |
|  *RECAP*  | RECAP: Retrieval-Enhanced Context-Aware Prefix Encoder for Personalized Dialogue Response Generation |      ACL'23       |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2306.07206) [![Github](https://img.shields.io/github/stars/isi-nlp/RECAP.svg?style=social&label=Github)](https://github.com/isi-nlp/RECAP/tree/main)                     |
| *GSMN*  | Personalized Response Generation via Generative Split Memory Network |     NAACL'21       |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://aclanthology.org/2021.naacl-main.157/) |

## Personalized Adaptation

### One PEFT All Users (One4All)

|  Method   |                                             Paper Title                                              |   Published At    |                                                                                                                                  Link                                                                                                                                  |
| :-------: | :--------------------------------------------------------------------------------------------------: | :---------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  *PEFT-U* |       PEFT-U: Parameter-Efficient Fine-Tuning for User Personalization      |        Arxiv'24           |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2407.18078) [![Github](https://img.shields.io/github/stars/ChrisIsKing/Parameter-Efficient-Personalization.svg?style=social&label=Github)](https://github.com/ChrisIsKing/Parameter-Efficient-Personalization) |
|  *PLoRA* | Personalized LoRA for Human-Centered Text Understanding |     AAAI'24          |                  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2403.06208) [![Github](https://img.shields.io/github/stars/yoyo-yun/PLoRA.svg?style=social&label=Github)](https://github.com/yoyo-yun/PLoRA) |
| *LM-P / CLS-P*  | Personalized Large Language Models |           SENTIRE@   ICDM'24 |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2402.09269) |
|  UserIdentifier | UserIdentifier: Implicit User Representations for Simple and Effective Personalized Sentiment Analysis |   FL4NLP @  ACL'22         |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2110.00135) |
| *Review-LLM*  | Review-LLM: Harnessing Large Language Models for Personalized Review Generation |      Arxiv'24      |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2306.07206)  |
|  *MiLP* | Personalized LLM Response Generation with Parameterized Memory Injection |        Arxiv'24    |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2404.03565) [![Github](https://img.shields.io/github/stars/MatthewKKai/MiLP.svg?style=social&label=Github)](https://github.com/MatthewKKai/MiLP) |
|  *PROPER* | PROPER: A Progressive Learning Framework for Personalized Large Language Models with Group-Level Adaptation  |        Arxiv'25    |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2503.01303) |


### One PEFT Per Users (One4One)

|  Method   |                                             Paper Title                                              |   Published At    |                                                                                                                                  Link                                                                                                                                  |
| :-------: | :--------------------------------------------------------------------------------------------------: | :---------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| *RecLoRA*  | Lifelong Personalized Low-Rank Adaptation of Large Language Models for Recommendation   |      Arxiv'24      | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2408.03533) |
| *iLoRA*  | Customizing Language Models with Instance-wise Lora for Sequential Recommendation |    NeurIPS'24        |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2408.10159) [![Github](https://img.shields.io/github/stars/AkaliKong/iLoRA.svg?style=social&label=Github)](https://github.com/AkaliKong/iLoRA) |
|  *UserAdapter* | UserAdapter: Few-Shot User Learning in Sentiment Analysis |   ACL'21 Findings         |      [![Paper](https://img.shields.io/badge/Paper-blue)](https://aclanthology.org/2021.findings-acl.129.pdf) |
|  *PocketLLM* | PocketLLM: Enabling On-Device Fine-Tuning for Personalized LLMs |    PrivateNLP@   ACL'24        |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2407.01031) |
| *OPPU*  | Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning |   EMNLP'24         | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2402.04401) [![Github](https://img.shields.io/github/stars/TamSiuhin/OPPU.svg?style=social&label=Github)]( https://github.com/TamSiuhin/OPPU) |
|  *Per-Pcs* | Personalized Pieces: Efficient Personalized Large Language Models through Collaborative Efforts |    EMNLP'24        | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2406.10471) [![Github](https://img.shields.io/github/stars/TamSiuhin/Per-Pcs.svg?style=social&label=Github)](https://github.com/TamSiuhin/Per-Pcs) |
|   | Personalized Collaborative Fine-Tuning for LLMs |  COLM'24           |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2404.09753) [![Github](https://img.shields.io/github/stars/epfml/personalized-collaborative-llms.svg?style=social&label=Github)](https://github.com/epfml/personalized-collaborative-llms) |
|  *FDLoRA* | FDLoRA: Personalized Federated Learning of Large Language Model via Dual LoRA Tuning | Arxiv'24   |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2406.07925) |
|  _HYDRA_ | HYDRA: Model Factorization Framework for Black-Box LLM Personalization | NeurIPS'24           |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2406.02888) [![Github](https://img.shields.io/github/stars/night-chen/HYDRA.svg?style=social&label=Github)](https://github.com/night-chen/HYDRA) |



## Personalized Alignment

### Data Construction
|    Name     |                         Paper Title                          |     Published At     |                             Link                             |
| :-----------: | :----------------------------------------------------------: | :------------------: | :----------------------------------------------------------: |
|      | Aligning LLMs with Individual Preferences via Interaction | COLING'25 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2410.03642) [![Github](https://img.shields.io/github/stars/ShujinWu-0814/ALOE.svg?style=social&label=Github)](https://github.com/ShujinWu-0814/ALOE) |
| *PLUM* | On the Way to LLM Personalization: Learning to Remember User Conversations | Arxiv'24 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2411.13405) |
|      | Aligning to Thousands of Preferences via System Message Generalization | NeurIPS'24 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2405.17977) [![Github](https://img.shields.io/github/stars/allenai/FineGrainedRLHF.svg?style=social&label=Github)](https://github.com/kaistAI/Janus) |
|      | Enabling On-Device Large Language Model Personalization with Self-Supervised Data Selection and Synthesis | DAC'24 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2311.12275) |
| *PRISM* | The PRISM Alignment Dataset: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models | NeurIPS'24 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2404.16019) [![Github](https://img.shields.io/github/stars/HannahKirk/prism-alignment.svg?style=social&label=Github)](https://github.com/HannahKirk/prism-alignment) | 
| PersonalLLM | PersonalLLM: Tailoring LLMs to Individual Preferences | Arxiv'24 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2409.20296) [![Github](https://img.shields.io/github/stars/HannahKirk/prism-alignment.svg?style=social&label=Github)](https://huggingface.co/datasets/namkoong-lab/PersonalLLM) |

### Optimization

|    Method     |                         Paper Title                          |     Published At     |                             Link                             |
| :-----------: | :----------------------------------------------------------: | :------------------: | :----------------------------------------------------------: |
| *MORLHF* | Fine-Grained Human Feedback Gives Better Rewards for Language Model Training | NeurIPS'23 |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2306.01693) [![Github](https://img.shields.io/github/stars/allenai/FineGrainedRLHF.svg?style=social&label=Github)](https://github.com/allenai/FineGrainedRLHF)    |
| *MODPO*  | Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization | ACL'24 Findings | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2310.03708) [![Github](https://img.shields.io/github/stars/ZHZisZZ/modpo.svg?style=social&label=Github)](https://github.com/ZHZisZZ/modpo) |
| *Personalized Soups* | Personalized Soups: Personalized Large Language Model Alignment via Post-hoc Parameter Merging | Arxiv'23 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2310.11564) [![Github](https://img.shields.io/github/stars/joeljang/RLPHF.svg?style=social&label=Github)](https://github.com/joeljang/RLPHF) |
| *Reward Soups* | Rewarded soups: towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards | NuerIPS'23 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2306.04488) |
| *MOD* | Decoding-Time Language Model Alignment with Multiple Objectives | NeurIPS'24 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2406.18853) [![Github](https://img.shields.io/github/stars/srzer/MOD.svg?style=social&label=Github)](https://github.com/srzer/MOD) |
| *PAD* | PAD: Personalized Alignment of LLMs at Decoding-Time | Arxiv'24 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2410.04070) |
| *PPT* | Personalized Adaptation via In-Context Preference Learning | Arxiv'24 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2410.14001) |
| *VPL* | Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning | NeurIPS'24 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2408.10075v1) [![Github](https://img.shields.io/github/stars/WEIRDLabUW/vpl_llm.svg?style=social&label=Github)](https://github.com/WEIRDLabUW/vpl_llm) |
| *REST-PG* | Reasoning-Enhanced Self-Training for Long-Form Personalized Text Generation | Arxiv'25 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2501.04167) |



## Analysis

|  Keyword   |                                             Paper Title                                              |   Published At    |                                                                                                                                  Link                                                                                                                                  |
| :-------: | :--------------------------------------------------------------------------------------------------: | :---------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| *Role of User Profile*  | Understanding the Role of User Profile in the Personalization of Large Language Models |  Arxiv'24           | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2406.17803) [![Github](https://img.shields.io/github/stars/Bingo-W/Personalisation-in-LLM.svg?style=social&label=Github)](https://github.com/Bingo-W/Personalisation-in-LLM) |
| *RAG vs. PEFT*  | Comparing Retrieval-Augmentation and Parameter-Efficient Fine-Tuning for Privacy-Preserving Personalization of Large Language Models |  Arxiv'24           | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2409.09510) |
| *Safety-Utility*  | Exploring Safety-Utility Trade-Offs in Personalized Language Models |  Arxiv'24           | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2406.11107) |


## Benchmark
|                      Name                     |                     Paper Title                      | Published At |                                                                                                                                                         Link                                                                                                                                                         |
| :---------------------------------------------------: | :--------------------------------------------------: | :----------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| *LaMP* | LaMP: When Large Language Models Meet Personalization |  ACL'24   | [![Home](https://img.shields.io/badge/Home-red)](https://lamp-benchmark.github.io/) [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2304.11406) [![Github](https://img.shields.io/github/stars/LaMP-Benchmark/LaMP.svg?style=social&label=Github)](https://github.com/LaMP-Benchmark/LaMP) |
| *LongLaMP* | LongLaMP: A Benchmark for Personalized Long-form Text Generation |  Arxiv'24   | [![Home](https://img.shields.io/badge/Home-red)](https://longlamp-benchmark.github.io/) [![Paper](https://img.shields.io/badge/Paper-blue)](https://www.arxiv.org/abs/2407.11016) [![Github](https://img.shields.io/github/stars/LongLaMP-benchmark/LongLaMP-benchmark.github.io.svg?style=social&label=Github)](https://github.com/LongLaMP-benchmark/LongLaMP-benchmark.github.io) |
|  *PerLTQA* |       PerLTQA: A Personal Long-Term Memory Dataset for Memory Classification, Retrieval, and Fusion in Question Answering      |        SIGHAN@  ACL'24           |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2402.16288) [![Github](https://img.shields.io/github/stars/Elvin-Yiming-Du/PerLTQA.svg?style=social&label=Github)](https://github.com/Elvin-Yiming-Du/PerLTQA) |
|  *PEFT-U* |       PEFT-U: Parameter-Efficient Fine-Tuning for User Personalization      |        Arxiv'24           |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2407.18078) [![Github](https://img.shields.io/github/stars/ChrisIsKing/Parameter-Efficient-Personalization.svg?style=social&label=Github)](https://github.com/ChrisIsKing/Parameter-Efficient-Personalization) |
| *PER-CHAT*  | Personalized Response Generation via Generative Split Memory Network |     NAACL'21       |                     [![Paper](https://img.shields.io/badge/Paper-blue)](https://aclanthology.org/2021.naacl-main.157/) [![Github](https://img.shields.io/github/stars/Willyoung2017/PER-CHAT.svg?style=social&label=Github)](https://github.com/Willyoung2017/PER-CHAT)|
| *REGEN*  | Beyond Retrieval: Generating Narratives in Conversational Recommender Systems |    Arxiv'24        | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2410.16780) |
|  *PersonalLLM* |       PersonalLLM: Tailoring LLMs to Individual Preferences      |       Arxiv'24           |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2402.16288) |
| *ALOE* |  Aligning LLMs with Individual Preferences via Interaction | Arxiv'24 |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2410.03642) [![Github](https://img.shields.io/github/stars/ShujinWu-0814/ALOE.svg?style=social&label=Github)]( https://github.com/ShujinWu-0814/ALOE)| 
| *PGraphRAG* |  Personalized Graph-Based Retrieval for Large Language Models | Arxiv'25 |  [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2501.02157) [![Github](https://img.shields.io/github/stars/PGraphRAG-benchmark/PGraphRAG.svg?style=social&label=Github)](https://github.com/PGraphRAG-benchmark/PGraphRAG)| 
| *PrefEval*|Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs| ICLR'25 | [![Home](https://img.shields.io/badge/Home-red)](https://prefeval.github.io/) [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/pdf/2502.09597) [![Github](https://img.shields.io/github/stars/amazon-science/PrefEval.svg?style=social&label=Github)](https://github.com/amazon-science/PrefEval)|
| *LongMemEval*| LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory | ICLR'25 | [![Home](https://img.shields.io/badge/Home-red)](https://xiaowu0162.github.io/long-mem-eval/) [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/abs/2410.10813) [![Github](https://img.shields.io/github/stars/xiaowu0162/LongMemEval.svg?style=social&label=Github)](https://github.com/xiaowu0162/LongMemEval)|
| *IMPLEXCONV*| Toward Multi-Session Personalized Conversation: A Large-Scale Dataset and Hierarchical Tree Framework for Implicit Reasoning | Arxiv'25 | [![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/pdf/2503.07018) |

# Citation

```bibtex
@article{liu2025survey,
  title={A Survey of Personalized Large Language Models: Progress and Future Directions},
  author={Liu, Jiahong and Qiu, Zexuan and Li, Zhongyang and Dai, Quanyu and Zhu, Jieming and Hu, Minda and Yang, Menglin and King, Irwin},
  journal={arXiv preprint arXiv:2502.11528},
  year={2025}
}
```


