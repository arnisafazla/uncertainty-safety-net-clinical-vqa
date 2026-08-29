# Uncertainty Is Not a Safety Net for Clinical VQA, but Can It Anticipate Model Failure?

This repository contains the code, data processing scripts, and experiments accompanying the paper:

**Uncertainty Is Not a Safety Net for Clinical VQA, but Can It Anticipate Model Failure?**  
Arnisa Fazla*, Alberto Testoni∗, Ameen Abu-Hanna, Barbara Plank, Iacer Calixto     
[arXiv:2606.16583](https://arxiv.org/abs/2606.16583) - Accepted to EMNLP 2026 (Findings).

<!-- <p align="center"> <img src="./images/fig_1.png" width="300"/> </p>

<p align="center"> <em>Overview of our evaluation of uncertainty estimation in clinical VQA. While uncertainty does not reliably track model correctness under stress, it can anticipate predictions that are vulnerable to model failure.</em> </p> -->

---
## Repository Status

⚠️ **Work in Progress**  
This repository is being updated gradually. Some scripts and documentation may be incomplete or subject to change. Please check back regularly for updates.

---

## Overview

Reliable uncertainty estimation (UE) is critical for the safe deployment of clinical vision-language models (VLMs), where uncertain predictions should be identified and escalated to a clinician.
In this work, we systematically study whether current UE methods provide a reliable signal of prediction trustworthiness in clinical visual question answering (VQA). Our contributions are:

1. **Benchmarking uncertainty estimation.**
We systematically benchmark 8 uncertainty estimation methods across 12 clinical VLMs on clinical VQA, evaluating how reliably uncertainty identifies correct and incorrect predictions.
2. **Characterizing the dependence of UE quality on model accuracy.**
We find that UE quality is not an intrinsic property of the uncertainty estimation method. Instead, it closely tracks model accuracy, degrading precisely where model performance is weakest, and where reliable uncertainty estimates are most needed.
3. **Stress-testing uncertainty with NOTA perturbations.**
We introduce NOTA (None-of-the-Above) perturbations, which hide the correct answer among the multiple-choice options. Under this stress test, model accuracy collapses while uncertainty changes only marginally, revealing systematic miscalibration and a failure of current UE methods to reflect prediction fragility.
4. **Using uncertainty as a diagnostic signal of model fragility.**
Despite this limitation, we find that uncertainty on the unperturbed input reliably anticipates which predictions will collapse under NOTA perturbations. This suggests that UE contains diagnostic information about model fragility, motivating perturbation-based evaluation as a promising direction toward safer clinical deployment.

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@misc{fazla2026uncertaintysafetynetclinical,
      title={Uncertainty Is Not a Safety Net for Clinical VQA, but Can It Anticipate Model Failure?}, 
      author={Arnisa Fazla and Alberto Testoni and Ameen Abu-Hanna and Barbara Plank and Iacer Calixto},
      year={2026},
      eprint={2606.16583},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.16583}, 
}