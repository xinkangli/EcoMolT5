








# EcoMolT5: Ecological-Prompted Multimodal Molecular T5

**EcoMolT5** is an instruction-driven, structure-enhanced multimodal molecular prediction framework that integrates **molecular graphs**, **fingerprints**, and **natural-language task instructions**.  
The model introduces a **fingerprint-guided contrastive learning module** and a **Weak Ecological Prompt (WEP)** mechanism to enhance structure–semantic alignment and improve interpretability in environmental molecular property prediction.

---

## 

This repository is an **independent extension** based on the open-source [**GIMLET**](https://github.com/zhao-ht/GIMLET) framework (MIT License).  
The original GIMLET model provides an instruction-based molecule–language foundation architecture for property prediction.  
EcoMolT5 expands upon this foundation by introducing the following key innovations:

1. **Fingerprint-Guided Contrastive Learning**  
   Aligns graph representations with global molecular semantics for improved discriminability.

2. **Integration of the QM9 Dataset**  
   Incorporates quantum-mechanical properties (HOMO/LUMO) to enhance multi-scale physical and biochemical understanding.

3. **Weak Ecological Prompt (WEP)**  
   Embeds ecotoxicological semantics through soft linguistic cues (e.g., *may cause bioaccumulation*) to improve interpretability and zero/few-shot transfer.

4. **Structure-Aware Attention Bias**  
   Injects topological and bond-type priors into Transformer fusion layers for fine-grained structural modeling.

---

## 📘 Citation



```bibtex
@article{evans2025ecomolt5,
  title={EcoMolT5: Ecological-Prompted Multimodal Molecular T5},
  author={Evans, Eric},
  year={2025},
  note={An instruction-driven multimodal framework integrating GIMLET and QM9 for environmental molecular property prediction}
}


@inproceedings{zhao2023gimlet,
  title={GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning},
  author={Zhao, Hengtong and others},
  booktitle={NeurIPS},
  year={2023},
  license={MIT}
}

