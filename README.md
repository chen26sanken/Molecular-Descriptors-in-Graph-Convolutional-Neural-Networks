# Efficiently Utilizing Molecular Descriptors in Graph Convolutional Neural Networks for Small Molecule Drug Discovery

This repository contains the code accompanying the manuscript:

**Efficiently Utilizing Molecular Descriptors in Graph Convolutional Neural Networks for Small Molecule Drug Discovery**  
Qingwen Chen\*¹, Kaito Fukui\*², Hiroaki Santo², Fumio Okura², Takeshi Yamada³, Kazuhiko Nakatani¹, Yasuyuki Matsushita²  
\*Equal contribution  

¹ SANKEN, The University of Osaka, Mihogaoka 8-1, Ibaraki, Osaka, 567-0047, JAPAN  
² Graduate School of Information Science and Technology, The University of Osaka, Yamadaoka 1-5, Suita, 565-0871, JAPAN  
³ Nucleic Acid Drug Discovery Center, Tokyo University of Science, 5-45 Yushima, Bunkyo City, 113-8510, JAPAN

---

## Abstract
Small molecule drug discovery remains a key therapeutic strategy for treating various diseases. This study explores and compares three methods of integrating molecular descriptors into Graph Convolutional Neural Networks (GCNNs) to predict small molecules targeting CAG repeat DNA. Using graph-based molecular representations, we assess how descriptor incorporation affects predictive performance. Our results show that combining graph structural information with molecular descriptors enhances accuracy, though different integration strategies yield varying effects on learning descriptor features. This study provides a reference for optimizing molecular descriptor utilization in graph-based models, improving small molecule activity prediction, and advancing in silico drug discovery research.

**Keywords**: small molecule, drug discovery, GCNN, molecular descriptors

---

## Table of Contents
- [Overview](#overview)
- [Scripts](#scripts)
- [Requirements](#requirements)


---

## Overview
This project implements the full pipeline to preprocess DNA sequence data, generate labels, and train a Graph Convolutional Neural Network (GCNN) model that leverages molecular descriptors for small molecule prediction tasks.

## Scripts
- **1_data_separation_kfukui_1.1.py**  
  Separates raw molecular structure data into structured input files.

- **2_input_processing_kfukui_1.1.py**  
  Processes the separated inputs and converts them into graph representations.

- **3_label_generation_kfukui_1.0.py**  
  Put classification labels aligned with input molecular graph data.

- **4_GCN_DNA_kfukui_1.1.py**  
  Defines, trains, and evaluates the GCN model incorporating molecular descriptors.

---

## Requirements
- Python 3.7+
- PyTorch
- NumPy
- scikit-learn
- NetworkX

Install the necessary libraries via:
```bash
pip install torch numpy scikit-learn networkx
```

---

## Citation
If you find this code useful in your research, please cite the following paper:

> Qingwen Chen, Kaito Fukui, Hiroaki Santo, Fumio Okura, Takeshi Yamada, Kazuhiko Nakatani, Yasuyuki Matsushita.  
> *Efficiently Utilizing Molecular Descriptors in Graph Convolutional Neural Networks for Small Molecule Drug Discovery.*

---
