# MIAS (Model Intelligent Analysis System)

This repository contains the configuration, datasets, and evaluation scripts for the MIAS project. The model is fine-tuned using LLaMA-Factory based on the DeepSeek-LLM-7b-base architecture.

## 📂 Directory Structure

* **`models/`**: Contains the model configuration files (`adapter_config.json`, `tokenizer`, etc.).
* **`dataset/`**: Contains the training datasets which have been augmented using the **MGA (Multi-Grain Augmentation)** method.
* **`test/`**: Contains Q&A sets used for model evaluation and testing.

## 🛠️ Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/your-username/mias.git](https://github.com/your-username/mias.git)
    cd mias
    ```

2.  Install the required dependencies, specifically **LLaMA-Factory**:
    ```bash
    pip install llamafactory
    ```

## ⬇️ Model Weights Download

Due to GitHub's file size limitations, the trained LoRA adapter weights (`adapter_model.safetensors`) are hosted externally. You must download them and place them in the `models/` directory to run the model.

| Model Name | Link | Password |
| :--- | :--- | :--- |
| **MIAS-adapter** | [Baidu Netdisk (Pan.baidu)](https://pan.baidu.com/s/1bgdpSSkflTG3O-gyrDCqKg?pwd=2026) | **2026** |

**Setup Instructions:**
1.  Download the file from the link above.
2.  Unzip/Extract the contents.
3.  Move the `adapter_model.safetensors` file into the `models/` folder in this repository, ensuring it sits alongside `adapter_config.json`.

## 📊 Datasets

The training data is located in the `dataset/` folder.
* **Augmentation**: The original data has been expanded using the **MGA (Multi-Grain Augmentation)** strategy to enhance model robustness and generalization.

## wm Evaluation

The `test/` folder contains specific Question & Answer pairs designed to test the model's domain capability.