# Tamil Text-to-Image Generation with Stable Diffusion

This repo has supporting materials for my Masters dissertation. 

## Overview
This repository contains Jupyter notebooks, scripts, and models for research focused on improving text-to-image generation for the Tamil language using Stable Diffusion models. The contents include fine-tuning scripts, evaluation tools, and other supporting files necessary to replicate the experiments and generate the results discussed in the study.

## Repository Structure

### 1. Notebooks
- labse_embeddings.ipynb: Generates Multilingual CLIP (M-CLIP) embeddings for Tamil captions
- formatting_datasets.ipynb: Prepares Hugging Face datasets for training and testing, including images and captions.
- fine_tune_sd2_base.ipynb: Fine-tunes the Stable Diffusion 2 Base model using the prepared dataset.
- fine_tune_sd_v1_4.ipynb: Fine-tunes the Stable Diffusion v1.4 model using the prepared dataset.
- fine_tune_sd_v1_5.ipynb: Fine-tunes the Stable Diffusion v1.5 model using the prepared dataset.
- evaluation_v1_4.ipynb: Performs inference on the test dataset using the fine-tuned Stable Diffusion v1.4 model.
- plots.ipynb: Generates plots and calculates FID scores for evaluating the models.

### 2. Pipeline Changes
To fine-tune the models and perform inference using the Stable Diffusion pipelines, certain files from the Diffusers library need to be replaced with custom versions:

- **Fine-Tuning:**
  1. Clone the Diffusers library from [Diffusers GitHub Repository](https://github.com/huggingface/diffusers). (code is in fine_tune_*.ipynb notebooks)
  2. Replace the `diffusers/examples/text_to_image/train_text_to_image.py` script with the modified script provided in this repository.

- **Inference Pipeline:**
  1. Replace the following files with the modified versions provided:
     - `diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`
     - `model_index.json` (generated after running the fine-tune script).

### 3. Dataset
The dataset used for training and evaluation can be accessed from the following link:
- [MMDravi Dataset on Zenodo](https://zenodo.org/records/4765597#.YKEB-yYo_0M)

If you use this dataset, please cite the following paper:

```
@inproceedings{chakravarthi-etal-2019-multilingual,
    title = "Multilingual Multimodal Machine Translation for {D}ravidian Languages utilizing Phonetic Transcription",
    author = "Chakravarthi, Bharathi Raja  and
      Priyadharshini, Ruba  and
      Stearns, Bernardo  and
      Jayapal, Arun  and
      S, Sridevy  and
      Arcan, Mihael  and
      Zarrouk, Manel  and
      McCrae, John P",
    booktitle = "Proceedings of the 2nd Workshop on Technologies for MT of Low Resource Languages",
    month = aug,
    year = "2019",
    address = "Dublin, Ireland",
    publisher = "European Association for Machine Translation",
    url = "https://www.aclweb.org/anthology/W19-6809",
    pages = "56--63",
}
```

### 4. Models
The fine-tuned model checkpoints and accompanying JSON files were saved to my google drive folder and can be downloaded from the following link
- [Download Fine-Tuned Models](https://drive.google.com/drive/folders/1pCiFFQqwdmvbJTSxisVXTyLBLsnZalF3?usp=sharing)

## Instructions
1. **Install Dependencies:**
   - Install the required libraries as mentioned in the notebooks and scripts.
   - Ensure you have the latest version of the Diffusers library from Hugging Face.

2. **Generate Embeddings:**
   - Run `labse_embeddings.ipynb` to generate M-CLIP embeddings.

3. **Prepare Dataset:**
   - Use `formatting_datasets.ipynb` to prepare the training and testing datasets.

4. **Fine-Tune Models:**
   - Fine-tune the Stable Diffusion models using the respective notebooks: `fine_tune_sd2_base.ipynb`, `fine_tune_sd_v1_4.ipynb`, `fine_tune_sd_v1_5.ipynb`.

5. **Evaluate Models:**
   - Perform inference using `evaluation_v1_4.ipynb`.
   - Generate plots and calculate FID scores using `plots.ipynb`.

6. **Pipeline Modifications:**
   - Replace the necessary files in the Diffusers library as described in the "Pipeline Changes" section before running fine-tuning or inference scripts.

