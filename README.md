
# Flower Image Classifier - Deep Learning with PyTorch

This project trains an image classification model using transfer learning to recognize different species of flowers from the Oxford 102 Flower dataset. It's part of the Udacity Data Scientist Nanodegree program and demonstrates how to preprocess image data, build and train a model using PyTorch, perform inference, and visualize predictions.

---

## Project Overview

- **Model**: VGG16 (pre-trained on ImageNet)
- **Dataset**: Oxford 102 Flower Categories
- **Classes**: 102 flower species
- **Framework**: PyTorch
- **Accuracy**: ~70%+ on the test set
- **Device**: Compatible with CPU & GPU (T4, etc.)

---

## Project Structure

```
.
├── checkpoint.pth                # Saved model checkpoint
├── Flower Classifier.ipynb       # Complete Colab-ready Jupyter Notebook
├── cat_to_name.json              # Class-to-name mapping for flower labels
├── train.py / predict.py         # (Optional) Standalone scripts
├── README.md                     # Project documentation (you are here)
```

---

## Features

- Transfer learning with VGG16
- Data augmentation for training set
- Model evaluation and testing
- Top-5 predictions with probabilities
- Sanity checking with Matplotlib
- Inference-ready pipeline

---

## How to Use

### Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/Flower%20Classifier.ipynb)

> Replace `YOUR_USERNAME/YOUR_REPO` with your actual GitHub path

---

### Requirements

If running locally:

```bash
pip install torch torchvision matplotlib seaborn numpy pandas pillow
```

---

## Example Output

![Prediction Example](insert-your-screenshot-url-here)

- Input: An image of a flower
- Output: Top 5 predicted flower names with probabilities

---

## Acknowledgements

- Oxford 102 Flower Dataset
- [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
- Pre-trained models from `torchvision.models`

---

## Next Steps

- Convert notebook into deployable web app
- Extend with more flower datasets or fine-tune for other objects
- Deploy with Flask or Streamlit (bonus)

---

## License

This project is for educational purposes and is distributed under the MIT License.
