# 🐶 Dog Breed Classification & Generation App

An end-to-end computer vision project for dog image understanding, combining **CNNs, ResNet, VAEs, GANs** (for now, intended expansion on ML models), and an interactive **Streamlit UI**.

Users can:
- Upload an image and get **top-3 dog breed predictions** 
- Check **“Is this a dog?”** with confidence
- Visualize model confidence

---

## 📌 Features

### 🔍 Classification
- Custom CNN (VGG-style)
- ResNet-18 (transfer learning)
- Top-3 predictions with confidence bars

### 🐕 Dog Detection
- Binary dog vs non-dog check using classifier confidence

### 🧠 Generative Models
- Convolutional VAE (reconstruction + sampling)
- Convolutional GAN (image generation)

### 🖥️ Web App
- Streamlit-based UI
- Upload images & run inference in-browser

---

## 🗂️ Project Structure
```
./
  main.py
  app.py
  checkpoints/
  utils/
  models/
  samples/
  inference/
  eval/
  data/
    dogs/
      test/
      train/
      val/
    raw_dogs/
  notebooks/
  generation/
  class_names.txt
  requirements.txt
  Dockerfile
```



---

## 🧪 Training

```bash
python main.py
````

Trains and saves:

* CNN
* ResNet-18
* VAE
* GAN

All checkpoints are stored in `./checkpoints`.

---

## 📊 Evaluation

Run evaluation scripts manually:

```bash
python eval/run_eval.py
```

Analysis and plots live in:

* `analysis.ipynb`

---

## 🚀 Running the App (Docker) (DEV)

### Build Image

```bash
docker build -t dog-breed-app .
```

### Run Container

```bash
docker run -p 8501:8501 dog-breed-app
```

Then open:

```
http://localhost:8501
```

---

## 🧑‍💻 Running Locally (Dev Mode)

### 1️⃣ Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit

```bash
streamlit run app.py
```

---

## 📦 Requirements

* Python 3.10+
* PyTorch (CPU)
* Streamlit
* torchvision
* numpy, matplotlib, sklearn

(Handled automatically in Docker)

---

## 🧠 Future Work

* VAE and GAN integration (training and eval is complete)
* Grad-CAM visualizations
* Conditional GAN by breed
* Model ensemble predictions
* Improved App and UI
* Deployment to cloud 

---

## ♾️ Limitations

* Computational power
