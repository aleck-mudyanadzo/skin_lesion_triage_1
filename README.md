# Skin Lesion Triage

**A web tool that looks at a phone photo of a skin spot and says whether it should be seen by a specialist soon.**

Final year Computer Science project, Midlands State University (Zimbabwe)
Authors: Aleck Mudyanadzo and Primrose S. Ncube

> **Important:** This tool does not diagnose anything. It is a research prototype. It cannot replace a doctor, and nobody should make health decisions based on it. If you are worried about a skin spot, see a health professional.

---

## 1. The problem, in plain words

Skin problems range from harmless rashes to serious cancers like melanoma. When skin cancer is found early, it is far easier to treat. Found late, it can be deadly.

In Zimbabwe there are very few skin specialists (dermatologists), and most of them work in a few big centres. People in rural and peri urban areas may wait months or travel far just to be looked at. Many nurses and clinic staff who see these patients first have no quick way to tell "this can wait" from "this needs a specialist now".

Most people do have a smartphone, though. So the question behind this project is:

**Can a phone photo help a nurse or student sort skin spots into "probably fine" and "please get this checked soon"?**

That sorting job is called **triage**. It is the same idea as in a hospital emergency room, where staff decide who needs attention first. This project does not name the disease. It only helps decide who should be looked at sooner.

## 2. What the system does

1. A user opens the web page and uploads a photo of a skin spot.
2. The system cleans up the photo (more on that below).
3. The system gives a **risk score** and a simple label: either *low concern* or *suspicious, refer for review*.
4. It also shows a **heatmap** on the photo, highlighting the parts of the image that pushed the system towards its answer. This lets a human check whether the computer looked at the actual spot or at something silly like a shadow.

That is the whole flow: photo in, risk score and heatmap out.

## 3. What kind of AI is this?

This is the part people often find confusing, so here it is step by step.

**Artificial Intelligence (AI)** is a broad term for computers doing tasks that normally need human judgement.

**Machine Learning** is one way of doing AI. Instead of writing rules by hand ("if the spot is darker than X, then..."), we show the computer many examples and let it work out the patterns itself.

**Deep Learning** is a type of machine learning built from many layers of tiny calculating units, loosely inspired by how brain cells connect. Deep learning is very good at understanding images.

**Convolutional Neural Network (CNN)** is the specific kind of deep learning used here. Think of it as a very fast, very patient reader of pictures. Its early layers notice simple things like edges and colours. Later layers combine those into shapes and textures. The final layer combines everything into an answer.

**Transfer Learning** is the trick that makes this project practical. Training a picture reading network from nothing needs millions of images and expensive computers. Instead, we start with networks that have already learned to see on huge general photo collections, and then teach them our specific job with a smaller set of skin images. It is like hiring someone who already knows how to read and then training them in one specialised subject, instead of teaching a baby to read first.

We tested two such starting networks:

| Network | Plain description |
| --- | --- |
| **ResNet50** | A large, powerful network. Usually more accurate but heavier to run. |
| **MobileNetV2** | A small, light network designed for phones and cheap servers. Faster, slightly less powerful. |

The type of task is **binary classification**: every photo is placed into one of two groups, *benign* (not cancer) or *malignant suspect* (could be cancer).

## 4. Where the learning data comes from

The system learned from **HAM10000**, a well known public collection of about 10,000 close up skin images, each with a confirmed diagnosis from experts. It is widely used by researchers.

Two honest things to know about this data:

* It has far more harmless spots than dangerous ones, roughly four to one. If we ignored this, the system could get a high score by simply guessing "harmless" every time. We corrected for it during training.
* It was mostly collected with special medical cameras, and it under represents darker skin tones. That is a known problem in this field, and it is one reason this tool is not ready for real patients.

## 5. Why phone photos are the hard part

Photos in a clinic corridor or a village are not like photos in a research lab. They can be blurry, dim, grainy, or have hair across the spot. Many AI systems that score well in the lab do badly on such images. This mismatch is called **domain shift**, and dealing with it is the main focus of the project.

To cope, every photo goes through a cleaning stage (built with a tool called OpenCV) before the AI sees it:

* **Hair removal:** digitally removes hairs that cross the spot.
* **Noise reduction:** smooths out grain from cheap cameras.
* **Contrast correction:** evens out lighting so dark or washed out photos are easier to read.
* **Blur check:** if the photo is too blurry to trust, the system rejects it and asks for a new one instead of guessing.

We also made a test set of deliberately blurred and badly lit images to see how the system copes with poor quality input.

## 6. Why the heatmap matters

A common worry about AI is that it is a "black box": it gives an answer and nobody knows why. To make the system more open, we use a method called **Grad-CAM**. It paints a colour overlay on the photo, where warmer colours show the areas that influenced the answer most. A nurse can quickly see whether the AI focused on the lesion itself. If the heat is on the background, the answer should not be trusted.

## 7. How well does it work so far?

The main safety goal is **recall**, which answers: *"Of all the spots that really were dangerous, how many did the system catch?"* In a triage tool, missing a dangerous spot is worse than raising a false alarm, so we set a target of catching at least 85 percent.

Results so far, with the honest picture:

* **MobileNetV2:** caught about **85.3 percent** of dangerous cases on validation data, and about **80.9 percent** on the separate test data. So it met the target during tuning but fell a little short on unseen data. We are reporting both numbers because the test figure is the more realistic one.
* **ResNet50:** trained, but the final tuned numbers are still being recovered and compared. This section will be updated when they are confirmed.

The system also has an adjustable **decision threshold**, which is like a sensitivity dial. Turning it towards caution catches more dangerous cases but also flags more harmless ones. We tune it on validation data to aim for the recall target.

## 8. What is inside (the tech stack)

For readers who do want the technical side:

| Part | Tool |
| --- | --- |
| Programming language | Python |
| Web server | Flask |
| Page design | Bootstrap 5 |
| AI training and use | TensorFlow / Keras |
| Image cleaning | OpenCV |
| Measuring performance | scikit learn |
| Storage | SQLite (small built in database) |
| Training computer | Google Colab (free cloud GPU) |

The web app has these main pages and endpoints:

* `/` : the upload page and result
* `/predict` : receives a photo and returns the result
* `/health` : reports whether the system and model are loaded
* `/stats` : shows anonymous usage and survey summaries
* `/survey` : a short usability questionnaire

**Privacy:** the database does not store names or personal details. Photos are stored only as a scrambled fingerprint (a hash), along with the model used, the score and the label. This is only used to evaluate the system.

## 9. How usefulness is being tested

Performance numbers alone do not show that a tool is usable. We are asking 15 to 20 final year medical and nursing students to try it and complete the **System Usability Scale (SUS)**, a standard 10 question survey that produces a score out of 100. The survey form is built into the app.

## 10. Limitations (please read)

* It is a **triage aid, not a diagnosis tool**.
* It only separates benign from suspicious. It does not name conditions.
* It was trained on a public dataset that lacks good representation of darker skin tones, so it may perform worse on them. Testing on more diverse, real world images is the most important next step.
* Very blurry photos or photos with heavy hair cover are rejected.
* Speed on a free cloud server has not yet been measured.
* No clinical trial has been done. It has not been approved for medical use.

## 11. Running it yourself

You need Python 3.10 or newer.

```bash
git clone https://github.com/aleck-mudyanadzo/skin_lesion_triage_1.git
cd skin_lesion_triage_1
pip install -r requirements.txt
python run.py
```

Then open the address shown in the terminal (usually `http://127.0.0.1:5000`).

The trained model files are large and are not stored in this repository. Training was done on Google Colab using the training script, and the resulting `.keras` files need to be placed in the `models` folder before predictions will work.

*(Adjust the folder names and start command above if your project layout differs.)*

## 12. What comes next

* Confirm and compare the final ResNet50 and MobileNetV2 results
* Complete the usability study with medical and nursing students
* Deploy the app online
* Test on more varied, real phone photos, especially across different skin tones

## 13. Credits and references

* Tschandl, Rosendahl and Kittler, "The HAM10000 dataset," *Scientific Data*, 2018 (the training images)
* Esteva et al., "Dermatologist level classification of skin cancer with deep neural networks," *Nature*, 2017 (showed AI can match specialists on certain tasks)

Supervised as part of the Bachelor of Science Honours Degree in Computer Science, Department of Computer Science, Faculty of Science and Technology, Midlands State University.

## Disclaimer

This software is for academic research and education only. It is provided as is, with no medical warranty. It must not be used to diagnose, treat or make decisions about any real patient.
