# Fine-Tune a Hugging Face Model on AWS: Workshop Walkthrough

**AI & ML Club × AWS Club**

In this workshop, students take a pretrained model from Hugging Face (`distilbert-base-uncased`) and fine-tune it to detect the emotion in a sentence (sadness, joy, love, anger, fear, surprise). They use the `dair-ai/emotion` dataset and do everything in **Amazon SageMaker AI**. They train it twice:

1. **In a notebook**, so they can see each step.
2. **As a SageMaker training job**, the "real" cloud way: AWS starts a machine, runs the code, saves the model to S3 and shuts the machine down.

Everything runs on **CPU instances that are covered by the SageMaker free tier**. No GPU quota is needed.

> **How to read this doc**
> - Numbered steps are what students click or type.
> - `> Say:` blocks are presenter talking points.
> - `Expected:` lines tell you what students should see. If they don't see it, check the **Troubleshooting** section at the bottom.

---

## Timing at a glance

| Part | What happens | Time |
|---|---|---|
| 0 | Sign in, pick the region | 10 min |
| 1 | Open SageMaker Studio and start a notebook | 10 min |
| 2 | Fine-tune the model in the notebook | 35–40 min |
| 3 | Run the same training as a SageMaker training job | 20–25 min |
| 4 | Clean up (**do not skip**) | 5 min |

(The concept slides go before Part 0.)

---

## BEFORE THE WORKSHOP: presenter checklist

Do all of this **at least 3 days before** the event.

- [ ] **Decide how students get AWS accounts.** Ask the AWS club whether they can get temporary event accounts from AWS (e.g. AWS Workshop Studio or AWS Academy). If they can, students don't need credit cards and you can skip most of the setup below. If not, students must create their own AWS accounts **before the event**, because verification can take time.
- [ ] **Pick ONE region for everyone** and put it on a slide. This guide uses **US East (N. Virginia) `us-east-1`**. Mixed regions are the #1 cause of "I can't find my stuff."
- [ ] **If students use personal accounts:** send them the "Student pre-work" section below and ask them to finish it before they arrive. Setting up a SageMaker domain takes 5–10 minutes, and you don't want 40 people doing it at the same time on venue Wi-Fi.
- [ ] **Do a full dry run** of Parts 1–4 in a fresh account. Time it. Library versions change, so if a cell fails, fix it here and update this doc.
- [ ] **Prepare a backup:** the Part 2 code also runs in Google Colab (it's plain Python). Have a Colab link ready in case some students can't sign in to AWS.
- [ ] **Have 1–2 TAs** (ideally from the AWS club) who did the dry run and can walk around.
- [ ] **Cost note:** For new AWS accounts, the SageMaker AI free tier (first 2 months) covers 250 hrs/month of `ml.t3.medium` notebooks and 50 hrs/month of `ml.m5.xlarge` training. This workshop uses about 1.5 notebook hours and about 15 training minutes per student. It's still a good idea for students to set up a zero-spend budget alert (see pre-work).

### Student pre-work (send this out ahead of time)

1. Create an AWS account at **aws.amazon.com** → **Create an AWS Account**.
2. Sign in to the **AWS Console**.
3. In the **top-right corner**, click the region name and select **US East (N. Virginia)**.
4. **Set a budget alert (recommended):** In the top search bar, type **Billing** → open **Billing and Cost Management** → click **Budgets** (left menu) → **Create budget** → choose **Use a template** → **Zero spend budget** → enter your email → **Create budget**.
5. **Set up SageMaker:**
   1. In the top search bar, type **SageMaker AI** and click it. (Be careful to pick "Amazon SageMaker **AI**", not plain "Amazon SageMaker".)
   2. In the left menu, click **Domains** (sometimes under **Admin configurations**).
   3. Click **Create domain**.
   4. Choose **Set up for single user (Quick setup)** → click **Set up**.
   5. Wait until the status says **Ready** (5–10 minutes). You can close the tab once you see it.

---

## PART 0: Sign in (10 min)

> **Say:** "Everyone sign in to the AWS Console. Before you do anything else, look at the top-right corner. It needs to say **N. Virginia**. If it doesn't, click it and change it."

1. Go to **console.aws.amazon.com** and sign in (or use the event account link from the AWS club).
2. **Top-right corner:** confirm the region is **US East (N. Virginia)**. If not, click it and select it.
3. In the top search bar, type **SageMaker AI** and open it.

**Expected:** the Amazon SageMaker AI dashboard.

---

## PART 1: Open SageMaker Studio and start a notebook (10 min)

> **Say:** "SageMaker Studio is AWS's browser-based workspace for ML. Inside it we'll launch JupyterLab. It's the same Jupyter you might already know, except it's running on an AWS computer instead of your laptop."

1. In the SageMaker AI left menu, click **Studio**.
2. Under the user profile dropdown, keep the default user (e.g. `default-…`) and click **Open Studio**.
   - A new browser tab opens with SageMaker Studio. If you get a welcome tour pop-up, click **Skip Tour**.
3. In Studio's left side panel, under **Applications**, click **JupyterLab**.
4. Click **+ Create JupyterLab space** (top right).
5. **Name:** `emotion-workshop` → click **Create space**.
6. On the space settings screen, check:
   - **Instance:** `ml.t3.medium` (this is the free-tier one, **do not change it**)
   - **Image:** the default **SageMaker Distribution** (latest version is fine)
7. Click **Run space**. Wait 1–3 minutes until the status shows **Running**.
8. Click **Open JupyterLab**. Another tab opens.
9. In JupyterLab, click **File → New → Notebook**.
10. If asked to select a kernel, choose **Python 3 (ipykernel)** → **Select**.
11. Rename the notebook: right-click **Untitled.ipynb** in the left file browser → **Rename** → `emotion-workshop.ipynb`.

**Expected:** an empty notebook with one code cell.

> **How to run a cell:** click inside the cell, then press **Shift + Enter**. While it's running you'll see `[*]` on the left. When it's done, the `*` turns into a number.
>
> **Rule for the whole workshop:** run cells **in order, top to bottom**, and wait for each one to finish before running the next.

---

## PART 2: Fine-tune the model in the notebook (35–40 min)

For each step, **copy the code block into a new cell** and run it. To add a new cell, click the **+** button in the notebook toolbar, or press `Esc` then `B`.

### Step 2.1: Install libraries

```python
%pip install -q "transformers>=4.44,<5" "datasets>=2.20" "accelerate>=0.26" "sagemaker>=2.200,<3"
```

**Expected:** it finishes after about 1 minute. Red text about dependency warnings is OK. A line with `ERROR:` in it is not.

**IMPORTANT, restart the kernel now:** click **Kernel → Restart Kernel…** → **Restart**. This loads the freshly installed libraries.

> **Say:** "`transformers` is Hugging Face's library for loading and training models. `datasets` is their library for loading datasets. `sagemaker` is the AWS library we'll use in Part 3."

### Step 2.2: Check that everything loaded

```python
import torch, transformers, datasets
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("torch:", torch.__version__)
print("GPU available?", torch.cuda.is_available())
```

**Expected:** three version numbers and `GPU available? False`. False is correct, because we're on a cheap CPU machine on purpose.

### Step 2.3: Load the dataset from Hugging Face

```python
from datasets import load_dataset

ds = load_dataset("dair-ai/emotion")
print(ds)

labels = ds["train"].features["label"].names
print("Labels:", labels)

print("\nA few examples:")
for i in range(5):
    row = ds["train"][i]
    print(f"  {labels[row['label']]:>9} | {row['text']}")
```

**Expected:** a summary showing `train` (16000 rows), `validation` (2000) and `test` (2000), the 6 labels, and 5 example tweets with their emotions.

> **Say:** "We just downloaded a dataset from the Hugging Face Hub with one line. Every row is a piece of text and a label. That's what *supervised learning* means: we show the model examples where we already know the right answer."
>
> Optional: show huggingface.co/datasets/dair-ai/emotion on the projector.

### Step 2.4: Take a small slice (so training is fast)

```python
train_ds = ds["train"].shuffle(seed=42).select(range(2000))
test_ds  = ds["test"].shuffle(seed=42).select(range(500))

print("Training examples:", len(train_ds))
print("Test examples:", len(test_ds))
```

**Expected:** `Training examples: 2000` and `Test examples: 500`.

> **Say:** "We're only using 2,000 of the 16,000 examples so training finishes in minutes on a CPU. The **test** set is data the model never sees during training. That's how we check whether it actually learned or just memorized."

### Step 2.5: Tokenization (how models "read")

```python
from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

example = "i feel so happy today"
encoded = tokenizer(example)
print("Token IDs:", encoded["input_ids"])
print("Tokens:   ", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
```

**Expected:** a list of numbers, then the matching tokens: `['[CLS]', 'i', 'feel', 'so', 'happy', 'today', '[SEP]']`.

> **Say:** "Models can't read words, only numbers. The tokenizer splits text into pieces and turns each piece into an ID. `[CLS]` and `[SEP]` are special start and end markers. Every model on Hugging Face comes with its own matching tokenizer."
>
> Fun extra: have someone try a weird word like `"unbelievably"` to show it splitting into sub-words.

### Step 2.6: Tokenize the whole dataset

```python
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

train_tok = train_ds.map(tokenize, batched=True)
test_tok  = test_ds.map(tokenize, batched=True)
print("Done tokenizing!")
```

**Expected:** two progress bars, then `Done tokenizing!`.

### Step 2.7: Load the pretrained model

```python
from transformers import AutoModelForSequenceClassification

id2label = {i: name for i, name in enumerate(labels)}
label2id = {name: i for i, name in enumerate(labels)}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)
print(f"Model loaded with {model.num_parameters():,} parameters")
```

**Expected:** `Model loaded with ~67,000,000 parameters`. You'll **also** see a warning like *"Some weights of DistilBertForSequenceClassification were not initialized… You should probably TRAIN this model…"* **That warning is expected.**

> **Say:** "DistilBERT already understands English. Someone else trained it on a huge amount of text. But we just bolted a brand-new 'emotion classifier' layer on top, and that layer is random. That's what the warning is telling us. A model is really just ~67 million numbers called **weights**. Training means adjusting those numbers."

### Step 2.8: Test the model BEFORE training

```python
def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top = int(probs.argmax())
    return f"{id2label[top]} ({probs[top]:.0%} sure)"

test_sentences = [
    "i just got accepted into my dream internship",
    "i miss my family so much it hurts",
    "why would you lie to me like that",
    "i heard a noise downstairs and i am home alone",
]

print("BEFORE training:")
for s in test_sentences:
    print(f"  {s!r} -> {predict(s)}")
```

**Expected:** basically random guesses, often the same label for everything, at low confidence (about 16–25%).

> **Say:** "Remember these answers. They're garbage, and that's the point. Let's train it."

### Step 2.9: Train! (about 8–12 minutes)

```python
import numpy as np
from transformers import TrainingArguments, Trainer

def compute_metrics(eval_pred):
    logits, true_labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == true_labels).mean())}

training_args = TrainingArguments(
    output_dir="emotion-model",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=25,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    compute_metrics=compute_metrics,
)

trainer.train()
```

**Expected:** a progress bar counting up to **250 steps**. A table appears with **Training Loss** going *down* and, after each epoch, **Accuracy**.

> **Say (while it trains, this is a good time for Q&A):**
> - "An **epoch** is one full pass through the 2,000 training examples. We're doing 2."
> - "**Batch size 16** means the model looks at 16 examples, checks how wrong it was, and nudges its weights. 2,000 ÷ 16 = 125 nudges per epoch, × 2 epochs = 250 steps."
> - "**Loss** measures how wrong the model is. We want it going down."
> - "**Learning rate** is how big each nudge is. Too big and it overshoots, too small and it takes forever."
> - "Notice this is slow, because we're on a tiny 2-CPU machine. In Part 3 we'll see how AWS lets you rent bigger machines for exactly as long as you need them."

**If it's running too slowly** (more than 15 min estimated): stop the cell (■ button), change `range(2000)` to `range(1000)` in Step 2.4, and rerun Steps 2.4 → 2.9.

### Step 2.10: Test the model AFTER training

```python
results = trainer.evaluate()
print(f"Accuracy on unseen test data: {results['eval_accuracy']:.1%}\n")

print("AFTER training:")
for s in test_sentences:
    print(f"  {s!r} -> {predict(s)}")
```

**Expected:** accuracy well above random (random would be about 17%). It usually lands somewhere around 60–85% with this small slice, and the four sentences now get sensible labels.

> **Say:** "Same model, same sentences. It went from random to actually useful in about 10 minutes, with only 2,000 examples. That's the power of **fine-tuning** a pretrained model instead of training from scratch."

### Step 2.11: Try your own sentences

```python
my_sentence = "type anything you want here"
print(predict(my_sentence))
```

> **Say:** "Change the text and rerun it. Try to trick it. Sarcasm usually breaks it. Why do you think that is?" (Answer: the training data barely has sarcasm, and the model only learned the patterns it saw.)

---

## PART 3: Do it the cloud way with a SageMaker training job (20–25 min)

> **Say (AWS club can lead this part):** "What we just did is great for experimenting, but real teams don't train in a notebook. They package the training code into a script, and SageMaker:
> 1. starts a fresh machine (any size, including GPUs),
> 2. runs the script inside a **Docker container**,
> 3. saves the trained model to **S3** (AWS's file storage),
> 4. **shuts the machine down automatically**, so you only pay for the minutes it ran.
>
> The notebook stays tiny and cheap. The heavy lifting happens elsewhere."

Keep working in the **same notebook**, adding cells below.

### Step 3.1: Make a folder for the training code

```python
import os
os.makedirs("code", exist_ok=True)
print("Folder created")
```

### Step 3.2: Write the list of libraries the training machine needs

Copy this **exactly**. The first line, `%%writefile`, saves the cell to a file instead of running it.

```python
%%writefile code/requirements.txt
transformers==4.44.2
datasets==2.21.0
accelerate==0.33.0
```

**Expected:** `Writing code/requirements.txt`

### Step 3.3: Write the training script

This is the same code from Part 2, put into one file.

```python
%%writefile code/train.py
import argparse
import os

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_size", type=int, default=2000)
    args = parser.parse_args()

    # SageMaker tells us where to save the model via this environment variable.
    # Anything saved here gets uploaded to S3 automatically when the job ends.
    model_dir = os.environ.get("SM_MODEL_DIR", "model")
    model_name = "distilbert-base-uncased"

    ds = load_dataset("dair-ai/emotion")
    labels = ds["train"].features["label"].names
    train_ds = ds["train"].shuffle(seed=42).select(range(args.train_size))
    test_ds = ds["test"].shuffle(seed=42).select(range(500))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_tok = train_ds.map(tokenize, batched=True)
    test_tok = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={i: n for i, n in enumerate(labels)},
        label2id={n: i for i, n in enumerate(labels)},
    )

    def compute_metrics(eval_pred):
        logits, true_labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == true_labels).mean())}

    training_args = TrainingArguments(
        output_dir="/tmp/checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    print(f"FINAL TEST ACCURACY: {results['eval_accuracy']:.1%}")

    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()
```

**Expected:** `Writing code/train.py`. In the left file browser you can now open the `code` folder and see both files.

### Step 3.4: Connect to SageMaker and check the container exists

```python
import sagemaker
from sagemaker import image_uris

session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = session.boto_region_name

print("Region:", region)
print("Role:", role)
print("Default S3 bucket:", session.default_bucket())

# Sanity check: make sure AWS has a CPU PyTorch training container for this version
image = image_uris.retrieve(
    framework="pytorch", region=region, version="2.1", py_version="py310",
    instance_type="ml.m5.xlarge", image_scope="training",
)
print("Training container:", image)
```

**Expected:**
- `Region: us-east-1`
- a role ARN like `arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole-…`
- a bucket name like `sagemaker-us-east-1-123456789012`
- a container image URI ending in something like `pytorch-training:2.1-cpu-py310`

> **Say:**
> - "The **role** is an IAM permission slip. It's what allows SageMaker to read and write our S3 bucket on our behalf."
> - "The **S3 bucket** is where the trained model will land."
> - "That **training container** is a Docker image that AWS maintains with PyTorch pre-installed. Our script runs inside it. This is exactly what Docker is for: same environment, every time, on any machine."

### Step 3.5: Launch the training job

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",          # the script to run
    source_dir="code",               # folder uploaded to the job (includes requirements.txt)
    role=role,
    framework_version="2.1",
    py_version="py310",
    instance_type="ml.m5.xlarge",    # free-tier eligible CPU machine
    instance_count=1,
    hyperparameters={"epochs": 2, "train_size": 2000},
    base_job_name="emotion-workshop",
)

estimator.fit(wait=True)
```

**Expected:** logs start streaming into the notebook:
1. `Starting - Starting the training job…`
2. `Starting - Preparing the instances for training…` (this takes **3–5 minutes**, as AWS is starting a machine for you)
3. `Downloading - Downloading the training image…`
4. `Training - Training image download completed…` followed by `pip install` output, then the same progress bars you saw in Part 2
5. `FINAL TEST ACCURACY: …%`
6. `Completed - Training job completed`, plus **Billable seconds: ~600–900**

> **Note on why this uses `PyTorch` instead of `HuggingFace`:** SageMaker also has a `HuggingFace` estimator, but its training containers only support GPU instances. Using the PyTorch container plus a `requirements.txt` that installs `transformers` lets us stay on a free-tier CPU machine. On a GPU account, the `HuggingFace` estimator is the usual choice.

### Step 3.6: Watch it in the AWS Console (while it runs)

While the job is in the "Preparing" phase, switch to the **AWS Console browser tab** (the one with the SageMaker AI dashboard):

1. In the SageMaker AI left menu, click **Training** → **Training jobs**.
2. Click the job named `emotion-workshop-…` at the top of the list.
3. Point out on the projector:
   - **Status:** InProgress → Completed
   - **Instance type:** ml.m5.xlarge
   - **Monitor** section → **View logs**, which opens CloudWatch (the same logs as in the notebook)
   - **Output** section → **S3 model artifact**, where the trained model ends up

> **Say:** "That machine didn't exist 5 minutes ago, and it'll be gone in 10. You can change `instance_type` to a GPU like `ml.g5.xlarge` and the **exact same script** runs faster. That's the point of the cloud."

### Step 3.7: Find your trained model in S3

```python
print("Your trained model is here:")
print(estimator.model_data)
```

**Expected:** a path like `s3://sagemaker-us-east-1-…/emotion-workshop-…/output/model.tar.gz`

Optional, click-through version: in the AWS Console search bar, type **S3** → open the bucket named `sagemaker-us-east-1-…` → open the `emotion-workshop-…` folder → `output` → there's `model.tar.gz`.

### Step 3.8 (optional, if time allows): Use the model the cloud trained

```python
!aws s3 cp {estimator.model_data} ./job-model.tar.gz
!mkdir -p job-model && tar -xzf job-model.tar.gz -C job-model

from transformers import pipeline
cloud_clf = pipeline("text-classification", model="job-model")

for s in test_sentences:
    print(f"  {s!r} -> {cloud_clf(s)[0]}")
```

**Expected:** predictions similar to Step 2.10. This model was trained on a completely different machine and pulled back from S3.

---

## PART 4: Clean up (5 min, DO NOT SKIP)

> **Say:** "The training job already shut itself down. But your **notebook machine is still running** until you stop it. Let's turn it off so nobody gets a surprise bill."

1. In JupyterLab: **File → Shut Down** → confirm. (You can close that tab.)
2. In the **SageMaker Studio** tab: left panel → **JupyterLab** → find `emotion-workshop` → click **Stop**. Wait until the status says **Stopped**.
3. Double-check: in Studio's left panel, click **Running instances** (or the **Home** page "running instances" count). It should show **nothing running**.
4. You do **not** need to delete the training job, since finished jobs cost nothing.
5. The model file in S3 is about 250 MB, which is well within the S3 free tier. If you want it gone: **S3** → your `sagemaker-us-east-1-…` bucket → select the `emotion-workshop-…` folder → **Delete**.

**Expected:** no running spaces or instances.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Can't find the domain, Studio, or your notebook | Wrong region | Top-right of the console → switch to **US East (N. Virginia)** |
| "Open Studio" is greyed out or there's no user profile | SageMaker domain not created or still creating | SageMaker AI → **Domains**. If it's missing, do the pre-work step 5. If it's "Pending", wait. |
| Step 2.2 says `ModuleNotFoundError` or shows old versions | Kernel wasn't restarted after install | **Kernel → Restart Kernel**, then rerun from Step 2.2 |
| Step 2.3 hangs or fails to download | Venue Wi-Fi is fine (the download happens on AWS), so it's usually a temporary Hugging Face Hub issue | Rerun the cell. If it still fails, pair the student with a neighbour. |
| `eval_strategy` error in Step 2.9 | Very old `transformers` version loaded | Restart the kernel (Step 2.1 installed a new one). As a last resort, rename it to `evaluation_strategy`. |
| Training in Step 2.9 is very slow (>20 min) | `ml.t3.medium` is small and burstable | Stop the cell, use `range(1000)` in Step 2.4, and rerun 2.4 → 2.9 |
| Kernel dies or "out of memory" | Something else is using RAM | **Kernel → Restart Kernel**, then rerun Step 2.3 onward (skip 2.1) |
| `ImportError: cannot import name 'PyTorch'` or estimator errors in Step 3.5 | SageMaker SDK v3 installed instead of v2 | Rerun Step 2.1 (it pins `sagemaker<3`), restart the kernel, then rerun 3.4 → 3.5 |
| Step 3.4 `image_uris.retrieve` raises `ValueError` | That container version isn't available in this region or SDK version | Try `version="2.2"` (and the same in 3.5). Find this during the dry run. |
| Step 3.5 `ResourceLimitExceeded` | Account has no quota for `ml.m5.xlarge` training | Very new accounts sometimes need a quota bump: **Service Quotas** → **Amazon SageMaker** → search `ml.m5.xlarge for training job usage` → **Request increase** → 1. In the meantime, pair up with a neighbour. |
| Step 3.5 job **Failed** | Usually a typo in `train.py` or a `pip` failure | Scroll up in the logs to the first red `Error`. Most often it's a copy-paste problem in Step 3.3. Re-copy the cell exactly. |
| `AccessDenied` on S3 in Step 3.8 | Role permissions | Skip 3.8. It's optional. |

---

## Presenter notes

- **Pacing:** Part 2's training step (2.9) is your built-in Q&A and concepts break. Part 3's "Preparing the instances" wait is the AWS club's moment to explain Docker containers, IAM and S3 using the console.
- **Keep everyone in sync:** after each step, ask people to put a hand up or put a sticky note on their laptop when they get the **Expected** output. Don't move on until most of the room is there. TAs handle stragglers.
- **If a student falls far behind:** have them skip to Step 2.10 output on the projector and rejoin at Part 3. Part 3 is self-contained once Steps 2.1 and 3.1–3.4 are done.
- **Before the event, test the four sentences in Step 2.8** in your dry run and swap any that come out wrong after training. A clean "wow" moment matters.