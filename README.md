# Augmentrum
(Unlocking the full augmentation spectrum in MRS data)

A modular system for spectral augmentation of synthetic and in-vivo MRS data, including subject sampling, coil/average selection, fast processing, and signal-level modifications.  

---

## 🔹 Features

- **Subject Sampling**
  - Reproducible train/val/test splits
  - Random or deterministic sampling

- **Coil / Average Selection**
  - Flexible coil and average sampling
  - Phased array coil modeling

- **Processing Pipeline (Raw → Processed)**
  - Coil combination
  - Alignment
  - Outlier removal
  - Averaging
  - Eddy current correction (ECC)
  - Truncation
  - Water removal
  - Frequency shifting
  - Phase correction

- **Signal-level Augmentation (Processed → Augmented)**
  - Noise corruption (amplitude, phase, frequency)
  - Background (baselines, random walks, random peaks)
  - Shimming issues (Lorentzian & Gaussian broadening)
  - Eddy current corruption

- **Pseudo-raw Simulation (Processed → Pseudo-Raw)**
  - Coils from phased array
  - Averages with scanner drifts, motion artifacts, etc.

---

## 📦 Package Structure

```
augmentrum/
├── augmentation/              # Augmentation modules
│   ├── signal_peturber.py
│   └── pipeline.py
│
├── processing/                # Raw → Processed pipeline
│   ├── raw_processor.py
│   └── utils.py
│
├── dataset/                   # Dataset loaders
│   ├── base_dataset.py
│   ├── fmrsinpain.py
│   └── brainbeats.py
│
├── sampling/                  # Subject splitting and sampling
│   ├── subject_splitter.py
│   └── coil_average_sampler.py
│
└── utils/                     # Helper functions
    └── philips.py
```

---

## 🚀 Installation

```bash
git clone https://github.com/yourname/augmentrum.git
cd augmentrum
pip install -e .
```

---

## ▶️ Example Usage

Run the example pipeline:

```bash
python -m examples.run_pipeline
```

Sample output:

```
Train batch: torch.Size([16, 2, 2048]) torch.Size([16, 2, 2048])
Validation batch: torch.Size([16, 2, 2048]) torch.Size([16, 2, 2048])
Test batch: torch.Size([16, 2, 2048]) torch.Size([16, 2, 2048])
```

---

## 📚 Roadmap

### ✅ Completed / Near-complete
- Raw → Processed pipeline (coil combination, alignment, outlier removal, averaging, ECC, truncation, water removal, frequency shifting, phase correction)  
- Noise corruption (amplitude, phase, frequency)  
- Dataloader module: automatic splits, online/offline sampling  

### 🚧 In Progress
- Background augmentation: baselines, random walks, peaks  
- Shimming issues: Lorentzian & Gaussian broadening  
- Eddy current corruption  
- Processed → pseudo-raw modeling (coils, averages)  

### 📝 To Do
- Performance benchmarking (processing speed check)  
- Make it pip-installable
- Documentation and tutorials