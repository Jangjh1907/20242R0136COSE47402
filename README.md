
# **ChemBERTa Fine-Tuning Project**

이 프로젝트는 **ChemBERTa 모델**을 사용하여 Canonical 및 Non-Canonical SMILES 데이터를 기반으로 모델을 Fine-Tuning하는 과정을 다룹니다. 또한 프로젝트의 전체 워크플로우는 Jupyter Notebook에서 제공됩니다.

---

## **구조**

```
\FinalProject
  ├── DL_ Final Project (Full).ipynb       # 전체 프로젝트 진행 workflow가 담겨 있음
  ├── Final_Project_SMILES.py              # Non-Canonical SMILES에 대해 Fine-Tuning 코드
  ├── Fianl_Project_Canonical_SMILES.py    # Canonical SMILES에 대해 Fine-Tuning 코드
  ├── Dataset                              # 데이터셋 폴더
       ├── qm9.csv                         # Training에 사용된 원본 데이터셋
       ├── (가공된 데이터셋들)             # Fine-Tuning에 사용된 추가 데이터셋
```

---

## **사용 방법**

### **1. 환경 설정**
ChemBERTa 모델을 실행하기 위해 Hugging Face `transformers` 라이브러리를 설치해야 합니다.

```bash
pip install transformers
```

---

### **2. 실험 수행**

#### **로컬에서 실행**
1. `Final_Project_SMILES.py` 실행 (Non-Canonical SMILES)
2. `Fianl_Project_Canonical_SMILES.py` 실행 (Canonical SMILES)

```bash
python Final_Project_SMILES.py
python Fianl_Project_Canonical_SMILES.py
```

#### **Google Colab에서 실행**
1. `\FinalProject\Dataset` 폴더 전체를 Colab에 업로드합니다.
2. `DL_ Final Project (Full).ipynb`를 Colab에서 열고 셀을 실행합니다.

---

## **데이터셋**

- **qm9.csv**: QM9 데이터셋은 Fine-Tuning에 사용된 원본 데이터입니다.
- 추가로 가공된 데이터셋은 `\FinalProject\Dataset` 폴더에 포함되어 있습니다.

---

## **참고 사항**

- **로컬 실행** 시 Python 3.7 이상과 `transformers` 라이브러리가 필요합니다.
- **Google Colab 사용** 시, 모든 데이터 파일을 업로드한 후 `.ipynb` 파일을 실행하세요.
