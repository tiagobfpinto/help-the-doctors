# Cross-Label Duplicates in Clinical Text

Group 64 — Tiago Pinto (25%), Gonçalo Narciso Rolo (25%), Gabriel Alexandre Lino Simões Ferreira (25%), Miguel Neves Antão Ermida Parreira (25%).

## 1. Introduction

Clinical notes can carry competing specialty labels. We ask: *How does cross-label duplication affect the evaluation and prediction of medical specialties from clinical text?* Collection-dependent rules can limit transfer [1].

## 2. Data

### 2.1. Data Analysis

CSV repairs retain 2,669 cases, 12 specialties and supplied labels. Surgery contributes 893 cases (33.5%), versus Dermatology's 25 (0.9%). Cross-label copies affect 1,528 cases (57.2%), including 72/79 Neurosurgery cases; descriptions, titles and transcriptions are identical. Incomplete records remain. We exclude `keywords`: all 2,219 nonempty values start with the target specialty.

### 2.2. Data Augmentation

To diversify minority-class n-grams, we recombine sections [2]. For the four classes with n<100 (Table 2), each eligible training note yields one variant borrowing a complete same-heading body from a same-class original in another duplicate group. Both notes need two distinct headings and chosen bodies of at least eight words. Titles/other sections remain unchanged; SHA-256(seed 0, fold, record ID) selects pairs. Five Dermatology examples were inspected before fitting: note 300's "A 14-day-old..." history receives 308's "This 34-year-old gentleman..."; infant sections remain, demonstrating clinical inconsistency. Copies repeat the same sources. Five-fold totals are 641 R/645 G mixtures, reused by LR/SVC.

## 3. Models

Using scikit-learn [3], multinomial Naive Bayes (NB) follows the reference: lowercased, Porter-stemmed `description` TF-IDF, Laplace smoothing (alpha=1) and empirical priors. Logistic regression (LR) and linear support vector classification (SVC) provide n-gram baselines [4]. They use separately L2-normalised `sample_name`/`transcription` TF-IDF blocks. Both compare originals, mixtures and copies; NB also differs in representation.

Exclusion (EX) masks labels from original training-fold matches of the lowercased, whitespace-normalised title/transcription pair, choosing the highest remaining score. Empty/no-match/all-masked cases use raw prediction; NB uses the same lookup. Unseen-label continuation is a corpus hypothesis, not a clinical rule.

## 4. Experimental Setup

Five-fold `StratifiedKFold` over rows (R) and `StratifiedGroupKFold` over notes (G) are shuffled with seed 0; seven configurations with/without EX share folds within each protocol. G groups normalised transcriptions, using titles when absent and unique IDs when both fields are empty. Features, weights, augmentation and EX indices use training folds only; pooled validation originals yield accuracy, macro F1 and per-class metrics. EX is inactive in G, which tests unseen notes within this corpus. Choices remain exploratory [5]. Before recombination fits, promotion required R+EX macro/Dermatology F1 improvements over family originals and SVC+EX, at most one accuracy point loss, accuracy >49% and all-class F1 ≥25%. Passing mixtures rank by macro F1, Dermatology F1, then accuracy; otherwise SVC+EX remains. The official 398-case test only generates predictions; its scores are unavailable.

### 4.1. Parameters/Hyperparameters

LR/SVC use lowercased word unigram/bigram TF-IDF, sublinear term frequency, `min_df=2` (minimum document frequency), no stemming/stopword removal, L2, C=0.25 (inverse regularisation) and at most 2,000 iterations. LR uses multinomial log loss; SVC uses one-versus-rest squared hinge. TF-IDF refits on originals and additions. LR/SVC weights N/(12*n_class) use original sample/class counts; after assignment to originals/additions, total weight is normalised to N. Copies control increased source influence; mixing also changes TF-IDF.

## 5. Results

Tables 1–2 give overall/per-class results; Figure 1 shows R confusions. For selected SVC+mix, EX raises R accuracy from 47.28% to 77.41%. All 1,269 matched validation labels are absent from training matches. Versus unaugmented SVC+EX, Dermatology precision rises 44.8% to 56.5%, with recall unchanged at 52% (13/25). LR mixtures lower R+EX macro F1 from 73.27% to 70.13%. SVC mixtures pass, but copies match rounded R+EX macro/Dermatology F1 (75.45/54.17%). Promotion does not establish a lexical benefit.

**Table 1. Out-of-fold results (%). R: rows; G: grouped notes; EX: exclusion. Bold: selected. Grouped EX equals raw.**

| System | CV | Accuracy | Macro F1 | Minimum F1 |
|---|---|---:|---:|---:|
| NB | R | 38.67 | 13.23 | 0.00 |
| NB | R+EX | 55.41 | 35.35 | 0.00 |
| NB | G | 41.14 | 15.26 | 0.00 |
| LR | R | 48.78 | 51.81 | 31.86 |
| LR | R+EX | 76.62 | 73.27 | 47.76 |
| LR | G | 55.75 | 58.19 | 45.08 |
| SVC | R | 46.16 | 49.77 | 23.82 |
| SVC | R+EX | 77.56 | 75.26 | 48.15 |
| SVC | G | 57.10 | 60.28 | 43.51 |
| **SVC+mix** | **R+EX** | **77.41** | **75.45** | **54.17** |
| SVC+mix | G | 56.88 | 60.92 | 42.73 |

**Table 2. Selected-system support, row-wise precision/recall/F1 and grouped F1 (%).**

| Specialty | n | P_R | R_R | F1_R | F1_G |
|---|---:|---:|---:|---:|---:|
| Cardiovascular-Pulmonary | 309 | 75.6 | 83.2 | 79.2 | 61.5 |
| Dermatology | 25 | 56.5 | 52.0 | 54.2 | 59.3 |
| Gastroenterology | 195 | 72.2 | 80.0 | 75.9 | 59.6 |
| General Medicine | 212 | 69.7 | 89.2 | 78.3 | 76.2 |
| Neurology | 184 | 69.3 | 71.2 | 70.2 | 58.4 |
| Neurosurgery | 79 | 51.5 | 84.8 | 64.1 | 45.9 |
| Obstetrics-Gynecology | 136 | 72.1 | 87.5 | 79.1 | 63.5 |
| Ophthalmology | 73 | 76.9 | 95.9 | 85.4 | 72.3 |
| Orthopedic | 300 | 73.7 | 83.3 | 78.2 | 57.5 |
| Psychiatry-Psychology | 41 | 84.1 | 90.2 | 87.1 | 83.1 |
| Radiology | 222 | 73.5 | 73.9 | 73.7 | 51.1 |
| Surgery | 893 | 96.1 | 68.6 | 80.1 | 42.7 |

![Figure 1. Selected-system row-wise confusion matrix. Rows: supplied labels; columns: predictions. Class order: Table 2.](paper_assets/confusion_matrix.png)

## 6. Discussion

Surgery to Orthopedic is the most frequent remaining confusion (66 cases). Case 1892, *Wrist Ganglion Excision*: supplied Surgery, predicted Orthopedic. "After administering appropriate antibiotics and general anesthesia, the upper extremity was prepped and draped in the usual standard fashion. The arm was exsanguinated with an Esmarch and tourniquet inflated to 250 mmHg. I made a transverse incision directly over the ganglion." Operative language and wrist anatomy plausibly suggest different tags; EX has no training match. This is interpretation, not feature attribution.

Case 309, *Cheek-Neck Facelift*: supplied Dermatology, predicted Radiology. "Temporal cheek-neck facelift (CPT 15825). Submental suction assisted lipectomy (CPT 15876)… This patient is a 65-year-old female who has progressive aging changes of the face and neck. The patient demonstrates the deformities described above and has requested surgical correction." EX masks Surgery from an identical training copy but chooses an incorrect remaining label.

Case 1383, *Anterior Lumbar Fusion*: supplied Orthopedic, predicted Neurosurgery. "Dr. X made the approach and once we were at the L5-S1 disk space, we removed the disk and we placed a 13-mm PEEK vertebral spacer filled with a core of bone taken from the L5 vertebral body." Identical corpus copies carry Orthopedic, Neurosurgery and Surgery. Only Surgery is masked here; Neurosurgery remains available, illustrating ambiguity in these corpus annotations.

Cross-label copies create conflicting supervision yet enable annotation completion, answering the research question. EX's corpus assumption can fail on repeated labels; approximate copies, patient dependence and selection bias remain. Synthetic patient coherence is distinct from predictive benefit.

## 7. Future Work

Independent collections, adjudicated multi-label notes and clinician-reviewed augmentation would test transfer; nested selection would limit development bias.

**LLM use.** ChatGPT assisted data preprocessing and LaTeX preparation; Claude helped check theoretical reasoning.

## Bibliography

[1] R. Geirhos et al. [Shortcut Learning in Deep Neural Networks](https://doi.org/10.1038/s42256-020-00257-z). Nature Machine Intelligence, 2020.

[2] S. Yoon, G. Kim and K. Park. [SSMix: Saliency-Based Span Mixup for Text Classification](https://aclanthology.org/2021.findings-acl.285/). Findings of ACL-IJCNLP, 3225–3234, 2021.

[3] F. Pedregosa et al. [Scikit-learn: Machine Learning in Python](https://jmlr.org/papers/v12/pedregosa11a.html). JMLR, 12:2825–2830, 2011.

[4] S. Wang and C. Manning. [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/). ACL, 90–94, 2012.

[5] G. C. Cawley and N. L. C. Talbot. [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://jmlr.org/papers/v11/cawley10a.html). JMLR, 11:2079–2107, 2010.
