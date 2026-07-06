# Task 5: Decision Trees and Random Forests
**AI & ML Internship — Elevate Labs**

## Objective
Learn tree-based models (Decision Trees, Random Forests) for classification, using the Heart Disease dataset.

## Files in this Repository
| File | Description |
|---|---|
| `Task5_Decision_Trees_Random_Forests.ipynb` | Jupyter notebook with full code, plots, and outputs |
| `heart-selected-columns.csv` | Dataset used |
| `README.md` | This file |

## Dataset
`heart-selected-columns.csv` — 1025 rows, 10 columns:
`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak`

**Note on the label:** this file does not include the usual `target` (disease
present/absent) column. To still complete the classification task, the
notebook derives a **risk label** from a composite clinical risk score built
out of the 8 features most consistently linked to cardiac risk (age, resting
BP, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise
angina, and ST depression). Each is standardized and combined (flipping sign
for max heart rate, since lower is riskier), and patients above the median
score are labeled "high risk" (1), otherwise "low risk" (0). This is disclosed
in detail in the notebook (Section 2) — if you get access to the dataset's
real diagnosis column later, you can substitute it directly and the rest of
the pipeline works unchanged. This also explains why model accuracy comes out
very high (~98–100%): the label is a deterministic function of the features
themselves, unlike a real clinical diagnosis which has noise the model can't
see.

## What Was Done
1. Loaded and explored the dataset (distributions, correlations, missing values, duplicates).
2. Built the derived binary risk target (see note above).
3. Split data into train/test sets (80/20, stratified).
4. Trained a Decision Tree Classifier and visualized it.
5. Analyzed overfitting by sweeping `max_depth` from 1–20 and comparing train vs. test accuracy.
6. Trained a Random Forest (200 trees) and compared its accuracy to the tuned single tree.
7. Compared feature importances between the Decision Tree and Random Forest.
8. Evaluated both models with 5-fold stratified cross-validation.

## Key Results
- An unpruned Decision Tree overfits (perfect train accuracy, slightly lower test accuracy).
- Limiting tree depth (best found around depth ≈ 9 here) closes most of that gap.
- The Random Forest matched or slightly outperformed the single tree and had **lower variance** across cross-validation folds — the expected benefit of bagging.
- `oldpeak`, `thalach`, `exang`, and `cp` were consistently the most important features across both models.

## Tools Used
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn (`DecisionTreeClassifier`, `RandomForestClassifier`, `plot_tree`, `cross_val_score`).

## How to Run
1. Open `Task5_Decision_Trees_Random_Forests.ipynb` in Jupyter Notebook / JupyterLab.
2. Make sure `heart-selected-columns.csv` is in the same folder as the notebook.
3. Run all cells top to bottom (`Kernel > Restart & Run All`).

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Task5_Decision_Trees_Random_Forests.ipynb
```

---

## Interview Questions & Answers

**1. How does a decision tree work?**
A decision tree splits the dataset repeatedly on the feature and threshold that best separates the classes (or reduces error, for regression), forming a tree of if/else rules. Each internal node is a test on a feature, each branch is an outcome of that test, and each leaf is a predicted class (or value). To classify a new sample, you walk down the tree from the root, following the branch that matches the sample's feature values until you reach a leaf.

**2. What is entropy and information gain?**
Entropy measures the impurity/disorder of a set of labels — it's 0 when all samples in a node belong to one class, and highest when classes are evenly mixed. Information gain is the reduction in entropy achieved by splitting a node on a particular feature: the tree picks the split that gives the largest information gain (i.e., produces the "purest" child nodes) at each step.

**3. How is random forest better than a single tree?**
A random forest trains many decision trees, each on a random bootstrap sample of the data and a random subset of features at each split, then averages/votes their predictions. This reduces variance and overfitting compared to one deep tree, since individual trees' errors tend to cancel out when many diverse trees are combined. It's generally more accurate and robust to noise, at the cost of being less interpretable and more computationally expensive.

**4. What is overfitting and how do you prevent it?**
Overfitting is when a model learns the training data too closely — including its noise — so it performs very well on training data but poorly on unseen data. For decision trees, it happens when the tree grows too deep and creates highly specific rules for individual samples. It can be prevented by limiting tree depth (`max_depth`), requiring a minimum number of samples per leaf/split, pruning the tree after growing it, or using ensembles like random forests; more training data and cross-validation to tune these parameters also help.

**5. What is bagging?**
Bagging (Bootstrap Aggregating) is training multiple copies of a model on different random samples of the training data (drawn with replacement), then combining their predictions (majority vote for classification, averaging for regression). It reduces variance by averaging out the idiosyncrasies of any one training sample. Random forests use bagging plus random feature selection at each split.

**6. How do you visualize a decision tree?**
In scikit-learn, `sklearn.tree.plot_tree()` draws the tree directly with matplotlib, showing the split condition, impurity, sample counts, and predicted class at each node. Alternatively, `export_graphviz()` exports the tree to Graphviz's DOT format, which can be rendered into an image with the `graphviz` library for a more polished, exportable diagram.

**7. How do you interpret feature importance?**
Feature importance (e.g., `model.feature_importances_` in scikit-learn) reflects how much each feature contributes to reducing impurity across all the splits in the tree(s) it's used in — higher values mean the feature was more useful for making accurate splits. It doesn't indicate a fixed real-world causal effect and can be biased toward high-cardinality or correlated features, but it's a useful, quick way to see which inputs the model relies on most.

**8. What are the pros/cons of random forests?**
*Pros:* generally high accuracy, robust to overfitting compared to a single tree, handles both numerical and categorical features well, provides feature importance, requires little feature scaling/preprocessing, and handles missing/noisy data reasonably well.
*Cons:* less interpretable than a single tree ("black box" ensemble), slower to train and predict with many trees, larger memory footprint, and can still overfit on very noisy data or perform poorly on data with rare, informative interactions if not tuned properly.
