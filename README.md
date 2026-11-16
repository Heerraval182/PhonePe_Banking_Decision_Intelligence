Domain: Banking

# PhonePe Banking Decision Intelligence — Capstone

This repository implements the four parts of the IIT Mandi Banking capstone using the supplied synthetic data generator. The PhonePe setting is only a scenario: the records are invented and are not PhonePe customer data. This analysis is an academic exercise, not financial or credit advice.

## How to run

Use Python 3.10 or newer. From the repository root, run these commands in order:

```bash
python -m pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

Alternatively, run the complete sequence with:

```bash
python run_all.py
```

The generator should print:

```text
800 10 170 e85e7605b784f4142844f79ad2dd0b8e
```

All calculations are implemented with Python and NumPy. The project does not use Scikit-learn, TensorFlow, PyTorch, external APIs, downloaded datasets or paid services. The scripts regenerate the committed data and all files in `outputs/`.

## Part 1 findings

1. **Planning-mean test:** The estimated monthly spend is INR 8.9382 thousand, compared with the planning assumption of INR 8.0 thousand. The test statistic is 2.8930, which exceeds the two-tailed critical value 1.96, so H0 is rejected at the 5% level. A Type I error here would mean concluding that average borrower spend differs from INR 8,000 when the true population mean is actually INR 8,000. The difference is 0.9382 thousand, or about INR 938.20 per month; this is statistically significant and may also matter to budgeting, although business materiality depends on scale and costs.
2. **Sample size:** The calculated requirement is 1,293 records for a margin of error of INR 500 at 95% confidence, while the available dataset contains 800. The current sample is therefore smaller than the requested planning target.
3. **Gradient descent:** The absolute slope gap between gradient descent and the closed-form regression is 0.0000 to four decimal places. Standardising the predictor and repeatedly applying the full-batch gradients with the specified learning rate brings the iterative solution to the same least-squares line.
4. **Residual spread:** The raw residual standard-deviation ratio (upper/lower fitted-value halves) is 3.7525, showing a pronounced funnel pattern. For the log-log model the ratio is 1.0396, much closer to 1, so the log transformation substantially reduces the changing residual spread.
5. **Multiple regression:** The predictors with coefficient magnitude above 0.05 are X2 (failed transaction count), X3 (credit utilisation), X4 (average monthly inflow), and X6 (irregular salary flag). The multiple-regression adjusted R-squared is 0.7403, compared with 0.6561 for the log-log model. On this dataset, the extra predictors improve adjusted fit. Coefficients have different units and should not be interpreted as directly comparable importance scores.

## Part 2 findings

1. **Learning rate:** The 200-update losses from lowest to highest are 0.3703 (η = 1.0), 0.3776 (η = 0.1), and 0.5091 (η = 0.01). The smallest learning rate makes slower progress in the same number of updates; among these tested values, η = 1.0 gives the lowest loss.
2. **Important logistic weights:** The three largest absolute weights are X3 (0.9013), X1 (-0.5554), and X2 (0.5505). X3 has the largest odds ratio, 2.4629: a one-training-standard-deviation increase in credit utilisation multiplies the modelled odds of the outcome by about 2.46, holding the other standardised predictors fixed.
3. **Why accuracy is insufficient:** The blind model that predicts no defaults has accuracy 0.7875, while the logistic model at τ = 0.5 has accuracy 0.8042. Since only 51 of the 240 test records are positive, a high accuracy can hide missed defaults. Lowering τ from 0.5 to 0.3 raises recall from 0.4706 to 0.7451, while precision falls from 0.5455 to 0.4634.
4. **Logistic versus KNN:** At τ = 0.3 the logistic model has F1 = 0.5714 and recall = 0.7451. The selected KNN model (K = 5) has F1 = 0.3696 and recall = 0.3333. Logistic regression performs better on both reported measures for this test split.
5. **Cost-based alert threshold:** The logistic model's training-selected threshold is 0.15. Its test cost is INR 6,050, compared with INR 11,800 at τ = 0.5 and INR 20,400 when nobody is flagged. I would select logistic regression at τ = 0.15 for this exercise because it has the lowest of these reported test costs. The threshold is selected using training data to avoid using the test set to tune the model and produce an optimistically biased evaluation.

## Part 3 findings

1. **Elbow and WCSS:** The successive WCSS drops are 659.8282 (K=2 to 3), 108.0739 (K=3 to 4), 171.0575 (K=4 to 5), and 45.5466 (K=5 to 6). The specified elbow rule selects K = 3. The drops do not decrease steadily because the brief fixes reproducible starting rows; these initialisations are not guaranteed to find a global optimum. K=3 is used for the segment profiles as required.
2. **Borrower segments:**
   - **A — Established, steady users:** 322 records; mean account age 47.50 months, credit utilisation 0.2250 and outcome rate 1.86%. A reasonable action is to maintain low-friction servicing and offer optional repayment reminders rather than intensive collections.
   - **B — Active, higher-spend users:** 255 records; mean failed transactions 6.07, inflow INR 28.72 thousand and outcome rate 23.53%. A reasonable action is to monitor payment friction and provide proactive repayment reminders.
   - **C — High-utilisation, elevated-risk users:** 223 records; mean credit utilisation 0.7118, repayment score 38.52 and outcome rate 46.64%. This segment has the highest outcome rate. A reasonable action is to review affordability and offer early support before a missed payment.
3. **Hierarchical clustering:** At the largest-gap cut, single linkage gives four clusters: `{E1}`, `{E2,E3,E6}`, `{E4}`, and `{E5}`. Complete linkage and average linkage each give two clusters: `{E1,E4,E5}` and `{E2,E3,E6}`. Single linkage joins groups when even one close pair exists, which can create chaining; complete linkage considers the farthest cross-cluster pair, while average linkage uses the average of all cross-cluster distances. The different linkage rules explain the different cut structures.
4. **PCA:** PC1 is most positively associated with account age (loading 0.5317) and repayment score (0.4418), and negatively associated with credit utilisation (-0.5187) and failed transactions (-0.4100). Three components are needed to retain at least 80% of the variance (cumulative ratio 0.8616). Mean PC1 scores are 1.7773 for A, -0.7831 for B and -1.6709 for C, separating the established/steady group from the higher-utilisation groups along this component. The standardised feature means are zero before PCA.

## Part 4 findings

1. **Best first split:** S4 (account age X1 ≤ 24 months) has the highest information gain (0.1252) and Gini gain (0.0606), so it is the best of the four candidate splits by both criteria. It separates a younger-account side with a higher positive share from the other side. Growing a tree until every leaf is pure would fit noise and small-sample patterns; a maximum-depth limit or minimum-leaf-size rule can control this overfitting.
2. **Model comparison:** Logistic regression is the strongest of the five voting-table options by F1 (0.5714), with recall 0.7451. The stump has F1 0.5532 and recall 0.7647; hard voting has F1 0.5344 and recall 0.6863; soft voting has F1 0.5507 and recall 0.7451. Combining these models did not improve F1 over logistic regression alone on this test set.
3. **AdaBoost:** After round-one normalisation, the two points misclassified by h1 (points 3 and 7) have weights 0.25 each, for a combined weight of 0.50. Round two gives h2 a larger coefficient (α₂ = 0.8047) than h1 (α₁ = 0.5493) because h2's weighted error is lower (0.1667 versus 0.2500) under the updated distribution.
4. **Neural-network sizing:** The MLP layer parameters are 12,560; 272; and 170, totalling 13,002. The CNN layer parameters are 156; 2,416; 12,832; and 330, totalling 15,734. The required CNN option 2 shape and parameter table is:

| Layer | Output shape | Parameters |
|---|---:|---:|
| Input | 28×28×1 | 0 |
| Conv1 (5×5, 6 filters, padding 2) | 28×28×6 | 156 |
| ReLU | 28×28×6 | 0 |
| Max-pool 1 (2×2) | 14×14×6 | 0 |
| Conv2 (5×5, 16 filters, no padding) | 10×10×16 | 2,416 |
| ReLU | 10×10×16 | 0 |
| Max-pool 2 (2×2) | 5×5×16 | 0 |
| Flatten | 400 | 0 |
| Dense1 (32 neurons) | 32 | 12,832 |
| Output (10 neurons) | 10 | 330 |

For the RNN, W_xh is 128×64, W_hh is 128×128 and W_hy is 16×128; the hidden and output biases contain 128 and 16 parameters. The total is 26,768 parameters and does not depend on sequence length because the same recurrent weights are reused at every time step. For ten digit classes, ReLU hidden layers and softmax output with categorical cross-entropy are a suitable choice. Very deep plain networks can suffer vanishing gradients, making early layers learn slowly; skip connections provide a shorter path for information and gradients. The out-of-bag fraction for 560 draws with replacement is 0.3676.

## Model-choice memo

**To: Head of Lending-Risk Team**

I recommend the logistic-regression model for the structured borrower data, using the training-selected probability threshold of 0.15 for this synthetic-data evaluation. At this threshold, the test precision is 0.3814, recall is 0.8824 and F1 score is 0.5325; the estimated test cost is INR 6,050. This is lower than the cost of INR 11,800 at the conventional 0.5 threshold and INR 20,400 when no borrower is flagged. For a separate model-quality comparison at threshold 0.3, logistic regression has F1 = 0.5714 and recall = 0.7451, outperforming KNN (F1 0.3696; recall 0.3333). Thus, 0.15 is preferred for the stated cost objective, while 0.3 gives a higher F1 score. The stump reaches slightly higher recall (0.7647 at 0.3) but has lower F1 (0.5532), while neither voting approach improves F1 over logistic regression. I would therefore pilot the logistic model with human review and monitor false alarms, missed defaults and segment-level outcomes before operational use. The principal limitation is that the dataset is synthetic and the evaluation uses one fixed train/test split with only 51 positive test records. I would not prioritise cheque OCR or payment-remark classification before validating the structured model, because those are separate use cases and would not address the immediate evidence gap in default-risk performance.

## Limitations

1. The dataset is fully synthetic and cannot establish how real PhonePe users or partner-lender borrowers behave.
2. The evaluation uses one fixed stratified split; results may change under a different sample or time-based validation.
3. The test set contains only 51 positive outcomes, so precision, recall, F1 and cost estimates have sampling uncertainty.
4. The cost values are simplified assumptions from the assignment and do not include all operational, customer-experience, regulatory or fairness considerations.
5. The models do not include real-time transaction history, lender-specific policy rules, drift monitoring or a production feedback loop. The results should not be used to make real credit decisions.
