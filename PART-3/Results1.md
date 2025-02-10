
---

# Comprehensive Report on Static Word Embeddings Bias Evaluation

This report describes the process and outcome of running the bias evaluation script using the GoogleNews word embeddings. The script calculates a WEAT‐like effect size to quantify associations between two groups of target words and two groups of attribute words.

## Overview

- **Command Executed:**  
  ```bash
  python static_embeddings_bias.py --model_path data/models/GoogleNews-vectors-negative300.bin.gz
  ```

- **Results Obtained:**
  - **Effect Size:** 1.679
  - **Sample Size:** 8
  - **Target Set 1 Mean Differential Association:** 0.008
  - **Target Set 2 Mean Differential Association:** -0.160

These numbers summarize the bias measured between the two target word sets with respect to two attribute word sets.

## Script Description

The provided Python script performs the following main steps:

1. **Loading the Embeddings:**
   - The script uses the `gensim` library to load the word vectors from the specified model file (`GoogleNews-vectors-negative300.bin.gz`).  
   - It prints a message indicating that the model is being loaded.

2. **Defining Target and Attribute Sets:**
   - **Target Set 1:** `['doctor', 'engineer', 'scientist', 'programmer']`
   - **Target Set 2:** `['nurse', 'teacher', 'librarian', 'homemaker']`
   - **Attribute Set 1:** `['he', 'man', 'his', 'male']`
   - **Attribute Set 2:** `['she', 'woman', 'her', 'female']`
   - These sets are chosen to assess gender bias in associations between certain professions and gendered terms.

3. **Computing Differential Associations:**
   - For each word in the target sets, the script retrieves its embedding (if available).
   - It then computes cosine similarities between each target word and every word in each attribute set.
   - The **mean cosine similarity** is calculated for:
     - Each target word with the words in attribute set 1.
     - Each target word with the words in attribute set 2.
   - The **differential association** for a target word is the difference between these two mean cosine similarities:
     - \( s(w, A_1, A_2) = \text{mean}_{a\in A_1} \cos(w, a) - \text{mean}_{b\in A_2} \cos(w, b) \)

4. **Calculating the WEAT Effect Size:**
   - The script groups the differential associations by target set.
   - It computes:
     - The mean differential association for **Target Set 1** (reported as 0.008).
     - The mean differential association for **Target Set 2** (reported as -0.160).
   - The overall **effect size** is then defined as:
     -  
       \[
       d = \frac{\text{mean differential association of Target Set 1} - \text{mean differential association of Target Set 2}}{\text{standard deviation of all differential associations}}
       \]
   - In this run, the effect size computed is **1.679**.
   - The **sample size** is the total number of target words used (4 from each set, yielding 8).

5. **Missing Words Check:**
   - The evaluator keeps track of words that are not found in the vocabulary.
   - In your run, all provided words were present, so no missing words were reported.

## Interpretation of the Results

- **Effect Size (1.679):**  
  This is a large effect size by common conventions (for instance, Cohen’s d values above 0.80 are generally considered large). It suggests that there is a strong differential association between the two target sets with respect to the attribute sets.

- **Target Set Differential Associations:**
  - **Target Set 1 Mean:** 0.008  
    The near-zero mean differential association implies that the words in this set (e.g., “doctor”, “engineer”, etc.) are almost equally associated with both sets of gendered attribute words.
  - **Target Set 2 Mean:** -0.160  
    A negative mean differential association indicates that words in this set (e.g., “nurse”, “teacher”, etc.) tend to have a stronger association with attribute set 2 (e.g., “she”, “woman”, “her”, “female”) than with attribute set 1.
  
- **Sample Size (8):**  
  The effect size is based on a small sample (only 8 words in total). Although the calculated effect size is large, note that using a larger set of target words may provide a more stable and generalizable estimate.

## Discussion

- **Bias Measurement Approach:**  
  The methodology is inspired by the Word Embedding Association Test (WEAT) introduced by Caliskan et al. (2017). By comparing the associations of two sets of target words with two sets of attribute words, the script provides a quantifiable measure of bias in the embedding space.

- **Implications:**  
  The large effect size indicates that the static GoogleNews embeddings encode a strong bias in the context of these specific word sets. In practice, this might reflect stereotypical associations (e.g., associating certain professions with a particular gender).

- **Limitations:**
  - **Small Target Sets:**  
    The evaluation is performed on only 4 words per target set, which is a very small sample. Results could be sensitive to the choice of words.
  - **Generality:**  
    Although the effect size is high, it only reflects bias with respect to these particular sets of words. A more comprehensive analysis would use larger and more diverse sets of targets and attributes.
  - **Model and Data Issues:**  
    The GoogleNews vectors were trained on older data and may reflect historical biases that do not necessarily represent current language use.

## Conclusion

The script demonstrates how to compute a WEAT-like effect size for static word embeddings. With an effect size of 1.679, the analysis reveals a strong bias: while the first set of target words shows nearly neutral associations, the second set is significantly skewed toward the female attribute set. Although the sample size is small, the methodology clearly illustrates how biases in word embeddings can be quantitatively assessed.

---
