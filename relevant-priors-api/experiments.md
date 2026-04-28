# Experiments

## Approach

I implemented a hybrid model combining rule-based heuristics with TF-IDF similarity to determine whether a prior study is relevant to the current study.

## Features Used

- Study description (text)
- Study date (temporal gap)
- Modality (MRI, CT, etc.)
- Body region (brain, head, chest, etc.)

## Heuristic Rules

1. Same modality + same body region → always relevant
2. Brain ↔ Head mapping → considered relevant within 5 years
3. Very old studies (>12 years) → considered irrelevant
4. TF-IDF similarity used to capture semantic similarity between study descriptions

## What Worked

- Rule-based logic ensured strong clinical matches were always captured
- TF-IDF similarity helped match studies with different wording
- Batch processing improved performance and avoided timeouts

## What Did Not Work

- Pure rule-based approach missed semantic matches
- Computing TF-IDF per prior was inefficient and slower

## Final Solution

I combined rule-based filtering with TF-IDF similarity computed per case, which improved both performance and accuracy.

## Future Improvements

- Train a supervised model using labeled data
- Use medical ontologies for better body-part normalization
- Use embeddings (e.g., BERT) for deeper semantic understanding