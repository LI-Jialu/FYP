# FYP_CUHK_CSCI4998 

## Features 
* SVM 
* SVM + Sliding window (3 labels)
* SVM + Sliding window (5 labels)
* SVM + Sliding window + PCA
* SVM + Sliding window + PCA + Condition 

## File Structure 
1. Collect Data 
    * Collect orderbook data: `dowload_order_book.py`
    * Collect fear and greed index: `select_fng.py`
    * Calculate *threshold*: `calculate_threshold.py`
2. Data preprocessing 
    * Split data into windows: `interval_split.py`
3. Model 
    * Select features, labels and construct original SVM model: `svm.py`
    * Build self-designed model from original SVM model: `model_builder.py`
4. Backtesting 
    * From prediction results to investment positions: `generate_pos.py`
    * Take actions of positions: `trading.py`
    * Evaluate performances: `backtesting.py` (The full version of backtesting is not used in `main.py`, but write a simplified version in `main.py`.)

## Branch Description 
1. Master 
2. CPU_Version 
    The version to be deployed locally 
3. GPU_Version 
    The version to be deployed on Colab
