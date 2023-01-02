# ADL-Final

做的是 Hahow 的 user / subgroup 預測。

大多數`.py`檔的參數都可以用`python script.py -h`來看，但是因為重複性太高，我讓他們全部都用一樣的參數，用一樣的parse function。如果想要知道什麼參數需要給的話，可以該給的全給、看參數的`help`看看是否相關，或是直接去對應的`.sh`檔看範例參數。

步驟如下：
1. 在`feature.sh, train.sh, pred.sh`裡面**直接改成你要的參數**，包含所有的path都要確認是否正確。
2. 照順序去跑這三個script，最後的結果會在指定的`tt_model_dir`裡面。

## Feature Extraction
可以產出一些前處理過的dataset，產出的dataset檔會給後面的script當作參數。
對應的script是`feature_extract.py`和`feature.sh`。在執行後面的script前需要用這個script產出必要的檔案 (i.e. `feature_ds, useritem_example_ds`)。

## Training
用的是`python two-towel.py --task "train"`。也可以直接執行`bash train.sh`，但是需要修改路徑等等。
比較特別的是需要用`feature_extract.py`產出前處理過後的dataset。可以用`utils.py`裡，`ArgumentManager.ttModelArguments`的參數去調整模型。如果要從某個checkpoint去load的話注意要調整成一樣的。另外，`tt_model_dir`(將)會是一個directory path，裡面有對應的模型。

## Predicting
用的是`python two-towel.py --task "predict"`產生course和`generate_subgroup/`的兩個script產生subgroup。對應`pred.sh`。