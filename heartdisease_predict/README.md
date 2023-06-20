## Heart disease prediction
1. 在正類別與負類別不平衡的情況下，可以使用`over sample`或是`down sample`的方式來消除筆數差異:
	- pandas: sample, sklearn: resample, imblearn: over_sampling/under_sampling 
	- 在此資料集中，使用`over sample`來針對`HeartDisease`進行資料增生，比較特別的是針對年齡性別劃分後再進行增生的效果會比不劃分來的低(testing AUC)。
2. NN設計:
	- 在NN最後一層Dense層的激勵函數使用`sigmoid`或`softmax`(loss皆為mse)，對於testing的TPR與TNR也會有影響，`sigmoid`的差異會比較大
	- 在最後一層Dense使用`softmax`時，loss選用`mean_squared_error`的testing TPR與TNR差異也會比loss選用`binary_crossentropy`來的大
	- output層使用`softmax`搭配`binary_crossentropy`可能可以得到較為平衡的結果
	- 調整fit參數時可以先使用預設，隨著訓練結果再去調整`batch_size`，`learning_rate`或是`early_stop中monitor的指標`
	- 在此資料集中，增加NN的隱藏層對於testing的結果並不會有太顯著的改變
3. 1D CNN結果:
	- 此資料使用1D CNN進行設計，在testing AUC上並不會有太明顯改變，可能是資料性質導致
4. 上述兩種DL模型在training與validation上表現都沒有太明顯的差異，但是在testing上卻會有不同的表現(AUC:78~83)，結果反映了DL對於未知資料還是具有不確定性，架構設計是十分重要的