## Heart disease prediction
- 利用Heart disease資料集進行有無Heart disease的預測，並探討在類別數量有明顯差距時，該如何進行資料前處理

## 模型預測結果
1. 資料處理步驟:
	- 針對`heart_2020_cleaned.csv`進行encoding，將文字資料轉為數值。並將前70%的資料作為training，剩餘30%作為testing
	- 針對training進行特徵篩選，刪除掉高相關的特徵，這一步會移除掉`PhysicalActivity`與`Race`兩個欄位
	- 針對training進行資料增生，確保罹病族群與健康族群的樣本數一致。資料增生的方式有兩種
		- age&sex: 依照年齡性別增生對應的資料
		- total: 直接增生罹病族群的資料
	- 將training中再拆分成80%的train與20%的validation，並將train中的資料依照0101...(HeartDisease)的方式排序
	- 針對train做MinMaxScaler()，並讓的validation與testing fit
	- 針對train進行建模，此處納入NN與1D CNN兩種不同建模策略，並設定monitor指標為ReCall
2. 建模結果:
	
|架構|over sample| output activation| compile loss| train result| testing result|
|----|-----------|------------------|-------------|-------------|---------------|
|  NN|      total|           sigmoid|binary_crossentropy|loss: 0.4840, recall: 0.8396, acc: 0.7720|auc: 0.8376, tpr: 0.8143, tnr: 0.7080|
|  NN|    age&sex|           softmax| mean_squared_error|loss: 0.1508, recall: 0.7766, acc: 0.7766|auc: 0.7815, tpr: 0.6523, tnr: 0.7663|
|  NN|      total|           softmax|binary_crossentropy|loss: 0.4791, recall: 0.7825, acc: 0.7825|auc: 0.8287, tpr: 0.7555, tnr: 0.7561|
| CNN|      total|           softmax|binary_crossentropy|loss: 0.4561, recall: 0.7868, acc: 0.7868|auc: 0.8228, tpr: 0.7865, tnr: 0.7219|

## 結果討論
1. 在正類別與負類別不平衡的情況下，可以使用`over sample`或是`down sample`的方式來消除筆數差異:
	- pandas: sample, sklearn: resample, imblearn: over_sampling/under_sampling 
	- 在此資料集中，使用`over sample`來針對`HeartDisease`進行資料增生，比較特別的是針對年齡性別進行增生(age&sex)的效果會比不劃分(total)來的低(testing AUC)。
2. NN設計:
	- 在NN最後一層Dense層的激勵函數使用`sigmoid`或`softmax`(loss皆為mse)，對於testing的TPR與TNR也會有影響，`sigmoid`的差異會比較大
	- 在最後一層Dense使用`softmax`時，loss選用`mean_squared_error`的testing TPR與TNR差異也會比loss選用`binary_crossentropy`來的大
	- output層使用`softmax`搭配`binary_crossentropy`可能可以得到較為平衡的結果
	- 調整fit參數時可以先使用預設，隨著訓練結果再去調整`batch_size`，`learning_rate`或是`early_stop中monitor的指標`
	- 在此資料集中，增加NN的隱藏層對於testing的結果並不會有太顯著的改變
3. 1D CNN結果:
	- 此資料使用1D CNN進行設計，在testing AUC上並不會有太明顯改變，可能是資料性質導致
4. 上述兩種DL模型在training與validation上表現都沒有太明顯的差異，但是在testing上卻會有不同的表現(AUC:78~83)，結果反映了DL對於未知資料還是具有不確定性，架構設計是十分重要的
