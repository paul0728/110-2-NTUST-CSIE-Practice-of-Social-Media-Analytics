useage:
bash results.sh 
跑出所有組合結果(每種設定5張圖) + 4 個csv檔案(Train/Test RF/GBDT 的f1)

python csv_transpose.py
將csv轉正(在results.sh中會執行)

folder:
data:儲存all feature+pca 完的結果+unconnected_link
results:所有結果
