# readme


# 環境建置
	>使用conda 安裝 py36.yaml 以及 labelme.yaml
	ex : conda env create -f environment.yaml
		>若無法執行 請試著重新安裝或者清除快取及其他有可能干擾之軟體
		若真的不行 再嘗試使用 : conda create --name py36 python=3.6
								conda install -c conda-forge nb_conda
								pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
								conda install git
								pip install git+https://github.com/albumentations-team/albumentations
								pip install git+https://github.com/qubvel/segmentation_models.pytorch
								
								
	


# 資料前處理
	label to mask.ipynb 內有詳細步驟

# 程式檔案
	## 訓練
	> train.py 執行的主程式
		>執行方法: python train.py 
			>詳細參數可參考主程式內 def get_args() 副函式 
			ex:python train.py --amp -s 1 -l 1e-5 -f checkpoint.pth -b 2
			-s:scale 縮放倍率, --amp:使用nvidia mix precition 加速,-b:batch_size ,-f:讀取之前訓練的結果 ,-l:設定學習率
	### 資料結構
	├── train.py
	│
	├── data
	│   ├── imgs
	│	│	│── *.jpg
	│	│	│	...
	│	│   └── *.jpg
	│   ├── masks
	│	│	│── *.png
	│	│	│	...
	│	│   └── *.png
	│   └── 
	├── p
	│	│── predict.py
	│	│── utils
	│	│	│── data_loading.py
			│── ...
			└── 

	## 預測
	> predict.py 輸出預測結果的主程式 
		先使用Predict Preprocess.ipynb 程式產生預測輸入與輸出的對應檔名的txt文字檔,程式將會讀取文字檔作為參數傳入執行預測。
		ex:python predict.py
	>Predict Preprocess.ipynb
		使用jupyter notebook 等程式開啟 並cd至predict.py 路徑下
			model: 要輸入的模型的檔案位置
			input_folder:要預測的圖片的資料夾路徑
			output_folder:輸出預測mask位置
		執行完會在根目錄產生一個predictALL.txt 即完成。
	>調整預測參數的方式
		請進到./P/util/data_loading.py內 126 行的位置修改，最後一次驗證時有使用A.FancyPCA。

# 訓練過程
	
	## BEST MODEL
	1. 訓練過程
		> 100 epoch lr=1e-5 @17 > 50 epoch lr=1e-5 @18 > 50 epoch lr=8e-6 @19 > 80 epoch lr=4e-6 @21 > 80 lr=4e-6 from 46 @26 | END
		除了最後一段訓練，也就是@26 換了不同的隨機種子 其餘種子皆相同。
		並且最後一次訓練將所有的圖像增強都取消只留corp

