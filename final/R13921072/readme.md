113-2 CV Final Project
Ganzin Team 2 你是我的眼






 --- Prerequisites ---
Python 3.8.10 Linux based Ubuntu

conda install -c conda-forge pytorch torchvision torchaudio matplotlib numpy Pillow scikit-learn scipy tqdm gdown opencv pandas
git clone https://github.com/AayushKrChaudhary/RITnet.git

Place dataset into dataset folder, input lists also need to be place within

dataset
| - CASIA-Iris-Lamp
| - CASIA-Iris-Thousand
| - Ganzin-J7EF-Gaze
| - list_CASIA-Iris-Lamp.txt
| - list_CASIA-Iris-Thousand.txt
| - list_Ganzin-J7EF-Gaze.txt
| - list_Ganzin-J7EF-Gaze_bonus.txt

 --- Prerequisites End ---







 --- Image Preprocessing - Execute in order ---
  **rubber sheet全過程耗時三小時，如有需要可直接將結果從我們的雲端下載**

seg.py - Undergo iris segmentation for dataset
circle_detection.py - detect center of iris and create circle around it, to label the area of iris
rubberSheetNormalisation.py - Produce flattened and normalized iris image



\\ Image Preprocessing 結果 (rubbersheet normalization images)，雲端下載鏈接:
https://drive.google.com/file/d/1jZH3EXLCYP3dIMlbxG8UDpQD-T3-rGrz/view?usp=sharing
(取代目前src現有的normalized_images文件夾)
 
 
 --- Image Preprocessing End ---





 --- Inference ---
inference.py - Find out results for the input, using the rubber sheet images results
Thousand and Lamp uses siamese_best_thou.pth (default), result are normalized within 0~1
Gaze and Bonus uses siamese_resnet101.pth, result are binarized using calculated threshold

Usage:
	python inference.py --dataset thousand
	python inference.py --dataset lamp
	python inference.py --dataset gaze --checkpoint './checkpoint/siamese_resnet101.pth'
	python inference.py --dataset bonus --checkpoint './checkpoint/siamese_resnet101.pth'

Output: result_thousand.txt, result_lamp.txt, result_gaze.txt, result_bonus.txt

 --- Inference End ---
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

