113-2 CV Final Project
Ganzin Team 2 你是我的眼

Detailed description for the files in src

src
|- checkpoint (to store the models)
	|- siamese_best_thou.pth (used for Thousand and Lamp dataset, default in inference.py
  	 - siamese_resnet101.pth (used for Gaze and Bonus dataset)



|- dataset (to place original, no processed images of the eyes from dataset into here)



|- labels (storing segmentation labels (.npy files) results from seg.py)
	| - 3 folders collecting all labels, all files in .npy form
	Drive Download Link: https://drive.google.com/file/d/1NZI9PNasz-8bKKh1ieEZWfCZk-Fsps9e/view?usp=sharing



|- normalized_images (storing image results from rubberSheetNormalization.py)
 	| - 3 folders collecting rubber sheet normalized images
 	Drive Download Link: https://drive.google.com/file/d/1jZH3EXLCYP3dIMlbxG8UDpQD-T3-rGrz/view?usp=sharing
 
 
 
| - RITnet (RITnet from github, to undergo Iris segmentation work)
	Github link: git clone https://github.com/AayushKrChaudhary/RITnet.git



| - best_threshold_XXX.txt (4 files) (obtained from tune_threshold.py)



| - circle_detection.py (to determine the center point of the iris and draw circles to mark the area of the iris) (outputs iris_localization_results_all.json)



| - dataset.py (helps to process dataset)



| - inference.py (check the results of the code by using the model)



| - iris_localization_results_all.json (output result from circular_detection.py)



| - rename_testpairs.py (rename the directories from the given input list so that it fulfills our directory order)



| - result_XXX (4 files) (Final results for the whole pipeline)



| - rubberSheetNormalisation.py (Using data from the .json file to normalize the eyes' images using rubber sheet method) (results saved in normalized_images folder)



| - seg.py (undergo iris segmentation, results saved in labels folder)



| - SiameseNN.py (model data for siamese_resnet101.pth)
 
 
 
| - SiameseNN_bestthou.py (model data for siamese_best_thou.pth)
 
 
 
 
| - tune_threshold.py (tune the threshold for the input data, must run rename_testpairs.py first)
	
	Usage:
	python tune_threshold.py --val_pairdir output/CASIA-Iris-Thousand_normalized.csv --checkpoint ./checkpoint/siamese_best_thou.pth
	python tune_threshold.py --val_pairdir output/CASIA-Iris-Lamp_normalized.csv --checkpoint ./checkpoint/siamese_best_thou.pth
	python tune_threshold.py --val_pairdir ./output/Ganzin-J7EF-Gaze_normalized.csv --checkpoint './checkpoint/siamese_resnet101.pth'
	python tune_threshold.py --val_pairdir ./output/Ganzin-J7EF-Gaze_bonus_normalized.csv --checkpoint './checkpoint/siamese_resnet101.pth'
	
	



| - utils.py (various utilities)
 

