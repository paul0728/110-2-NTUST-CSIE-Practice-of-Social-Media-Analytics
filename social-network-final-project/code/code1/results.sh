#!/bin/bash

#Declare a string array
embeddings=("GEMSEC" "GEMSECWithRegularization" "DeepWalk" "DeepWalkWithRegularization")
 
#Declare a string array
DatasetArray=("./gemsec_facebook_dataset/facebook_clean_data/artist_edges.csv" \ 
              "./gemsec_facebook_dataset/facebook_clean_data/athletes_edges.csv" \
              "./gemsec_facebook_dataset/facebook_clean_data/company_edges.csv" \ 
              "./gemsec_facebook_dataset/facebook_clean_data/government_edges.csv" \
              "./gemsec_facebook_dataset/facebook_clean_data/new_sites_edges.csv" \
              "./gemsec_facebook_dataset/facebook_clean_data/politician_edges.csv" \
              "./gemsec_facebook_dataset/facebook_clean_data/public_figure_edges.csv" \
              "./gemsec_facebook_dataset/facebook_clean_data/tvshow_edges.csv")

for val in ${DatasetArray[*]}
do
  for embedding in ${embeddings[*]}
  do
    python Facebook_Link_Prediction_paper.py --embedding $embedding --dataset_path $val
  done
  python Facebook_Link_Prediction.py --PCA --dataset_path $val 
  python Facebook_Link_Prediction.py --dataset_path $val 
done

python csv_transpose.py


