#!/bin/bash
models=("GEMSEC" "GEMSECWithRegularization" "DeepWalk" "DeepWalkWithRegularization")

#Declare a string array
DatasetArray=("artist" \ 
              "athletes" \
              "company_edges" \ 
              "government" \
              "new_sites" \
              "politician" \
              "public_figure" \
              "tvshow")
               
for model in ${models[*]}
do
  echo $model
  #create output folder
  if [ ! -d output/"${model}" ] ; then
    mkdir -p output/"${model}"/assignments
    mkdir -p output/"${model}"/cluster_means
    mkdir -p output/"${model}"/embeddings
    mkdir -p output/"${model}"/logs
  fi
  
  for val in ${DatasetArray[*]}
  do
    echo $val
    python src/embedding_clustering.py \
      --input data/"${val}"_edges.csv  \
      --embedding-output output/"${model}"/embeddings/"${val}"_embedding.csv \
      --log-output output/"${model}"/logs/"${val}"_log.json \
      --cluster-mean-output output/"${model}"/cluster_means/"${val}"_means.csv \
      --assignment-output output/"${model}"/assignments/"${val}".json \
      --num-of-walks 1 
  done 
done

