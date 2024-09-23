#!/bin/bash


#Declare a string array
DatasetArray=("tvshow")

for val in ${DatasetArray[*]}
do
  #多維
  
  echo $val
  echo "Gemsec"
  python main.py --dataset $val -e -v -r -g
  
  echo $val
  echo "GemsecWithRegularization" 
  python main.py --dataset $val -s -v -r -g
  
  echo $val
  echo "Deepwalk" 
  python main.py --dataset $val -l -v -r -g
  
  echo $val
  echo "DeepwalkWithRegularizatio"
  python main.py --dataset $val -k -v -r -g
  
  echo $val
  echo "Node2vector"
  python main.py --dataset $val -x -v -r -g
  
  #單維(有pca)
  echo $val
  echo "PCA"
  python main.py --dataset $val -a -p -m -d -w -n -v -r -g
  
  #單維(無pca)
  echo $val
  echo "noPCA"
  python main.py --dataset $val -a -v -r -g 
done










