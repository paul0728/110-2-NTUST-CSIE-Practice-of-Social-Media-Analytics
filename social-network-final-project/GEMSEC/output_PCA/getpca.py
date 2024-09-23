#!/bin/python
import csv
from sklearn.decomposition  import PCA
path = "./DeepWalk/embeddings/athletes_embedding.csv"

files = [ "artist_embedding.csv", "athletes_embedding.csv", "company_embedding.csv", "government_embedding.csv", "new_sites_embedding.csv", "politician_embedding.csv", "public_figure_embedding.csv", "tvshow_embedding.csv" ]
directories = [ "DeepWalk", "DeepWalkWithRegularization", "GEMSEC", "GEMSECWithRegularization" ]

from pathlib import Path
for directory in directories:
    outdir = Path(directory) / "PCA"
    outdir.mkdir(  parents=True, exist_ok=True )

    for file in files:
        infile = Path(directory) / "embeddings" / file
        outfile = outdir / file
        print( infile )

def PCA_transform_file( infile, outfile ):
    with open( path , 'r') as f:
        rows = csv.reader( f, delimiter=',' )
        data = list(rows)
    print( f"file data length: {len(data)}" )
    pca = PCA( n_components=1 )
    newdata = pca.fit_transform( data )

    with open( path2 , 'w') as f:
        writer = csv.writer( f, delimiter=',' )
        writer.writerows( newdata.tolist() )
    print( f"write file to {outfile} complete" )

### 
PCA_transform_file( path, 'a.csv' )

len( rows )

pca = PCA( n_components=1 )
newdata = pca.fit_transform( rows )
### 

### 
pca.singular_values_
pca.explained_variance_ratio_




