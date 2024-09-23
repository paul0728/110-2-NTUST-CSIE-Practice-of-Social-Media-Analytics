python src/embedding_clustering.py \
--input data/tvshow_edges.csv  \
--embedding-output output/GEMSEC/embeddings/tvshow_embedding.csv \
--log-output output/GEMSEC/logs/tvshow_log.json \
--cluster-mean-output output/GEMSEC/cluster_means/tvshow_means.csv \
--assignment-output output/GEMSEC/assignments/tvshow.json \
--num-of-walks 1 
