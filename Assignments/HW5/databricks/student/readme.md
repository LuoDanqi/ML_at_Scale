# Benchmarks

## Dataset:
/data/HW5/all-pages-indexed-out.txt   
2GB

---

## Cluster 6.4 ML
KyleH (small cluster)   
Driver - m4.xlarge, 16.0 GB Memory, 4 Cores, 0.75 DBU   
1 Worker - m4.xlarge, 16.0 GB Memory, 4 Cores, 0.75 DBU   


### PageRank version 1
trained 10 iterations in 8302.966378688812 seconds. (2.3 hours)

### PageRank version 2 
#### (a different implementation which doesn’t care about mass adding to 1 - only rank matters)
trained 10 iterations in 5790.945766210556 seconds. (1.6 hours)

### PageRank Graphframes
Command took 9.69 hours -- by kylehamilton@ischool.berkeley.edu at 3/1/2020, 7:21:53 PM on KyleH

---

## Cluster 6.2 ML
KyleHW4Test    
Driver - i3xlarge 30.5GB Memory, 4 cores, 1 DBU   
Workers - min2:max8, i3xlarge 30.5GB Memory, 4 cores, 1 DBU   


### PageRank version 1
trained 10 iterations in 1316.5695507526398 seconds. (22 minutes)   


### PageRank version 2
trained 10 iterations in 774.6455962657928 seconds. (13 minutes)


### PageRank Graphframes
trained 10 iterations in 8.5 minutes

---

__BONUS QUESTION:__ Why do think the graphframes version takes so much longer on the smaller cluster (longer than the homegrown implementations)? HINT: What is Spark's default number of partitions?

