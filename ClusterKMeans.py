import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score,adjusted_rand_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("customer_clusters.csv")
