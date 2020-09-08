This repository contains the necessary code to run the model described in:<br>
https://arxiv.org/abs/2009.02715

<h3>DATA</h3>
The dataset used on this work and this repository is <a href="http://chalearnlap.cvc.uab.es/dataset/38/description/">CLOTH3D</a>, with associated <a href="https://arxiv.org/abs/1912.02792">paper</a>
<br>
Path to data has to be specified at 'values.py'. Note that it also asks for the path to preprocessings, described below.

<h4>PREPROCESSING</h4>
In order to optimize data pipeline, we preprocess template outfits. The code to train the model assumes the preprocessing is done.
To perform this preprocessing, check the scripts at 'DeePSD/Preprocessing/'.
<ol>
  <li><b>outfit.py</b> .OBJ files are encoded as ASCII. To increase efficiency, we store vertex locations as float16 binary format and faces as int16 in binary format. It also creates an 'outfit.txt' with garments listed in order. Run this first.</li>
  <li><b>edges.py</b> Precomputes list of edges as [v0, v1] as int16 in binary format.</li>
  <li><b>laplacians.py</b> Precomputes laplacian matrices for each outfit.</li>
  <li><b>weights_prior.py</b> For the unsupervised approach. Precomputes blend weights labels by proximity to body in rest pose.</li>
</ol>

<h3>TRAIN</h3>
Once all preprocessings have been completed. Just run 'train.py' or 'train_unsupervised.py'.
