## Classifying Satellite Images - Troubleshooting

### References
https://github.com/developmentseed/label-maker
https://github.com/developmentseed/label-maker/blob/master/examples/nets/SageMaker_mx-lenet.ipynb
https://developmentseed.org/blog/2018/01/19/sagemaker-label-maker-case/

#### Building tippecanoe
**Error:**
No sqlite3.h
**Solution:**
```
sudo yum install sqlite-devel
```

**Error:**
None has no attribute read
Failure reason
```
AlgorithmError: uncaught exception during training: 'NoneType' object has no attribute 'read' Traceback (most recent call last): File "/usr/local/lib/python2.7/dist-packages/container_support/training.py", line 36, in start fw.train() File "/usr/local/lib/python2.7/dist-packages/mxnet_container/train.py", line 189, in train model = user_module.train(**kwargs_to_pass) File "/opt/ml/code/mx_lenet_sagemaker.py", line 104, in train train_iter, val_iter = prep_data(data_path) File "/opt/ml/code/mx_lenet_sagemaker.py", line 26, in prep_data data = np.load(find_file(data_path, 'data.npz')) File "/usr/local/lib/python2.7/dist-packages/numpy/lib/npyio.py", line 402, in load magic = fid.read(N) AttributeError: 'NoneType' object has no attribute 'read'
```

**Solution:**

```
# change
# def train(num_cpus, num_gpus, **kwargs):
# to
def train(channel_input_dirs,num_cpus, num_gpus, **kwargs):
```
and
```
# change
# train_iter, val_iter = prep_data(data_path)
# to
train_iter, val_iter = prep_data(channel_input_dirs['training'])
```

**Error:**
Training and validation accuracy bottom out at 60%

**Solution:**

```
# change
#optimizer_params={'learning_rate': 0.1},
# to
optimizer_params={'learning_rate': 0.03},
```

**Error:**
The object does not exist

**Solution:**
```
# change
# img_s3 = 'MXNnet_example/data/103922-57564-17.jpg'
# to
img_s3 = 'vietnam_building/tiles/103922-57724-17.jpg'
```
