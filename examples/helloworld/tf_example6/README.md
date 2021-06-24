tf_example6 example
=====================
This example is used to demonstrate how to use default user-facing APIs to quantize a model.

1. Download the FP32 model
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb

2. Update the root of dataset in conf.yaml
The configuration will will create a TopK metric function for evaluation and configure the batch size, instance number and core number for performance measurement.    
```yaml
evaluation:                                          # optional. required if user doesn't provide eval_func in Quantization.
 accuracy:                                           # optional. required if user doesn't provide eval_func in Quantization.
    metric:
      topk: 1                                        # built-in metrics are topk, map, f1, allow user to register new metric.
    dataloader:
      batch_size: 32 
      dataset:
        ImageRecord:
          root: /path/to/imagenet/                   # NOTE: modify to evaluation dataset location if needed
      transform:
        BilinearImagenet: 
          height: 224
          width: 224

 performance:                                        # optional. used to benchmark performance of passing model.
    configs:
      cores_per_instance: 4
      num_of_instance: 7
    dataloader:
      batch_size: 1 
      last_batch: discard 
      dataset:
        ImageRecord:
          root: /path/to/imagenet/                   # NOTE: modify to evaluation dataset location if needed
      transform:
        ResizeCropImagenet: 
          height: 224
          width: 224
          mean_value: [123.68, 116.78, 103.94]

```

3. Run quantization
We only need to add the following lines for quantization to create an int8 model.
```python
    from lpot import Quantization
    quantizer = Quantization('./conf.yaml')
    quantized_model = quantizer('./mobilenet_v1_1.0_224_frozen.pb')
    tf.io.write_graph(graph_or_graph_def=quantized_model,
                      logdir='./',
                      name='int8.pb',
                      as_text=False)
```
* Run quantization and evaluation:
```shell
    python test.py --tune
``` 

4. Run benchmark according to config
```python
     # Optional, run benchmark 
    from lpot import Benchmark
    evaluator = Benchmark('./conf.yaml')
    results = evaluator('./int8.pb')
 
```

* Run benchmark and please make sure benchmark should after tuning:
```shell
    python test.py --benchmark
``` 
