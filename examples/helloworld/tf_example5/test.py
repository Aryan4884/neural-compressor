import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
def main():

    import lpot
    quantizer = lpot.Quantization('./conf.yaml')
    model_path = "./mobilenet_v1_1.0_224_frozen.pb"
    quantized_model = quantizer(model_path)

     # Optional, run benchmark 
    from lpot import Benchmark
    evaluator = Benchmark('./conf.yaml')
    results = evaluator(quantized_model)
    batch_size = 1
    for mode, result in results.items():
       acc, batch_size, result_list = result
       latency = np.array(result_list).mean() / batch_size

       print('Accuracy is {:.3f}'.format(acc))
       print('Latency: {:.3f} ms'.format(latency * 1000))
      
if __name__ == "__main__":

    main()
