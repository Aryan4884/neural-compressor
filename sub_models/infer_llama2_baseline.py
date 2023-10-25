import onnx
# from onnx.checker import check_model
from onnx.external_data_helper import load_external_data_for_model
# sub_model_name = "./llama2_with_out_with_input_less_middle_sub_model_32.onnx"
# onnx_model = onnx.load(sub_model_name, load_external_data=False)
# initalizer_data_dir = "/mnt/disk4/modelHub/llama-2-7-onnx/"
# load_external_data_for_model(onnx_model, initalizer_data_dir)



import onnxruntime as ort

# ort.InferenceSession(onnx_model.SerializeToString() , providers=ort.get_available_providers())
# import pdb; pdb.set_trace()
# from neural_compressor.model import Model
# inc_model = Model(onnx_model)
# inc_model.topological_sort()
# sub_model = inc_model._model

# # check_result = check_model(sub_model)
import gc

import numpy as np
def get_input_sample_npz():
    sample_input_path = "/home/st_liu/workspace/projects/transformers/demo/llama_sample.npz"
    input_sample = np.load(sample_input_path)
    result = {}
    for key in input_sample.keys():
        result[key] = input_sample[key]
    return result


input_sample = get_input_sample_npz()

def optimize_model(model):
    from neural_compressor.model import Model
    inc_model = Model(model)
    inc_model.topological_sort()
    return inc_model._model

from onnxruntime.quantization.onnx_model import ONNXModel

def record_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.6f} seconds to run.")
        return result
    return wrapper


@record_time
def infer_model(model, input_sample, output_dict):
    session = ort.InferenceSession(path_or_bytes=model.SerializeToString(),providers=ort.get_available_providers())
    output_names = [out.name for out in model.graph.output]
    model_required_inputs_name_lst = [n.name for n in model.graph.input]
    # the model inputs come from two ways:
    #   the output of previous sub_model
    #   the global input
    input_feed = {}
    for input_name in model_required_inputs_name_lst:
        if input_name in input_sample:
            input_feed[input_name] = input_sample[input_name]
        elif input_name in output_dict:
            input_feed[input_name] = output_dict[input_name]
        else:
            raise Exception(f"input_name {input_name} is not in input_sample and output_dict")
    # import pdb; pdb.set_trace()
    print(f"*** Set the model inputs as {input_feed.keys()}")
    output = session.run(output_names=output_names, input_feed=input_feed)
    return output

    
output_dict = {}
model_idx_lst = range(32, -1, -1)
for model_idx in [0]:
    #sub_model = f"llama2_with_out_with_input_less_middle_sub_model_{model_idx}.onnx"
    #print(f"************ Start do inference with {model_idx} graph {sub_model}")
# for sub_model in ["with_out_with_input_less_middle_sub_model_12.onnx", "with_out_with_input_less_middle_sub_model_11.onnx"]:
    #model = onnx.load(sub_model)
    model_dir = "/mnt/disk4/modelHub/llama-2-7-onnx/"
    model_path = model_dir + 'decoder_model.onnx'
    model = onnx.load(model_path, load_external_data=False)
    print(f"load model without external data")
    load_external_data_for_model(model, model_dir)
    print(f"load external data for model")
    # ort_model = ONNXModel(model)
    # ort_model.topological_sort()
    import time
    start_time = time.time()
    # mdoel = optimize_model(model)
    # check_model(model)
    session = ort.InferenceSession(path_or_bytes=model_path, providers=ort.get_available_providers())
    print(f"Function took {time.time() - start_time:.6f} seconds to initialize the session.")
    output_names = [out.name for out in model.graph.output]
    model_required_inputs_name_lst = [n.name for n in model.graph.input]
    # the model inputs come from two ways:
    #   the output of previous sub_model
    #   the global input
    input_feed = {}
    for input_name in model_required_inputs_name_lst:
        if input_name in input_sample:
            input_feed[input_name] = input_sample[input_name]
        elif input_name in output_dict:
            input_feed[input_name] = output_dict[input_name]
        else:
            raise Exception(f"input_name {input_name} is not in input_sample and output_dict")
    # import pdb; pdb.set_trace()
    print(f"*** Set the model inputs as {input_feed.keys()}")
    output = session.run(output_names=output_names, input_feed=input_feed)
    end_time = time.time()
    print(f"Function '{model_path}' took {end_time - start_time:.6f} seconds to run.")
    output_dict = {name: value for name, value in zip(output_names, output)}
    input_sample.update(output_dict)
    # import pdb; pdb.set_trace()
    print(f"output name {[ (k, v.shape) for k, v in output_dict.items() ]}")
    del session
    gc.collect()
    input = output
import pdb; pdb.set_trace()
    