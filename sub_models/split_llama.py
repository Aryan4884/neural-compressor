"""
Purpose: Split the model 

1. Define the guard nodes  [node1, node2, ... (node10), node11, ...(node20), ...,(last_node)]
2. Create a new graph, add [node1, node2, ... (node10)]
3. Add needed initializers
4. Add needed input
5. Add output(be careful), the output should include the output of the last node


"""

import onnx
from onnx.checker import check_model
from onnx import shape_inference
model_dir = "/mnt/disk4/modelHub/llama-2-7-onnx/"
model_path = model_dir + 'decoder_model.onnx'
original_model = onnx.load(model_path, load_external_data=False)
node_lst = original_model.graph.node
import pdb; pdb.set_trace()

original_model = shape_inference.infer_shapes(original_model)


def get_value_info(model, name):
    for value_info in model.graph.value_info:
        if value_info.name == name:
            return value_info

# def get_node_by_name(node_name, model):
#     """
#     Purpose: Get node by name
#     """
#     for node in model.graph.node:
#         if node.name == node_name:
#             return node
# new_node_lst = [get_node_by_name(node_name, original_model) for node_name in sorted_nodes]
# import pdb; pdb.set_trace()


'/bert/encoder/layer.10/output/LayerNorm/Add_1'

def get_index_of_node_include_specific_string(node_lst, string):
    """
    Purpose: Get the index of node that include specific string
    """
    index_lst = []
    for index, node in enumerate(node_lst):
        # TODO 
        if index == 0:
            index_lst.append((-1, node.name))
        if string in node.name and "attention" not in node.name:
            index_lst.append((index, node.name))
        if index == len(node_lst) - 1:
            index_lst.append((index, node.name))
    return index_lst

# guard_string = "output/LayerNorm/Add_1" # For full transformer-block
# guard_string = "output/LayerNorm/Mul"
guard_string = "/mlp/down_proj/MatMul"
guard_node_index_lst = get_index_of_node_include_specific_string(node_lst, guard_string)
print(guard_node_index_lst)
print(len(guard_node_index_lst))
print(len(node_lst))

def group_nodes_by_guard_node_index_lst(node_lst, guard_node_index_lst):
    """
    Purpose: Group nodes by guard node index list
    """
    node_group_lst = []
    for i in range(len(guard_node_index_lst) - 1):
        sub_node_lst = []
        start = guard_node_index_lst[i][0] + 1
        end  = guard_node_index_lst[i+1][0] + 1
        sub_node_lst = node_lst[start:end]
        print(f"-"*20)
        print(f"Add {start} to {end} nodes to sub_node_lst {[n.name for n in sub_node_lst]}") 
        node_group_lst.append(sub_node_lst)
    return node_group_lst

sub_graph_lst = group_nodes_by_guard_node_index_lst(node_lst, guard_node_index_lst)

def get_input_by_name(input_name, original_model):
    """
    Purpose: Get input by name
    """
    for input in original_model.graph.input:
        if input.name == input_name:
            return input
    # for input in original_model.graph.output:
    #     if input.name == input_name:
    #         return input
    # for node in original_model.graph.node:
    #     for output in node.output:
    #         if output == input_name:
    #             return output


def get_output_by_name(output_name, original_model):
    """
    Purpose: Get output by name
    """
    for output in original_model.graph.output:
        if output.name == output_name:
            return output

        
def get_initializer_by_name(initializer_name, original_model):
    for initializer in original_model.graph.initializer:
        if initializer.name == initializer_name:
            return initializer


def get_sub_graph_inputs(sub_graph_nodes, sub_graph_initializers_name_set, original_model):
    """
    Purpose: Get sub graph inputs
    """
    sub_graph_inputs = []
    sub_graph_inputs_name_set = set()
    for node in sub_graph_nodes:
        for input in node.input:
            if input not in sub_graph_inputs_name_set and input not in sub_graph_initializers_name_set:
                res = get_input_by_name(input, original_model)
                if res:
                    sub_graph_inputs.append(res)
                    sub_graph_inputs_name_set.add(input)
    return sub_graph_inputs, sub_graph_inputs_name_set

def get_sub_graph_outputs(sub_graph_nodes, original_model):
    """
    Purpose: Get sub graph outputs
    """
    sub_graph_outputs = []
    for node in sub_graph_nodes:
        for output in node.output:
            if output not in sub_graph_outputs:
                print(f"Add {output} to sub_graph_outputs")
                res = get_output_by_name(output, original_model)
                if res:
                    sub_graph_outputs.append(res)
    return sub_graph_outputs

def get_sub_graph_initializers(sub_graph_nodes, original_model):
    """
    Purpose: Get sub graph initializers
    """
    sub_graph_initializers = []
    sub_graph_initializers_name_set = set()
    # import pdb; pdb.set_trace()
    for node in sub_graph_nodes:
        # if node.op_type == "MatMul":
        #     import pdb; pdb.set_trace()
        for input in node.input:
            res = get_initializer_by_name(input, original_model)
            if res:
                print(f"Added input{input} as sub graph's initializer")
                sub_graph_initializers.append(res)
                sub_graph_initializers_name_set.add(input)
    return sub_graph_initializers, sub_graph_initializers_name_set


def creat_sub_model_based_on_sub_graph_nodes(sub_graph_nodes, original_model, sub_graph_name):
    """
    Purpose: Create a sub graph based on sub graph nodes
    """
    all_nodes_input_name = set()
    all_nodes_output_name = set()
    constant_output_name = set()
    for node in sub_graph_nodes:
        all_nodes_input_name.update({input_name for input_name in node.input})
        all_nodes_output_name.update({output_name for output_name in node.output})
    for node in original_model.graph.node:
        if node.op_type == "Constant":
            constant_output_name.update({output_name for output_name in node.output})
    

    

    # sub_graph_outputs = get_sub_graph_outputs(sub_graph_nodes, original_model)
    # print("*"*10)
    # print(f"sub_graph_outputs: {[o.name for o in sub_graph_outputs]}")
    sub_graph_initializers, sub_graph_initializers_name_set = get_sub_graph_initializers(sub_graph_nodes, original_model)
    print(f"*"*10)
    print(f"sub_graph_initializers: {[i.name for i in sub_graph_initializers]}")
    cleaned_input_name = all_nodes_input_name -  sub_graph_initializers_name_set - all_nodes_output_name
    print(f"constant_output_name: {constant_output_name}")
    # print(f"cleaned input name should not includes constant output name: {cleaned_input_name.intersection(constant_output_name)}")
    # cleaned_input_name = cleaned_input_name - constant_output_name

    # if one input is the output of constant node, add the constant node into graph and not add it as the model's inputs.
    # TODO current implementation is quite trick and redundant, should be improved
    for node in original_model.graph.node:
        if node.op_type == "Constant":
            for output_name in node.output:
                if output_name in cleaned_input_name:
                    cleaned_input_name.remove(output_name)
                    print(f"Remove {output_name} from cleaned_input_name as it's the output of a constant node")
                    if node not in sub_graph_nodes:
                        sub_graph_nodes.append(node)
                        print(f"Add {node.name} to sub_graph_nodes as it's the output of a constant node")
    
    already_exist_input_name = set(input_name for input_name in cleaned_input_name if get_input_by_name(input_name, original_model))
    already_exist_input = [get_input_by_name(input_name, original_model) for input_name in already_exist_input_name ]
    # TODO how to determine the shape of the middle input
    new_augment_input = []
    # import pdb; pdb.set_trace()
    for input_name in cleaned_input_name - already_exist_input_name:
        shape = ()
        # # TODO should find weight to release these hard code and support the dynamic shape inference
        # if input_name == "/bert/Mul_output_0":
        #     shape = (8, 1, 1, 128)
        # if "output/LayerNorm/Mul_output_0" in input_name: # bert/encoder/layer.0/output/LayerNorm/Mul_output_0
        #     shape = (8, 128, 768)
        # import pdb; pdb.set_trace()
        value_info = get_value_info(original_model, input_name)
        if value_info:
            # import pdb; pdb.set_trace()
            new_augment_input.append(value_info)
        else:
            print(f"Cannot find {input_name} in value_info, add it as a new input")
            import pdb; pdb.set_trace()
            new_input = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, shape=shape )
            new_augment_input.append(new_input)

    
    print(f"*"*10)
    print(f"sub_graph_name : {sub_graph_name}")
    print(f"sub_graph_inputs: {cleaned_input_name}")
    print(f"sub_graph already exist inputs: {already_exist_input_name}")
    print(f"sub_graph new added inputs: {cleaned_input_name - already_exist_input_name}")
    sub_graph_input[sub_graph_name]  = cleaned_input_name
    #sub_graph_nodes = topology_sorted_node(sub_graph_nodes)
    # The output is determined by the other sub_graph
    # import pdb; pdb.set_trace()
    sub_graph = onnx.helper.make_graph(
        nodes = sub_graph_nodes,
        name = "sub_graph",
        inputs = already_exist_input + new_augment_input,
        outputs = [],
        initializer = sub_graph_initializers
        
    )
    sub_model = onnx.helper.make_model(sub_graph, **{'opset_imports': [onnx.helper.make_opsetid('', 14)]})
    # import pdb; pdb.set_trace()

    sub_model.ir_version = original_model.ir_version
    # from neural_compressor.model import Model
    # TODO the model graph nodes order should be updated, use a cleaner way to do it.
    # inc_model = Model(sub_model)
    # inc_model.topological_sort()
    # sub_model = inc_model._model

    # check_result = check_model(sub_model)
    # print(f"check_result: {check_result} for model {sub_graph_name}")
    return sub_model


import pdb; pdb.set_trace()
sub_graph_input = {} # key: sub_graph_name, value: sub_graph_input
sub_graph_output = {}

def output_in_other_sub_graph(output_name):
    return any([output_name in sub_graph_node_input_set for sub_graph_node_input_set in sub_graph_input.values()])

def update_sub_graph_nodes_output(sub_model,original_model):
    # the output is model's output
    # the output is the input of other sub-graph
    all_nodes_output_name = set()
    for node in sub_model.graph.node:
        all_nodes_output_name.update({output_name for output_name in node.output})
    sub_model_output = []
    sub_model_output_name = set()
    model_output_name_set = set(output.name for output in original_model.graph.output)
    for output_name in all_nodes_output_name:
        if output_name not in sub_model_output_name and output_name in model_output_name_set:
            sub_model_output.append(get_output_by_name(output_name, original_model))
            sub_model_output_name.add(output_name)
        if output_name not in sub_model_output_name and output_in_other_sub_graph(output_name):
            sub_model_output_name.add(output_name)
            # TODO remove the hard code for create the output tensor
            value_info = get_value_info(original_model, output_name)
            if value_info:
                sub_model_output.append(value_info)
            else:
                print(f"can not find {output_name} in value_info, add it as a new output")
                import pdb; pdb.set_trace()
                sub_model_output.append(onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, () ) )
    sub_model.graph.output.extend(sub_model_output)
    print(f"sub_model_output : {sub_model_output_name}")



for i, sub_graph in enumerate(sub_graph_lst[::-1]):
    sub_graph_name = f"sub_graph_{i}"
    # check_result = check_model(original_model)
    # print(f"check_result: {check_result} for original ===========")
    sub_model = creat_sub_model_based_on_sub_graph_nodes(sub_graph, original_model, sub_graph_name)
    update_sub_graph_nodes_output(sub_model, original_model)
    onnx.save(sub_model, f"llama2_with_out_with_input_less_middle_sub_model_{i}.onnx")
    print(f"Save sub_model_{i}.onnx")


# Added the guard node's output as model's output

