import tensorflow as tf
import os
import argparse
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow.compat.v1 as tf1

from model_espcn import espcn 
from model_rtvsrgan import rtvsrgan


def get_arguments():
    parser = argparse.ArgumentParser(description='generate binary model file')
    parser.add_argument('--model', type=str, default='rtvsrgan', choices=['espcn', 'rtvsrgan'],
                        help='What model to use for generation')
    parser.add_argument('--output_folder', type=str, default='./dnn_bin_models/',
                        help='where to put generated files')
    parser.add_argument('--ckpt_path', default='./logdir/',
                        help='Path to the model checkpoint, from which weights are loaded')
    parser.add_argument('--scale_factor', type=int, default=2, choices=[2, 3, 4],
                        help='What scale factor was used for chosen model')
    return parser.parse_args()


def change_input_shape(model,name):
    input = Input(batch_shape=(1,None,None,1),name="x")
    output = model(input)
    newModel = Model(input,output,name=name)
    return newModel

def write_model(model,model_name,output_folder):
    # Convert Keras model to ConcreteFunction
    full_model = tf1.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf1.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype,name="x"))

    frozen_func = convert_variables_to_constants_v2(full_model)
    graph_def = frozen_func.graph.as_graph_def()

    for i,t in enumerate(graph_def.node):
        #if t.name == 'NoOp':
        #    aux=t 
        if t.name == 'Identity':
            graph_def.node[i].name='y'
            graph_def.node[i].input.pop()  
    #graph_def.node.remove(aux)
    #print(frozen_func.graph.outputs)
            
    tf1.reset_default_graph()
    with tf1.Session() as sess:   
        sess.graph.as_default()
        tf1.import_graph_def(graph_def,name='')        
        output_graph_def = tf1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['y'])
        output_graph_def = tf.compat.v1.graph_util.remove_training_nodes(output_graph_def,
        protected_nodes='y')
    tf1.train.write_graph(output_graph_def, output_folder, model_name+'.pb', as_text=False)

def main():
    args = get_arguments()

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    if args.ckpt_path is None:
        print("Path to the checkpoint file was not provided")
        exit(1)

    if args.model == 'rtvsrgan':
        model = rtvsrgan()
        model.load_weights(args.ckpt_path+"model.ckpt")
        print("change_input_shape")
        model = change_input_shape(model,args.model)
        print("write_model")
        write_model(model,args.model,args.output_folder)
    elif args.model == 'espcn':
        print("MOdel")
        model = espcn()
        print("Load weights")
        model.load_weights(args.ckpt_path+"model.ckpt")
        print("change_input_shape")
        model = change_input_shape(model,args.model)
        print("write_model")
        write_model(model,args.model,args.output_folder)
    else:
        exit(1)

if __name__ == '__main__':
    main()
