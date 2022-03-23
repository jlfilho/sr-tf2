import tensorflow as tf
import os
import argparse
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow.compat.v1 as tf1

from models.espcn.model_espcn import ESPCN as espcn 
from models.imdn.model_imdn import IMDN
from models.rtsrgan.model_generator import G_RTSRGAN as g_rtsrgan
from models.rtvsrgan.model_generator import G_RTVSRGAN as g_rtvsrgan
from models.evsrnet.model_evsrnet import EVSRNet




def get_arguments():
    parser = argparse.ArgumentParser(description='Generate binary model file')
    parser.add_argument('--model', type=str, default='g_rtvsrgan', choices=['espcn','imdn', 'g_rtsrgan', 'g_rtvsrgan','evsrnet'],
                        help='What model to use for generation')
    parser.add_argument('--output_folder', type=str, default='./dnn_bin_models/',
                        help='where to put generated files')
    parser.add_argument('--ckpt_path', default='./checkpoint/g_rtvsrgan/',
                        help='Path to the model checkpoint, from which weights are loaded')
    parser.add_argument('--scale_factor', type=int, default=2, choices=[2, 3, 4],
                        help='What scale factor was used for chosen model')
    parser.add_argument('--cluster', type=str, default='generic',
                        help='where to put generated files')
    parser.add_argument('--inter_method', type=str, default=None, choices=['bilinear','lanczos3','lanczos5','bicubic','nearest','mitchellcubic'],
                        help='Type of interpolation resize used of same models')         
    return parser.parse_args()


def change_input_shape(model,name):
    input = Input(batch_shape=(1,None,None,1),name="x")
    output = model(input)
    newModel = Model(input,output,name=name)
    return newModel

def write_model(model,model_name,output_folder,cluster):
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
    tf1.train.write_graph(output_graph_def, output_folder, model_name+'_'+cluster+'.pb', as_text=False)

def main():
    args = get_arguments()

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    if args.ckpt_path is None:
        print("Path to the checkpoint file was not provided")
        exit(1)

    if args.model == 'espcn':
        print("Model")
        model = espcn(scale_factor=args.scale_factor)
        print("Load weights")
        model.load_weights(args.ckpt_path+"model.ckpt")
        print("change_input_shape")
        model = change_input_shape(model,args.model)
        print("write_model")
        write_model(model,args.model,args.output_folder,args.cluster)
    elif args.model == 'imdn':
        model = IMDN(scale_factor=args.scale_factor)
        print(args.ckpt_path+"model.ckpt")
        model.load_weights(args.ckpt_path+"model.ckpt")
        print("change_input_shape")
        model = change_input_shape(model,args.model)
        print("write_model")
        write_model(model,args.model,args.output_folder,args.cluster)
    elif args.model == 'g_rtsrgan':
        model = g_rtsrgan(scale_factor=args.scale_factor)
        model.load_weights(args.ckpt_path+"model.ckpt")
        print("change_input_shape")
        model = change_input_shape(model,args.model)
        print("write_model")
        write_model(model,args.model,args.output_folder,args.cluster)
    elif args.model == 'g_rtvsrgan':
        model = g_rtvsrgan(scale_factor=args.scale_factor)
        model.load_weights(args.ckpt_path+"model.ckpt")
        print("change_input_shape")
        model = change_input_shape(model,args.model)
        print("write_model")
        write_model(model,args.model,args.output_folder,args.cluster)
    elif args.model == 'evsrnet':
        model = EVSRNet(scale_factor=args.scale_factor,method=args.inter_method)
        print(args.ckpt_path+"model.ckpt")
        model.load_weights(args.ckpt_path+"model.ckpt")
        print("change_input_shape")
        model = change_input_shape(model,args.model)
        print("write_model")
        write_model(model,args.model,args.output_folder,args.cluster)
    else:
        exit(1)

if __name__ == '__main__':
    main()


