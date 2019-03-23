'''Mar 23'''
import tensorflow as tf

def main(model_path):
    saver = tf.train.import_meta_graph(model_path + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    sess = tf.Session()
    saver.restore(sess, model_path)
    
    output_node_names="y_pred"
    output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session
                input_graph_def, # input_graph_def is useful for retrieving the nodes 
                output_node_names.split(",")  
    )
    
    output_graph= model_path + ".pb"
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    sess.close()

if __name__ == '__main__':
    model_path = "./checkpoints/train/2019-3-23-19-39-4/model-7000"
    main(model_path)