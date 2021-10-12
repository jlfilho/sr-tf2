import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self,inputs=None, outputs=None,file_writer_cm=None,name=None):
        super(Model, self).__init__(inputs=inputs,outputs=outputs,name=name)
        self.file_writer_cm = file_writer_cm
        self.step = 0

    def train_step(self, data):
        if isinstance(data, tuple):
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        grads_vars_final = []
        step = self.step
        for gradient, variable in zip(gradients, trainable_vars):
            assert gradient is not None, variable.name

            variable_name = variable.name.replace(':', '_')
            scope = 'TrainLogs/' + variable_name + '/Values/'
            with self.file_writer_cm.as_default():
                tf.summary.scalar(scope + 'MIN', tf.reduce_min(variable),step=step)
                #tf.summary.scalar(scope + 'MAX', tf.reduce_max(variable),step=step)
                #tf.summary.scalar(scope + 'L2', tf.norm(variable),step=step)
                #tf.summary.scalar(scope + 'AVG', tf.reduce_mean(variable),step=step)

                scope = 'TrainLogs/' + variable_name + '/Gradients/'
                tf.summary.scalar(scope + 'MIN', tf.reduce_min(gradient),step=step)
                #tf.summary.scalar(scope + 'MAX', tf.reduce_max(gradient),step=step)
                #tf.summary.scalar(scope + 'L2', tf.norm(gradient),step=step)
                #tf.summary.scalar(scope + 'AVG', tf.reduce_mean(gradient),step=step)
            #if variable.name == 'final':
                #gradient *= 0.1
            grads_vars_final.append((gradient, variable))
            step+=1
            self.step=step

        # Update weights
        self.optimizer.apply_gradients(grads_vars_final)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}