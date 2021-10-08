import tensorflow as tf

class Model(tf.keras.Model):

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
        for gradient, variable in zip(gradients, trainable_vars):
            assert gradient is not None, variable.name

            if variable.name == 'final':
                gradient *= 0.1
            grads_vars_final.append((gradient, variable))

        # Update weights
        self.optimizer.apply_gradients(grads_vars_final)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}