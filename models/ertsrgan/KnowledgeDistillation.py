import tensorflow as tf
from models.metrics import ssim_loss


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
        self.time = []

    def get_run_time(self):
      if(len(self.time)>0):
         return sum(self.time)/len(self.time)
      else:
         return -1

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        perc_loss_fn,
        alpha=0.1,
        beta=0.2,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.perc_loss_fn = perc_loss_fn
        self.alpha = alpha
        self.beta = beta

    @tf.function
    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)
                                   
            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            
            distillation_loss = self.distillation_loss_fn(teacher_predictions,student_predictions)
            #distillation_loss = self.distillation_loss_fn(ssim_loss(y,y),ssim_loss(y,student_predictions))

            teacher_predictions = tf.keras.layers.Concatenate()([teacher_predictions, teacher_predictions, teacher_predictions])
            student_predictions = tf.keras.layers.Concatenate()([student_predictions, student_predictions, student_predictions])
            y = tf.keras.layers.Concatenate()([y, y, y])
            perc_loss = self.perc_loss_fn(y, student_predictions)

            loss = (1 - (self.alpha + self.beta)) * student_loss + self.alpha * distillation_loss + self.beta * perc_loss
            

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss, "perceptual_loss": perc_loss}
        )
        return results

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        #student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        #results.update({"student_loss": student_loss})
        return results



