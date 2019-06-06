from pyspark.ml import Pipeline, PipelineModel, Estimator, Transformer


# Extended ML Pipelines
#  callback function to refer to pre-stage attributes
class CallbackPipeline(Pipeline):
    def __init__(self, stages=None, callbacks=None):
        super(CallbackPipeline, self).__init__(stages=stages)
        self.callbacks = []
        self.callbacks.extend(callbacks)

    def _fit(self, dataset):
        stages = self.getStages()
        for stage in stages:
            if not (isinstance(stage, Estimator) or isinstance(stage, Transformer)):
                raise TypeError(
                    "Cannot recognize a pipeline stage of type %s." % type(stage))
        indexOfLastEstimator = -1
        for i, stage in enumerate(stages):
            if isinstance(stage, Estimator):
                indexOfLastEstimator = i
        transformers = []
        for i, stage in enumerate(stages):
            if len(self.callbacks) > i and self.callbacks[i]:
                self.callbacks[i](stage, transformers)
            if i <= indexOfLastEstimator:
                if isinstance(stage, Transformer):
                    transformers.append(stage)
                    dataset = stage.transform(dataset)
                else:  # must be an Estimator
                    model = stage.fit(dataset)
                    transformers.append(model)
                    if i < indexOfLastEstimator:
                        dataset = model.transform(dataset)
            else:
                transformers.append(stage)
        return PipelineModel(transformers)
