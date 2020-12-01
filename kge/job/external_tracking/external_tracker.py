
from kge.config import Config, Configurable
from kge.job import TrainingOrEvaluationJob


class ExternalTracker(Configurable):

    @staticmethod
    def create(config: Config, job: TrainingOrEvaluationJob):
        track_types = config.get("job.external_tracking")

        for track_type in track_types:
            if track_type == "weights-and-biases":
                from kge.job.external_tracking import WeightsAndBiases
                return WeightsAndBiases(config, job)
            else:
                raise ValueError("job.external_tracking.type")

    def __init__(self, config: Config, tracker_name: str, job: TrainingOrEvaluationJob):
        super().__init__(config, tracker_name)

        self._initialize(job)

        job.post_batch_hooks.append(self._log)
        job.post_run_hooks.append(self._finalize)

    def _initialize(self, job):
        raise NotImplementedError()

    def _log(self, job):
        """ All relevant info should be available in `job.current_trace`. """
        raise NotImplementedError()

    def _finalize(self, job):
        pass
        

