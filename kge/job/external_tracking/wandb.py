
from kge.config import Config, Configurable

from kge.job.external_tracking import ExternalTracker
from kge.job import TrainingOrEvaluationJob

import wandb

class WeightsAndBiases(ExternalTracker):

    def __init__(self, config: Config, job: TrainingOrEvaluationJob):
        super().__init__(config, "job.external_tracking.weights-and-biases", job)

    def _initialize(self, job):
        self.job_type = job.config.get("job.type")
        
        if job.parent_job:
            return

        project = self.get_option("project")
        group = self.get_option("group")

        job_type = type(job).__name__
        job_id = job.job_id
        
        wandb.init(
            project = project,
            group = group,
            job_type = job_type,
            id = job_id,
            config = job.config.options,
        )

    def _log(self, job):
        to_log = {}
        for key, value in job.current_trace.items():
            to_log[f'{self.job_type}_{key}'] = value
        wandb.log(to_log)

    def _finalize(self, job, result):
        if not job.parent_job:
            wandb.finish()
