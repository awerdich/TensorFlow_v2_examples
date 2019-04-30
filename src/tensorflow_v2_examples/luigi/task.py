import os
from hashlib import sha256
from luigi import LocalTarget
from luigi.task import flatten

def get_salted_version(task):
    """Create a salted id/version for this task and lineage
    :returns: a unique, deterministic hexdigest for this task
    :rtype: str
    """

    msg = ""
    for req in flatten(task.requires()):
        msg += get_salted_version(req)

    # Uniquely specify this task
    msg += ','.join([task.__class__.__name__, task.__version__,
                    ] + ['{}={}'.format(param_name, repr(task.param_kwargs[param_name]))
                        for param_name, param in sorted(task.get_params())
                        if param.significant
                    ]
                    )
    return sha256(msg.encode()).hexdigest()


# This will later be move to pset_utils.luigi.task
class TargetOutput:
    """Composition to replace :meth:`luigi.task.Task.output` and
    use a common file pattern.
    Example:
        class MyTask(Task):
        # Replace task.output
        output = TargetOutput(file_pattern='{task.__class__.__name__}-{task.date}',
                             ext='.tsv',
                             data_root='data/stream')"""
    
    
    def __init__(self, file_pattern='{task.__class__.__name__}',
                 data_root = 'data',
                 ext = '.txt',
                 target_class=LocalTarget, **target_kwargs):

        self.file_pattern = file_pattern
        self.data_root = data_root
        self.ext = ext
        self.target_class = target_class
        self.target_kwargs = target_kwargs

    def __get__(self, task, cls):
        if task is None:
            return self
        return lambda: self(task)

    def __call__(self, task):
        # Determine the path etc here
        file_name = self.file_pattern.format(task = task) + self.ext
        file_path = os.path.join(self.data_root, file_name)
        return self.target_class(file_path, **self.target_kwargs)

class SaltedOutput(TargetOutput):
    """Composition to replace :meth:`luigi.task.Task.output` and
        use a common file pattern and includes a salt.
        Example:
            class MyTask(Task):
            # Replace task.output
            output = TargetOutput(file_pattern='{task.__class__.__name__}-{task.date}',
                                 ext='.tsv',
                                 data_root='data/stream')"""

    def __init__(self, file_pattern='{task.__class__.__name__}-{salt}',
                 data_root = 'data',
                 ext = '.txt',
                 target_class = LocalTarget, **target_kwargs):

        super().__init__(file_pattern, data_root, ext, target_class)
        self.target_kwargs = target_kwargs

    # We just need to override the call function because the file name is different
    def __call__(self, task):
        # Determine the path etc here
        file_name = self.file_pattern.format(task=task, salt=get_salted_version(task)[:6]) + self.ext
        file_path = os.path.join(self.data_root, file_name)
        return self.target_class(file_path, **self.target_kwargs)

class Requirement:
    """Clones the target task: Instances of this class are the dependent tasks"""
    def __init__(self, task_class, **params):

        self.task_class = task_class
        self.params = params

    def __get__(self, task, cls):
        if task is None:
            return self

        return task.clone(self.task_class, **self.params)

class Requires:
    """Composition to replace :meth:`luigi.task.Task.requires`

    Example::

        class MyTask(Task):
            # Replace task.requires()
            requires = Requires()
            other = Requirement(OtherTask)

            def run(self):
                # Convenient access here...
                with self.other.output().open('r') as f:
                    ...

        >>> MyTask().requires()
        {'other': OtherTask()}

    """

    def __get__(self, task, cls):
        if task is None:
            return self

        # Bind self/task in a closure
        return lambda : self(task)

    def __call__(self, task):
        """Returns the requirements of a task

        Assumes the task class has :class:`.Requirement` descriptors, which
        can clone the appropriate dependencies from the task instance.

        :returns: requirements compatible with `task.requires()`
        :rtype: dict
        """
        # Search task.__class__ for Requirement instances
        # return {t: self.clone(t) for t in [other1, other2]}

        # Find the instance names of the Requirement class
        task_dict = task.__class__.__dict__
        req_instances = [key for key in task_dict.keys() if isinstance(task_dict[key], Requirement)]

        # Build a dictionary with the requirements
        return {t: getattr(task, t) for t in req_instances}
