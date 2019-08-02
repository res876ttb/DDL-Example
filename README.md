# DDL-Example

This repo shows some examples of how to perform distributed deep learning.

[TOC]

## Traditional distributed Tensorflow

Code: `mnist_distributed.py`

Discription: This code shows how to use low level of tensorflow python API to perform distributed training.

Run: 
`python3 ./mnist_distributed.py $ps $worker $task $index`
where ps is the list of parameter servers, worker is the list of workers, task is the type of job(ps or worker), index is the index of the job. Note that we have use this command to launch all the workers and parameter servers.

## Tensorflow with estimator

Description: codes here show tensorflow estimator code with 4 different kinds of distributed strategy.

1. Estimator without any distributed strategy
   * Code: `mnist_estimator.py`
   * Run:
  `python3 ./mnist_estimator.py`
2. MirroredStrategy
   - Code: `mnist_estimator_distribute_mirrored.py`
   - Run: 
   `python3 ./mnist_estimator_distribute_mirrored.py --numgpus=$NUMGPUS`
   where `$NUMGPUS` is the number of GPUs within a node we want to use.
3. CentralStorageStrategy
   - Code: `mnist_estimator_distribute_central.py`
   - Run:
   `python3 ./mnist_estimator_distribute_central.py`
   After running this command, it will take all GPUs for computation and CPU for gradients migration.
4. MultiWorkerMirroredStrategy
   - Code: mnist_estimator_distribute_multiworker.py
   - Run:
   `python3 ./mnist_estimator_distribute_multiworker.py --worker=$WORKER $jobname $index`
   where `$WORKER` is the list of worker, `$jobname` could only be worker, and `$index` is the index of current worker in the worker list.
5. ParameterServerStrategy
   - Code: `mnist_estimator_distribute_ps.py`
   - Run:
   `python3 ./mnist_estimator_distribute_ps.py --ps=$PS --worker=$WORKER $jobname $index`
   where `$PS` is the list of parameter servers, `$WORKER` is the list of workers, `$jobname` is the task name, which could only be worker or ps, and `$index` is the index of current job in the worker list or parameter server list.

## Keras

Description: codes here show how to use keras to perform multi-GPU and multi-machine traning.

1. Original keras code
   * Code: `mnist_keras.py`
   * Run:
     `python3 ./mnist_keras.py`
2. Multi-GPU
   * Code: `mnist_keras_distribute.py`
   * Run:
     `python3 ./mnist_keras_distribute.py --numgpus=$NUMGPUS`
     where `$NUMGPUS` is the number of GPUs. This code will do a multi-GPU training within a node.
3. Multi-machine
   * Code: `mnist_keras_distribute_ps.py`
   * Run:
     `python3 ./mnist_keras_distribute_ps.py --ps=$PS --worker=$WORKER $jobname $index`
     where `$PS` is the list of parameter server, `$WORKER` is the list of worker, `$jobname` is the task name, which could only be worker or ps, and `$index` is the index of current job in the worker list or parameter server list.

## Horovod

Description: codes here show how to run distributed with horovod.

1. Original tensorflow with horovod
   * Code: `mnist_horovod.py`
   * Run: 
     `mpirun -x PATH -x LD_LIBRARY_PATH -x NCCL_DEBUG=INFO --bind-to none -mca btl_openib_allow_ib 1 -mca btl openib $MPICONFIG python3 ./mnist_horovod.py` 
     where `$MPICONFIG` is the configure of MPI process. 
2. Keras with horovod
   * Code: `mnist_keras_distribute_horovod.py`
   * Run:
     `mpirun -x PATH -x LD_LIBRARY_PATH -x NCCL_DEBUG=INFO --bind-to none -mca btl_openib_allow_ib 1 -mca btl openib $MPICONFIG python3 ./mnist_keras_distribute_horovod.py`
     where `$MPICONFIG` is the configure of MPI process.