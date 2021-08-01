#### Pull the docker

```bash
singularity pull docker://springtonyzhao/nvidia_dlrm_pyt:1.2.3
```

#### Run the train job

Use Sbatch to submit

```bash
cd $SCRATCH
sbatch trainJob
```

trainJob:

```
export CNUM=12 # use how many cards
export MHOST=`scontrol show hostname`
MHOST=($MHOST)

scontrol show hostname > hostfile
echo "
mpirun -n $CNUM -ppn 3 -f hostfile singularity exec --bind /opt/apps:/opt/apps,/scratch1/07519/ziheng/Adalars:/workspace/dlrm,/scratch1/07519/ziheng/data_out:/data --nv /scratch1/07519/ziheng/nvidia_dlrm_pyt_1.2.3.sif python3 -m dlrm.scripts.dist_main --master_addr='`echo ${MHOST[0]}`' --master_port 12345 --mode train --dataset /data/dlrm/binary_dataset/split --seed 0 --epochs 1 --amp --log_path /scratch1/07519/ziheng/python_log/log.json " > command.sh

bash command.sh
```

其中（目前测出来是这样）

- `CNUM` 是进程数量。和 -n 后的值一致
- `ppn` 是node数量。和 -N 后的值一致

#### Change the Training Flags

Two methods. 

You can directly change the settings at  `dlrm/scripts/dist_main.py`

```python
# Training schedule flags
FLAGS.set_default("batch_size", 131072)
FLAGS.set_default("test_batch_size", 131072)
FLAGS.set_default("lr", 56.0)
FLAGS.set_default("warmup_factor", 0)
FLAGS.set_default("warmup_steps", 16000)
FLAGS.set_default("decay_steps", 20000)
FLAGS.set_default("decay_start_step", 16000)
FLAGS.set_default("decay_power", 2)
FLAGS.set_default("decay_end_lr", 0)
FLAGS.set_default("embedding_type", "joint_sparse")
```

But also use the `--` method to change the variable via the commands. Change the *trainJob* file in the front.

```
 --decay_start_step: Optimization step after which to start decaying the learning rate, if None will start decaying right after the
    warmup phase is completed
    (default: '64000')
    (an integer)
  --decay_steps: Polynomial learning rate decay steps. If equal to 0 will not do any decaying
    (default: '80000')
    (an integer)
  --embedding_dim: Dimensionality of embedding space for categorical features
    (default: '128')
    (an integer)
```

See `README.md`. for all the possible options.



