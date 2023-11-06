gpu_num=0
cd /opt/kwater/projects/detection

exp_name=resnet18

HYDRA_FULL_ERROR=1 python3 main.py \
    experiment_tool.experiment_name=kwater-chpark \
    experiment_tool.run_group=${exp_name} \
    experiment_tool.run_name=${exp_name} \
    trainer.gpus=${gpu_num} \
    trainer.fast_dev_run=False
