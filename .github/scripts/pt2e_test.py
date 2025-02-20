import os
import sys
import subprocess

def pt2e_test(dt,scenario):
    models = ["alexnet",
              "demucs",
              "dlrm",
              "hf_Albert",
              "hf_Bert",
              "hf_Bert_large",
              "hf_DistilBert",
              "hf_Roberta_base",
              "mnasnet1_0",
              "mobilenet_v2",
              "mobilenet_v3_large",
              "nvidia_deeprecommender",
              "pytorch_CycleGAN_and_pix2pix",
              "resnet152",
              "resnet18",
              "resnet50",
              "resnext50_32x4d",
              "shufflenet_v2_x1_0",
              "squeezenet1_1",
              "Super_SloMo",
              "timm_efficientnet",
              "timm_nfnet",
              "timm_regnet",
              "timm_resnest",
              "timm_vision_transformer",
              "timm_vision_transformer_large",
              "timm_vovnet",
              "vgg16"]

    if scenario == "accuracy" and scenario != " ":
        # cd inductor-tools/scripts/modelbench/quant/inductor_quant_acc.py
        os.chdir("inductor-tools/scripts/modelbench/quant")
        cmd = 'python inductor_quant_acc.py'
        #subprocess.run(['python', 'inductor_quant_acc.py'])
        try:
            subprocess.run(cmd, shell=True,check=True)
        except subprocess.CalledProcessError as e:
            print('cmd {cmd} error:{e}')
    elif scenario == "performance" and scenario != " ":
        # cd benchmark/run_benchmark.py
        os.chdir("benchmark")
        if dt == "INT8":
            for model in models:
                cmd = 'python run_benchmark.py xpu --test eval --channels-last --metrics throughputs --torchdynamo inductor --quantization pt2e -m %s' %model
                try:
                    subprocess.run(cmd,shell=True,check=True)
                except subprocess.CalledProcessError as e:
                    print ('error:{e}')
                #subprocess.run(['python', 'run_benchmark.py', 'xpu', '--test', 'eval', '--channels-last', '--metrics', 'throughputs', '--torchdynamo', 'inductor', '--quantization', 'pt2e', '-m', model ])
        if dt == "FP32":
            for model in models:
                cmd = 'python run_benchmark.py xpu --test eval --channels-last --metrics throughputs --torchdynamo inductor -m %s' %model
                try:
                    subprocess.run(cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print ('error:{e}')
                #subprocess.run(['python', 'run_benchmark.py', 'xpu', '--test', 'eval', '--channels-last', '--metrics',  'throughputs', '--torchdynamo', 'inductor', '-m', model])

if __name__ == '__main__':
    pt2e_test(sys.argv[1],sys.argv[2])