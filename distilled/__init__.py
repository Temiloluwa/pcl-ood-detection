from .hinton_distillation import distill_kd_main
from utils import load_yaml

CD = load_yaml("configs/distilled-kd-config.yml")
