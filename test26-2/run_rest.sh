cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
$P experiments/model_f_eval.py --ckpt best_model_f_seed0.pt best_model_f_nohk_seed0.pt best_model_f_nodg_seed0.pt best_model_f_synonly_seed0.pt 2>&1 | tee test26-2/e11.log
echo "E11 exit ${PIPESTATUS[0]}"
$P experiments/cross_dataset.py 2>&1 | tee test26-2/e9_looped.log
echo "E9 exit ${PIPESTATUS[0]}"
$P experiments/cross_dataset.py --ckpt best_model_d.pt 2>&1 | tee test26-2/e9_looped_model_d.log
echo "E9-D exit ${PIPESTATUS[0]}"
