python train_disease_model.py --task_name='anxiety' --batch_size=32 --epochs=10 --gpu_id='0' --do_train --lr 1e-3
python train_disease_model.py --task_name='bipolar' --batch_size=32 --epochs=10 --gpu_id='0' --do_train --lr 1e-3

python train_disease_model.py --task_name='anxiety' --batch_size=32 --epochs=10 --gpu_id='0' --do_train --model_name_or_path="roberta-base" --lr 1e-3
python train_disease_model.py --task_name='bipolar' --batch_size=32 --epochs=10 --gpu_id='0' --do_train --model_name_or_path="roberta-base" --lr 1e-3