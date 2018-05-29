echo '########################################################################################################################################################################################################'

echo 'MS dataset with 20K samples.'
echo 'Model - inception_resnet_v2'
echo 'Number of clones - 4' 

echo '########################################################################################################################################################################################################'

if [ "$TFFACE_ROOT_DIR" == "" ]; then
	export TFFACE_ROOT_DIR=/git-space
fi

echo 'Check TFFACE_ROOT_DIR, current TFFACE_ROOT_DIR is - '$TFFACE_ROOT_DIR

echo '########################################################################################################################################################################################################'

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$TFFACE_ROOT_DIR/tfmtcnn:$PYTHONPATH
export PYTHONPATH=$TFFACE_ROOT_DIR/tfface:$PYTHONPATH
export PYTHONPATH=$TFFACE_ROOT_DIR/slim-generic-dataset/research/slim:$PYTHONPATH

export DATASET_DIR=/tensorflow/datasets/MS_20K
export TRAIN_DIR=/tensorflow/models/inception_resnet_v2/MS_20K

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd $TFFACE_ROOT_DIR/slim-generic-dataset/research/slim

echo '########################################################################################################################################################################################################'

echo 'Create the dataset.'

python download_and_convert_data.py --dataset_name=generic --dataset_dir="${DATASET_DIR}" --source_dir=/datasets/mtcnn_images/MS_20K

echo 'Dataset is created.'

echo '########################################################################################################################################################################################################'

echo '--max_number_of_steps=150000 --learning_rate=0.01 --batch_size=40'

export CHECKPOINT_PATH=/tensorflow/models/tensorflow-slim/inception_resnet_v2/inception_resnet_v2.ckpt

python train_image_classifier.py --train_dir=${TRAIN_DIR}/01 --dataset_dir=${DATASET_DIR} --dataset_name=generic --dataset_split_name=train --num_clones=4 --model_name=inception_resnet_v2 --checkpoint_path=${CHECKPOINT_PATH} --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits --max_number_of_steps=150000 --batch_size=40 --learning_rate=0.01 --learning_rate_decay_type=fixed --save_interval_secs=1800 --save_summaries_secs=1800 --log_every_n_steps=1000 --optimizer=rmsprop --weight_decay=0.00004

echo 'Evaluate performance of the network.'

python eval_image_classifier.py --alsologtostderr --checkpoint_path=${TRAIN_DIR}/01 --dataset_dir=${DATASET_DIR} --dataset_name=generic --dataset_split_name=validation --model_name=inception_resnet_v2

echo '########################################################################################################################################################################################################'

echo '--max_number_of_steps=150000 --learning_rate=0.001 --batch_size=40'

export CHECKPOINT_PATH=${TRAIN_DIR}/01

python train_image_classifier.py --train_dir=${TRAIN_DIR}/02 --dataset_dir=${DATASET_DIR} --dataset_name=generic --dataset_split_name=train --num_clones=4 --model_name=inception_resnet_v2 --checkpoint_path=${CHECKPOINT_PATH} --max_number_of_steps=150000 --batch_size=40 --learning_rate=0.0001 --learning_rate_decay_type=fixed --save_interval_secs=1800 --save_summaries_secs=1800 --log_every_n_steps=1000 --optimizer=rmsprop --weight_decay=0.00004

echo 'Evaluate performance of the network.'

python eval_image_classifier.py --alsologtostderr --checkpoint_path=${TRAIN_DIR}/02 --dataset_dir=${DATASET_DIR} --dataset_name=generic --dataset_split_name=validation --model_name=inception_resnet_v2

echo '########################################################################################################################################################################################################'

#4) Run the TensorBoard.

#4.1) Run the TensorBoard server.  

#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
#tensorboard --logdir=/tensorflow/models/inception_resnet_v2/MS_20K --port 6007 &


#4.2) Run the FireFox.  
#firefox http://localhost:6007/ &

echo '########################################################################################################################################################################################################'
