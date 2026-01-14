name=predict_anomaly_snippet

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 PORT=29501 tools/dist_test.sh configs/$name.py work_dirs/predict_anomaly_snippet/权重保存/ann_constraint_epoch_4.pth 7