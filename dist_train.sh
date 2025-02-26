name=taa_snippet

current_datetime=$(date +"%Y%m%d_%H%M%S")
mkdir -p codes/$name/$current_datetime

# cp -r taa codes/$name/$current_datetime
find taa -name "*.py" -exec cp --parents {} codes/$name/$current_datetime/ \;

# cp -r configs codes/$name/$current_datetime
find configs -name $name.py -exec cp --parents {} codes/$name/$current_datetime/ \;

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 tools/dist_train.sh configs/$name.py 4