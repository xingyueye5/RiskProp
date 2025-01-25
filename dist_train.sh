name=tap

current_datetime=$(date +"%Y%m%d_%H%M%S")
mkdir -p codes/$name/$current_datetime

# cp -r tap codes/$name/$current_datetime
find tap -name "*.py" -exec cp --parents {} codes/$name/$current_datetime/ \;

# cp -r configs codes/$name/$current_datetime
find configs -name $name.py -exec cp --parents {} codes/$name/$current_datetime/ \;

tools/dist_train.sh configs/$name.py 4