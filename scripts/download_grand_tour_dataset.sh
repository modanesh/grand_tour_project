# read ./data/config.json
# for each entry, download the dataset

download_mission() {
    local timestamp=$1
    local mission_name=$2
    local output_dir="./data/${mission_name}"
    
    # Create output directory
    mkdir -p "${output_dir}"
    
    # download command_twist
    curl -L "https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/resolve/main/${timestamp}/data/anymal_command_twist.tar" -o anymal_command_twist.tar
    # download state_actuator
    curl -L "https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/resolve/main/${timestamp}/data/anymal_state_actuator.tar" -o anymal_state_actuator.tar
    # download state_odometry
    curl -L "https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/resolve/main/${timestamp}/data/anymal_state_odometry.tar" -o anymal_state_odometry.tar
    # tar -xvf and move to ./data/{MISSION_NAME}
    tar -xf anymal_command_twist.tar -C "${output_dir}"
    tar -xf anymal_state_actuator.tar -C "${output_dir}"
    tar -xf anymal_state_odometry.tar -C "${output_dir}"
    # remove .tar files
    rm anymal_command_twist.tar
    rm anymal_state_actuator.tar
    rm anymal_state_odometry.tar
}


NUM_MISSIONS=$(jq 'keys | length' ./data/config.json)

# Get mission names (keys) and iterate
i=0
for mission_name in $(jq -r 'keys[]' ./data/config.json); do
    timestamp=$(jq -r ".[\"${mission_name}\"].timestamp_label" ./data/config.json)
    download_mission "$timestamp" "$mission_name"
    i=$((i+1))
    echo "[${i}/${NUM_MISSIONS}] Downloaded mission: $mission_name"
done
